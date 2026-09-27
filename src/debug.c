#include "debug.h"

#include <stdint.h>
#include <inttypes.h>
#include <stdlib.h>
#include <math.h>

/* Temporary traversal storage has no ownership of tensor references. The
 * pointer set ensures both deep chains and shared DAGs run in linear time. */
typedef struct {
    const Tensor **nodes;
    size_t count, capacity;
    const Tensor **slots;
    size_t slot_capacity;
} Graph;

static size_t pointer_hash(const Tensor *t) {
    uintptr_t x = (uintptr_t)t >> 3;
    x ^= x >> (sizeof(uintptr_t) * 4);
    return (size_t)x;
}

static void graph_free(Graph *g) {
    free(g->nodes);
    free(g->slots);
}

/* Returns 1 on success (including an already visited tensor). */
static int graph_add(Graph *g, const Tensor *t) {
    if (g->slot_capacity) {
        size_t existing = pointer_hash(t) & (g->slot_capacity - 1);
        while (g->slots[existing]) {
            if (g->slots[existing] == t) return 1;
            existing = (existing + 1) & (g->slot_capacity - 1);
        }
    }
    if (g->slot_capacity == 0 || g->count >= g->slot_capacity / 2) {
        size_t capacity = g->slot_capacity ? g->slot_capacity * 2 : 64;
        if (capacity < g->slot_capacity || capacity > SIZE_MAX / sizeof(*g->slots)) return 0;
        const Tensor **slots = calloc(capacity, sizeof(*slots));
        if (!slots) return 0;
        for (size_t j = 0; j < g->slot_capacity; ++j) {
            const Tensor *old = g->slots[j];
            if (old) {
                size_t pos = pointer_hash(old) & (capacity - 1);
                while (slots[pos]) pos = (pos + 1) & (capacity - 1);
                slots[pos] = old;
            }
        }
        free(g->slots);
        g->slots = slots;
        g->slot_capacity = capacity;
    }
    size_t pos = pointer_hash(t) & (g->slot_capacity - 1);
    while (g->slots[pos]) {
        if (g->slots[pos] == t) return 1;
        pos = (pos + 1) & (g->slot_capacity - 1);
    }
    if (g->count == g->capacity) {
        size_t capacity = g->capacity ? g->capacity * 2 : 32;
        if (capacity < g->capacity || capacity > SIZE_MAX / sizeof(*g->nodes)) return 0;
        const Tensor **nodes = realloc(g->nodes, capacity * sizeof(*nodes));
        if (!nodes) return 0;
        g->nodes = nodes;
        g->capacity = capacity;
    }
    g->slots[pos] = t;
    g->nodes[g->count++] = t;
    return 1;
}

static int valid_node(const Tensor *t) {
    return t && t->size > 0 && t->data &&
           (t->ndim == 0 ? t->size == 1 :
            (t->ndim == 2 && t->shape && t->shape[0] > 0 && t->shape[1] > 0 &&
             (size_t)t->shape[0] * (size_t)t->shape[1] == (size_t)t->size)) &&
           t->n_parents >= 0 && (t->n_parents == 0 || t->parents);
}

static int graph_collect(Graph *g, const Tensor *root) {
    if (!valid_node(root) || !graph_add(g, root)) return 0;
    for (size_t i = 0; i < g->count; ++i) {
        const Tensor *node = g->nodes[i];
        if (!valid_node(node)) return 0;
        for (int j = 0; j < node->n_parents; ++j) {
            if (!valid_node(node->parents[j]) || !graph_add(g, node->parents[j])) return 0;
        }
    }
    return 1;
}

/* Use pointer values solely as local DOT identifiers, never as ownership or
 * persistent IDs. Graphviz labels are escaped separately. */
static int dot_id(FILE *file, const Tensor *t) {
    return fprintf(file, "n%" PRIxPTR, (uintptr_t)t) >= 0;
}

static int dot_label(FILE *file, const char *text) {
    for (const unsigned char *p = (const unsigned char *)text; *p; ++p) {
        if (*p == '"' || *p == '\\') {
            if (fputc('\\', file) == EOF || fputc(*p, file) == EOF) return 0;
        } else if (*p == '\n' || *p == '\r') {
            if (fputs("\\n", file) == EOF) return 0;
        } else if (*p < 32 || *p == 127) {
            if (fputc('?', file) == EOF) return 0;
        } else if (fputc(*p, file) == EOF) {
            return 0;
        }
    }
    return 1;
}

int tensor_dump_dot(const Tensor *root, const char *path) {
    if (!root || !path || !*path) return 0;
    Graph graph = {0};
    int ok = graph_collect(&graph, root);
    FILE *file = NULL;
    if (ok) {
        file = fopen(path, "w");
        if (!file) ok = 0;
    }
    if (ok && fputs("digraph tensors {\n", file) == EOF) ok = 0;
    for (size_t i = 0; ok && i < graph.count; ++i) {
        const Tensor *t = graph.nodes[i];
        ok = dot_id(file, t) && fputs(" [label=\"", file) != EOF &&
             dot_label(file, t->op_name ? t->op_name : "leaf");
        if (ok) {
            if (t->ndim == 0)
                ok = fprintf(file, "\\nshape=scalar\\nrequires_grad=%d\"];\n",
                             !!t->requires_grad) >= 0;
            else
                ok = fprintf(file, "\\nshape=(%d,%d)\\nrequires_grad=%d\"];\n",
                             t->shape[0], t->shape[1], !!t->requires_grad) >= 0;
        }
        for (int j = 0; ok && j < t->n_parents; ++j) {
            ok = dot_id(file, t->parents[j]) && fputs(" -> ", file) != EOF &&
                 dot_id(file, t) && fputs(";\n", file) != EOF;
        }
    }
    if (ok && fputs("}\n", file) == EOF) ok = 0;
    if (file && fclose(file) != 0) ok = 0;
    graph_free(&graph);
    return ok;
}

int tensor_check_finite(const Tensor *root, int check_gradients) {
    if (!root) return 0;
    Graph graph = {0};
    int ok = graph_collect(&graph, root);
    for (size_t i = 0; ok && i < graph.count; ++i) {
        const Tensor *t = graph.nodes[i];
        for (int j = 0; j < t->size; ++j) {
            if (!isfinite(t->data[j])) {
                fprintf(stderr, "minigrad: %s tensor %p data[%d] is nonfinite\n",
                        t->op_name ? t->op_name : "leaf", (const void *)t, j);
                ok = 0;
                break;
            }
            if (check_gradients && t->grad && !isfinite(t->grad[j])) {
                fprintf(stderr, "minigrad: %s tensor %p grad[%d] is nonfinite\n",
                        t->op_name ? t->op_name : "leaf", (const void *)t, j);
                ok = 0;
                break;
            }
        }
    }
    graph_free(&graph);
    return ok;
}

static void print_range(FILE *stream, const char *name, const float *data, int size) {
    if (!data || size <= 0) {
        fprintf(stream, "%s=none", name);
        return;
    }
    float low = 0, high = 0;
    int nfinite = 0, nan_count = 0, inf_count = 0;
    for (int i = 0; i < size; ++i) {
        float x = data[i];
        if (isnan(x)) ++nan_count;
        else if (isinf(x)) ++inf_count;
        else {
            if (!nfinite || x < low) low = x;
            if (!nfinite || x > high) high = x;
            ++nfinite;
        }
    }
    fprintf(stream, "%s=", name);
    if (nfinite) fprintf(stream, "[%g, %g]", (double)low, (double)high);
    else fputs("no finite values", stream);
    if (nan_count || inf_count) fprintf(stream, " (nan=%d inf=%d)", nan_count, inf_count);
}

void tensor_summary(const Tensor *tensor, FILE *stream) {
    if (!stream) stream = stdout;
    if (!tensor) {
        fputs("Tensor(NULL)\n", stream);
        return;
    }
    fprintf(stream, "Tensor(op=%s, shape=", tensor->op_name ? tensor->op_name : "leaf");
    if (tensor->ndim == 0) fputs("scalar", stream);
    else if (tensor->ndim == 2 && tensor->shape)
        fprintf(stream, "(%d,%d)", tensor->shape[0], tensor->shape[1]);
    else fputs("invalid", stream);
    fprintf(stream, ", requires_grad=%d, is_leaf=%d, ref_count=%d, ",
            !!tensor->requires_grad, !!tensor->is_leaf, tensor->ref_count);
    print_range(stream, "data", tensor->data, tensor->size);
    fputs(", ", stream);
    print_range(stream, "grad", tensor->grad, tensor->size);
    TensorMemoryStats stats = tensor_memory_stats();
    fprintf(stream, ", memory: live_tensors=%zu live_bytes=%zu peak_bytes=%zu)\n",
            stats.live_tensors, stats.live_bytes, stats.peak_bytes);
}
