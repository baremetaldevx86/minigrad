#include "engine_internal.h"
#include "kernels.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// ============================================================
// Internal allocation and validation helpers
// ============================================================

static void *checked_malloc(size_t size) {
    void *ptr = malloc(size == 0 ? 1 : size);
    if (!ptr) {
        fprintf(stderr, "minigrad: out of memory\n");
        exit(EXIT_FAILURE);
    }
    return ptr;
}

static void *checked_calloc(size_t count, size_t size) {
    if (size != 0 && count > SIZE_MAX / size) {
        fprintf(stderr, "minigrad: allocation size overflow\n");
        exit(EXIT_FAILURE);
    }

    void *ptr = calloc(count == 0 ? 1 : count, size == 0 ? 1 : size);
    if (!ptr) {
        fprintf(stderr, "minigrad: out of memory\n");
        exit(EXIT_FAILURE);
    }
    return ptr;
}

static void *checked_realloc(void *ptr, size_t size) {
    void *new_ptr = realloc(ptr, size == 0 ? 1 : size);
    if (!new_ptr) {
        fprintf(stderr, "minigrad: out of memory\n");
        exit(EXIT_FAILURE);
    }
    return new_ptr;
}

static int valid_matrix_size(int rows, int cols, int *size_out) {
    if (rows <= 0 || cols <= 0) return 0;

    if ((size_t)rows > SIZE_MAX / (size_t)cols) return 0;
    size_t size = (size_t)rows * (size_t)cols;
    if (size > (size_t)INT_MAX || size > SIZE_MAX / sizeof(float)) return 0;

    *size_out = (int)size;
    return 1;
}

static void report_error(const char *operation, const char *message) {
    fprintf(stderr, "minigrad: %s: %s\n", operation, message);
}

// Counters cover persistent engine-owned storage, not transient DFS buffers.
static TensorMemoryStats memory_stats;
static _Thread_local int gradient_enabled = 1;

static void memory_add(size_t bytes) {
    if (bytes > SIZE_MAX - memory_stats.live_bytes) {
        fprintf(stderr, "minigrad: memory counter overflow\n");
        exit(EXIT_FAILURE);
    }
    memory_stats.live_bytes += bytes;
    if (memory_stats.live_bytes > memory_stats.peak_bytes)
        memory_stats.peak_bytes = memory_stats.live_bytes;
}

static void memory_remove(size_t bytes) {
    memory_stats.live_bytes -= bytes;
}

TensorMemoryStats tensor_memory_stats(void) { return memory_stats; }

int grad_is_enabled(void) { return gradient_enabled; }

int grad_set_enabled(int enabled) {
    int previous = gradient_enabled;
    gradient_enabled = !!enabled;
    return previous;
}

// ============================================================
// Dynamic lists used by graph traversal and iterative release
// ============================================================

typedef struct {
    Tensor **items;
    int size;
    int capacity;
} TensorList;

typedef struct {
    Tensor *node;
    int next_parent;
} TopoFrame;

typedef struct {
    TopoFrame *items;
    int size;
    int capacity;
} FrameList;

// Open-addressed pointer set: graph traversal must not be quadratic on long
// chains or highly shared DAGs.
typedef struct {
    Tensor **slots;
    size_t count;
    size_t capacity;
} TensorSet;

static size_t tensor_hash(const Tensor *tensor) {
    uintptr_t value = (uintptr_t)tensor;
    value >>= 3;
    value ^= value >> (sizeof(uintptr_t) * CHAR_BIT / 2);
    return (size_t)value;
}

static void set_init(TensorSet *set) {
    set->capacity = 64;
    set->count = 0;
    set->slots = checked_calloc(set->capacity, sizeof(*set->slots));
}

static void set_insert_slot(Tensor **slots, size_t capacity, Tensor *tensor) {
    size_t index = tensor_hash(tensor) & (capacity - 1);
    while (slots[index]) index = (index + 1) & (capacity - 1);
    slots[index] = tensor;
}

// Returns 1 only when the tensor was not already present.
static int set_insert(TensorSet *set, Tensor *tensor) {
    if (set->count >= set->capacity / 2) {
        if (set->capacity > SIZE_MAX / 2 / sizeof(*set->slots)) {
            fprintf(stderr, "minigrad: graph is too large\n");
            exit(EXIT_FAILURE);
        }
        size_t new_capacity = set->capacity * 2;
        Tensor **new_slots = checked_calloc(new_capacity, sizeof(*new_slots));
        for (size_t i = 0; i < set->capacity; i++) {
            if (set->slots[i]) set_insert_slot(new_slots, new_capacity, set->slots[i]);
        }
        free(set->slots);
        set->slots = new_slots;
        set->capacity = new_capacity;
    }

    size_t index = tensor_hash(tensor) & (set->capacity - 1);
    while (set->slots[index]) {
        if (set->slots[index] == tensor) return 0;
        index = (index + 1) & (set->capacity - 1);
    }
    set->slots[index] = tensor;
    set->count++;
    return 1;
}

static void set_free(TensorSet *set) {
    free(set->slots);
}

static void list_init(TensorList *list) {
    list->size = 0;
    list->capacity = 32;
    list->items = checked_malloc(sizeof(Tensor *) * (size_t)list->capacity);
}

static void list_push(TensorList *list, Tensor *tensor) {
    if (list->size == list->capacity) {
        if (list->capacity > INT_MAX / 2 ||
            (size_t)list->capacity > SIZE_MAX / 2 / sizeof(*list->items)) {
            fprintf(stderr, "minigrad: graph is too large\n");
            exit(EXIT_FAILURE);
        }
        list->capacity *= 2;
        list->items = checked_realloc(
            list->items, sizeof(Tensor *) * (size_t)list->capacity);
    }
    list->items[list->size++] = tensor;
}

static Tensor *list_pop(TensorList *list) {
    return list->items[--list->size];
}

static void list_free(TensorList *list) {
    free(list->items);
    list->items = NULL;
    list->size = 0;
    list->capacity = 0;
}

static void frame_list_init(FrameList *list) {
    list->size = 0;
    list->capacity = 32;
    list->items = checked_malloc(sizeof(TopoFrame) * (size_t)list->capacity);
}

static void frame_list_push(FrameList *list, Tensor *node) {
    if (list->size == list->capacity) {
        if (list->capacity > INT_MAX / 2 ||
            (size_t)list->capacity > SIZE_MAX / 2 / sizeof(*list->items)) {
            fprintf(stderr, "minigrad: graph is too large\n");
            exit(EXIT_FAILURE);
        }
        list->capacity *= 2;
        list->items = checked_realloc(
            list->items, sizeof(TopoFrame) * (size_t)list->capacity);
    }

    list->items[list->size].node = node;
    list->items[list->size].next_parent = 0;
    list->size++;
}

static void frame_list_free(FrameList *list) {
    free(list->items);
    list->items = NULL;
    list->size = 0;
    list->capacity = 0;
}

// Iterative DFS avoids overflowing the C call stack on deep computation graphs.
static void build_topo(Tensor *root, TensorList *topo) {
    TensorSet visited;
    FrameList stack;
    set_init(&visited);
    frame_list_init(&stack);

    set_insert(&visited, root);
    frame_list_push(&stack, root);

    while (stack.size > 0) {
        TopoFrame *frame = &stack.items[stack.size - 1];
        Tensor *node = frame->node;

        if (frame->next_parent < node->n_parents) {
            Tensor *parent = node->parents[frame->next_parent++];
            if (!parent) {
                report_error("tensor_backward", "encountered a NULL parent");
                continue;
            }

            if (set_insert(&visited, parent)) frame_list_push(&stack, parent);
        } else {
            list_push(topo, node);
            stack.size--;
        }
    }

    frame_list_free(&stack);
    set_free(&visited);
}

// ============================================================
// Tensor creation
// ============================================================

Tensor *tensor_create_ex(float x, int requires_grad) {
    Tensor *tensor = checked_malloc(sizeof(Tensor));
    tensor->ndim = 0;
    tensor->shape = NULL;
    tensor->size = 1;
    tensor->data = checked_malloc(sizeof(float));
    tensor->grad = requires_grad ? checked_calloc(1, sizeof(float)) : NULL;
    tensor->data[0] = x;
    tensor->parents = NULL;
    tensor->n_parents = 0;
    tensor->backward = NULL;
    tensor->op_name = NULL;
    tensor->requires_grad = !!requires_grad;
    tensor->is_leaf = 1;
    tensor->ref_count = 1;
    memory_stats.live_tensors++;
    memory_add(sizeof(Tensor));
    memory_add(sizeof(float));
    if (tensor->grad) memory_add(sizeof(float));
    return tensor;
}

Tensor *tensor_create(float x) { return tensor_create_ex(x, 1); }

Tensor *tensor_create_matrix_ex(int rows, int cols, int requires_grad) {
    int size;
    if (!valid_matrix_size(rows, cols, &size)) {
        report_error("tensor_create_matrix_ex", "dimensions must be positive and fit in int");
        return NULL;
    }

    Tensor *tensor = checked_malloc(sizeof(Tensor));
    tensor->ndim = 2;
    tensor->shape = checked_malloc(2 * sizeof(int));
    tensor->shape[0] = rows;
    tensor->shape[1] = cols;
    tensor->size = size;
    tensor->data = checked_calloc((size_t)size, sizeof(float));
    tensor->grad = requires_grad ? checked_calloc((size_t)size, sizeof(float)) : NULL;
    tensor->parents = NULL;
    tensor->n_parents = 0;
    tensor->backward = NULL;
    tensor->op_name = NULL;
    tensor->requires_grad = !!requires_grad;
    tensor->is_leaf = 1;
    tensor->ref_count = 1;
    memory_stats.live_tensors++;
    memory_add(sizeof(Tensor));
    memory_add(2 * sizeof(int));
    memory_add((size_t)size * sizeof(float));
    if (tensor->grad) memory_add((size_t)size * sizeof(float));
    return tensor;
}

Tensor *tensor_create_matrix(int rows, int cols) {
    return tensor_create_matrix_ex(rows, cols, 1);
}

int tensor_set_requires_grad(Tensor *tensor, int enabled) {
    if (!tensor || !tensor->is_leaf || tensor->ref_count != 1 ||
        tensor->n_parents != 0 || tensor->backward != NULL) return 0;
    enabled = !!enabled;
    if (enabled == tensor->requires_grad) return 1;
    size_t bytes = (size_t)tensor->size * sizeof(float);
    if (enabled) {
        tensor->grad = checked_calloc((size_t)tensor->size, sizeof(float));
        memory_add(bytes);
    } else {
        free(tensor->grad);
        tensor->grad = NULL;
        memory_remove(bytes);
    }
    tensor->requires_grad = enabled;
    return 1;
}

int tensor_attach_operation(Tensor *out, Tensor *const *parents, int n_parents,
                            void (*backward)(Tensor *), const char *op_name) {
    if (!out || !parents || n_parents <= 0 || !backward || !op_name ||
        !out->is_leaf || out->ref_count != 1 || out->n_parents != 0 ||
        out->parents || out->backward ||
        (size_t)n_parents > SIZE_MAX / sizeof(Tensor *)) return 0;
    int tracked = 0;
    for (int i = 0; i < n_parents; i++) {
        if (!parents[i] || parents[i] == out) return 0;
        tracked |= !!parents[i]->requires_grad;
    }
    tracked &= grad_is_enabled();
    if (tracked) {
        Tensor **links = checked_malloc((size_t)n_parents * sizeof(*links));
        for (int i = 0; i < n_parents; i++) {
            links[i] = parents[i];
            tensor_retain(links[i]);
        }
        out->parents = links;
        out->n_parents = n_parents;
        out->backward = backward;
        if (!out->grad) {
            out->grad = checked_calloc((size_t)out->size, sizeof(float));
            memory_add((size_t)out->size * sizeof(float));
        }
        memory_add((size_t)n_parents * sizeof(*links));
    } else if (out->grad) {
        free(out->grad);
        out->grad = NULL;
        memory_remove((size_t)out->size * sizeof(float));
    }
    out->requires_grad = tracked;
    out->is_leaf = 0;
    out->op_name = op_name;
    return 1;
}

// ============================================================
// Memory management
// ============================================================

void tensor_retain(Tensor *tensor) {
    if (!tensor) return;

    if (tensor->ref_count == INT_MAX) {
        fprintf(stderr, "minigrad: tensor reference count overflow\n");
        exit(EXIT_FAILURE);
    }
    tensor->ref_count++;
}

void tensor_release(Tensor *tensor) {
    if (!tensor) return;

    if (tensor->ref_count <= 0) {
        report_error("tensor_release", "invalid reference count");
        return;
    }

    tensor->ref_count--;
    if (tensor->ref_count != 0) return;

    // Releasing a long graph recursively can overflow the C stack. Process
    // zero-reference nodes iteratively instead.
    TensorList pending;
    list_init(&pending);
    list_push(&pending, tensor);

    while (pending.size > 0) {
        Tensor *node = list_pop(&pending);

        for (int i = 0; i < node->n_parents; i++) {
            Tensor *parent = node->parents[i];
            if (!parent) continue;

            if (parent->ref_count <= 0) {
                report_error("tensor_release", "invalid parent reference count");
                continue;
            }

            parent->ref_count--;
            if (parent->ref_count == 0) {
                list_push(&pending, parent);
            }
        }

        memory_remove(sizeof(Tensor));
        memory_remove((size_t)node->size * sizeof(float));
        if (node->grad) memory_remove((size_t)node->size * sizeof(float));
        if (node->shape) memory_remove(2 * sizeof(int));
        if (node->parents) memory_remove((size_t)node->n_parents * sizeof(Tensor *));
        memory_stats.live_tensors--;
        free(node->parents);
        free(node->data);
        free(node->grad);
        free(node->shape);
        free(node);
    }

    list_free(&pending);
}

// ============================================================
// Shape and broadcasting helpers
// ============================================================

typedef enum { BINARY_INVALID, BINARY_SCALAR, BINARY_MATRIX } BinaryKind;

static BinaryKind binary_kind(const Tensor *a, const Tensor *b,
                              int *rows_out, int *cols_out) {
    if (!a || !b || !rows_out || !cols_out) return BINARY_INVALID;
    if (a->ndim == 0 && b->ndim == 0) {
        *rows_out = *cols_out = 1;
        return BINARY_SCALAR;
    }
    if ((a->ndim != 0 && a->ndim != 2) ||
        (b->ndim != 0 && b->ndim != 2)) return BINARY_INVALID;
    int ar = a->ndim == 0 ? 1 : a->shape[0];
    int ac = a->ndim == 0 ? 1 : a->shape[1];
    int br = b->ndim == 0 ? 1 : b->shape[0];
    int bc = b->ndim == 0 ? 1 : b->shape[1];
    if ((ar != br && ar != 1 && br != 1) ||
        (ac != bc && ac != 1 && bc != 1)) return BINARY_INVALID;
    *rows_out = ar > br ? ar : br;
    *cols_out = ac > bc ? ac : bc;
    int size;
    return valid_matrix_size(*rows_out, *cols_out, &size) ? BINARY_MATRIX : BINARY_INVALID;
}

static int broadcast_index(const Tensor *tensor, int row, int col) {
    if (tensor->ndim == 0) return 0;
    return (tensor->shape[0] == 1 ? 0 : row) * tensor->shape[1] +
           (tensor->shape[1] == 1 ? 0 : col);
}

static float broadcast_value(const Tensor *tensor, int row, int col,
                             int output_rows, int output_cols) {
    (void)output_rows;
    (void)output_cols;
    return tensor->data[broadcast_index(tensor, row, col)];
}

static void accumulate_broadcast_grad(Tensor *tensor, int row, int col,
                                      int output_rows, int output_cols,
                                      float value) {
    (void)output_rows;
    (void)output_cols;
    if (tensor->grad) tensor->grad[broadcast_index(tensor, row, col)] += value;
}

static Tensor *create_binary_output(const char *operation,
                                    Tensor *a, Tensor *b,
                                    int *rows_out, int *cols_out,
                                    BinaryKind *kind_out) {
    BinaryKind kind = binary_kind(a, b, rows_out, cols_out);
    if (kind == BINARY_INVALID) {
        report_error(operation, "shape mismatch");
        return NULL;
    }

    Tensor *output;
    if (kind == BINARY_SCALAR) {
        output = tensor_create_ex(0.0f, 0);
    } else {
        output = tensor_create_matrix_ex(*rows_out, *cols_out, 0);
    }

    if (!output) return NULL;
    *kind_out = kind;
    return output;
}

static int attach_binary_parents(Tensor *output, Tensor *a, Tensor *b,
                                 void (*backward)(Tensor *), const char *name) {
    Tensor *parents[] = {a, b};
    return tensor_attach_operation(output, parents, 2, backward, name);
}

static int attach_unary_parent(Tensor *output, Tensor *parent,
                               void (*backward)(Tensor *), const char *name) {
    Tensor *parents[] = {parent};
    return tensor_attach_operation(output, parents, 1, backward, name);
}

// ============================================================
// Backward kernels
// ============================================================

static void add_backward(Tensor *self);
static void sub_backward(Tensor *self);
static void mean_backward(Tensor *self);
static void mul_backward(Tensor *self);
static void pow_backward(Tensor *self);
static void exp_backward(Tensor *self);
static void tanh_backward(Tensor *self);
static void relu_backward(Tensor *self);
static void sqrt_backward(Tensor *self);
static void div_backward(Tensor *self);
static void matmul_backward(Tensor *self);
static void softmax_backward(Tensor *self);

static void add_backward(Tensor *self) {
    Tensor *a = self->parents[0];
    Tensor *b = self->parents[1];
    int rows, cols;

    if (binary_kind(a, b, &rows, &cols) == BINARY_INVALID) return;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float grad = self->grad[i * cols + j];
            accumulate_broadcast_grad(a, i, j, rows, cols, grad);
            accumulate_broadcast_grad(b, i, j, rows, cols, grad);
        }
    }
}

static void sub_backward(Tensor *self) {
    Tensor *a = self->parents[0];
    Tensor *b = self->parents[1];
    int rows, cols;

    if (binary_kind(a, b, &rows, &cols) == BINARY_INVALID) return;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float grad = self->grad[i * cols + j];
            accumulate_broadcast_grad(a, i, j, rows, cols, grad);
            accumulate_broadcast_grad(b, i, j, rows, cols, -grad);
        }
    }
}

static void mul_backward(Tensor *self) {
    Tensor *a = self->parents[0];
    Tensor *b = self->parents[1];
    int rows, cols;

    if (binary_kind(a, b, &rows, &cols) == BINARY_INVALID) return;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float grad = self->grad[i * cols + j];
            float av = broadcast_value(a, i, j, rows, cols);
            float bv = broadcast_value(b, i, j, rows, cols);
            accumulate_broadcast_grad(a, i, j, rows, cols, bv * grad);
            accumulate_broadcast_grad(b, i, j, rows, cols, av * grad);
        }
    }
}

static void div_backward(Tensor *self) {
    Tensor *a = self->parents[0];
    Tensor *b = self->parents[1];
    int rows, cols;

    if (binary_kind(a, b, &rows, &cols) == BINARY_INVALID) return;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float grad = self->grad[i * cols + j];
            float av = broadcast_value(a, i, j, rows, cols);
            float bv = broadcast_value(b, i, j, rows, cols);
            accumulate_broadcast_grad(a, i, j, rows, cols, grad / bv);
            accumulate_broadcast_grad(b, i, j, rows, cols,
                                      -av * grad / (bv * bv));
        }
    }
}

static void mean_backward(Tensor *self) {
    Tensor *a = self->parents[0];
    float grad = self->grad[0] / (float)a->size;

    if (!a->grad) return;
    for (int i = 0; i < a->size; i++) {
        a->grad[i] += grad;
    }
}

static void pow_backward(Tensor *self) {
    Tensor *a = self->parents[0];
    Tensor *b = self->parents[1];
    int rows, cols;

    if (binary_kind(a, b, &rows, &cols) == BINARY_INVALID) return;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float av = broadcast_value(a, i, j, rows, cols);
            float bv = broadcast_value(b, i, j, rows, cols);
            float grad = self->grad[i * cols + j];
            float output_value = self->data[i * cols + j];

            // x^1 at x=0 has derivative 1, x^n for n>1 has
            // derivative 0. Genuine singular/undefined domains follow libm.
            if (a->grad) {
                float da = av == 0.0f && bv == 0.0f ? 0.0f :
                           av == 0.0f && bv == 1.0f ? 1.0f * grad :
                           bv * powf(av, bv - 1.0f) * grad;
                accumulate_broadcast_grad(a, i, j, rows, cols, da);
            }
            if (b->grad) {
                // At a=0 and b>0 the function is constant zero in b;
                // avoid the otherwise indeterminate 0 * log(0).
                float db = av == 0.0f && bv > 0.0f ? 0.0f :
                           output_value * logf(av) * grad;
                accumulate_broadcast_grad(b, i, j, rows, cols, db);
            }
        }
    }
}

static void exp_backward(Tensor *self) {
    Tensor *a = self->parents[0];
    if (!a->grad) return;
    for (int i = 0; i < self->size; i++) {
        a->grad[i] += self->data[i] * self->grad[i];
    }
}

static void tanh_backward(Tensor *self) {
    Tensor *a = self->parents[0];
    if (!a->grad) return;
    for (int i = 0; i < self->size; i++) {
        float value = self->data[i];
        a->grad[i] += (1.0f - value * value) * self->grad[i];
    }
}

static void relu_backward(Tensor *self) {
    Tensor *a = self->parents[0];
    if (!a->grad) return;
    for (int i = 0; i < self->size; i++) {
        if (self->data[i] > 0.0f) {
            a->grad[i] += self->grad[i];
        }
    }
}

static void sqrt_backward(Tensor *self) {
    Tensor *a = self->parents[0];
    if (!a->grad) return;
    for (int i = 0; i < self->size; i++) {
        a->grad[i] += self->grad[i] / (2.0f * self->data[i]);
    }
}

static void matmul_backward(Tensor *self) {
    Tensor *a = self->parents[0];
    Tensor *b = self->parents[1];
    int m = a->shape[0];
    int n = a->shape[1];
    int k = b->shape[1];

    mg_matmul_backward(a->data, b->data, self->grad,
                       a->grad, b->grad, m, n, k);
}

static void softmax_backward(Tensor *self) {
    Tensor *a = self->parents[0];
    if (!a->grad) return;
    int rows = self->shape[0];
    int cols = self->shape[1];

    for (int i = 0; i < rows; i++) {
        float dot = 0.0f;
        for (int j = 0; j < cols; j++) {
            dot += self->grad[i * cols + j] * self->data[i * cols + j];
        }

        for (int j = 0; j < cols; j++) {
            float probability = self->data[i * cols + j];
            a->grad[i * cols + j] +=
                probability * (self->grad[i * cols + j] - dot);
        }
    }
}

// ============================================================
// Forward operations
// ============================================================

Tensor *tensor_add(Tensor *a, Tensor *b) {
    int rows, cols;
    BinaryKind kind;
    Tensor *c = create_binary_output("tensor_add", a, b, &rows, &cols, &kind);
    if (!c) return NULL;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            c->data[i * cols + j] =
                broadcast_value(a, i, j, rows, cols) +
                broadcast_value(b, i, j, rows, cols);
        }
    }

    if (!attach_binary_parents(c, a, b, add_backward, "add")) {
        tensor_release(c); return NULL;
    }
    return c;
}

Tensor *tensor_sub(Tensor *a, Tensor *b) {
    int rows, cols;
    BinaryKind kind;
    Tensor *c = create_binary_output("tensor_sub", a, b, &rows, &cols, &kind);
    if (!c) return NULL;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            c->data[i * cols + j] =
                broadcast_value(a, i, j, rows, cols) -
                broadcast_value(b, i, j, rows, cols);
        }
    }

    if (!attach_binary_parents(c, a, b, sub_backward, "sub")) {
        tensor_release(c); return NULL;
    }
    return c;
}

Tensor *tensor_mul(Tensor *a, Tensor *b) {
    int rows, cols;
    BinaryKind kind;
    Tensor *c = create_binary_output("tensor_mul", a, b, &rows, &cols, &kind);
    if (!c) return NULL;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            c->data[i * cols + j] =
                broadcast_value(a, i, j, rows, cols) *
                broadcast_value(b, i, j, rows, cols);
        }
    }

    if (!attach_binary_parents(c, a, b, mul_backward, "mul")) {
        tensor_release(c); return NULL;
    }
    return c;
}

Tensor *tensor_div(Tensor *a, Tensor *b) {
    int rows, cols;
    BinaryKind kind;
    Tensor *c = create_binary_output("tensor_div", a, b, &rows, &cols, &kind);
    if (!c) return NULL;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            c->data[i * cols + j] =
                broadcast_value(a, i, j, rows, cols) /
                broadcast_value(b, i, j, rows, cols);
        }
    }

    if (!attach_binary_parents(c, a, b, div_backward, "div")) {
        tensor_release(c); return NULL;
    }
    return c;
}

Tensor *tensor_pow(Tensor *a, Tensor *b) {
    int rows, cols;
    BinaryKind kind;
    Tensor *c = create_binary_output("tensor_pow", a, b, &rows, &cols, &kind);
    if (!c) return NULL;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            c->data[i * cols + j] = powf(
                broadcast_value(a, i, j, rows, cols),
                broadcast_value(b, i, j, rows, cols));
        }
    }

    if (!attach_binary_parents(c, a, b, pow_backward, "pow")) {
        tensor_release(c); return NULL;
    }
    return c;
}

Tensor *tensor_mean(Tensor *a) {
    if (!a || a->size <= 0) {
        report_error("tensor_mean", "input must be non-empty");
        return NULL;
    }

    float sum = 0.0f;
    for (int i = 0; i < a->size; i++) sum += a->data[i];

    Tensor *c = tensor_create_ex(sum / (float)a->size, 0);
    if (!attach_unary_parent(c, a, mean_backward, "mean")) {
        tensor_release(c); return NULL;
    }
    return c;
}

Tensor *tensor_sqrt(Tensor *a) {
    if (!a) {
        report_error("tensor_sqrt", "input is NULL");
        return NULL;
    }

    Tensor *c;
    if (a->ndim == 0) {
        c = tensor_create_ex(sqrtf(a->data[0]), 0);
    } else if (a->ndim == 2) {
        c = tensor_create_matrix_ex(a->shape[0], a->shape[1], 0);
        if (!c) return NULL;
        for (int i = 0; i < a->size; i++) c->data[i] = sqrtf(a->data[i]);
    } else {
        report_error("tensor_sqrt", "only scalars and matrices are supported");
        return NULL;
    }

    if (!attach_unary_parent(c, a, sqrt_backward, "sqrt")) {
        tensor_release(c); return NULL;
    }
    return c;
}

Tensor *tensor_expn(Tensor *a) {
    if (!a) {
        report_error("tensor_expn", "input is NULL");
        return NULL;
    }

    Tensor *c;
    if (a->ndim == 0) {
        c = tensor_create_ex(expf(a->data[0]), 0);
    } else if (a->ndim == 2) {
        c = tensor_create_matrix_ex(a->shape[0], a->shape[1], 0);
        if (!c) return NULL;
        for (int i = 0; i < a->size; i++) c->data[i] = expf(a->data[i]);
    } else {
        report_error("tensor_expn", "only scalars and matrices are supported");
        return NULL;
    }

    if (!attach_unary_parent(c, a, exp_backward, "exp")) {
        tensor_release(c); return NULL;
    }
    return c;
}

Tensor *tensor_Tanh(Tensor *a) {
    if (!a) {
        report_error("tensor_Tanh", "input is NULL");
        return NULL;
    }

    Tensor *c;
    if (a->ndim == 0) {
        c = tensor_create_ex(tanhf(a->data[0]), 0);
    } else if (a->ndim == 2) {
        c = tensor_create_matrix_ex(a->shape[0], a->shape[1], 0);
        if (!c) return NULL;
        for (int i = 0; i < a->size; i++) c->data[i] = tanhf(a->data[i]);
    } else {
        report_error("tensor_Tanh", "only scalars and matrices are supported");
        return NULL;
    }

    if (!attach_unary_parent(c, a, tanh_backward, "tanh")) {
        tensor_release(c); return NULL;
    }
    return c;
}

Tensor *tensor_relu(Tensor *a) {
    if (!a) {
        report_error("tensor_relu", "input is NULL");
        return NULL;
    }

    Tensor *c;
    if (a->ndim == 0) {
        c = tensor_create_ex(a->data[0] > 0.0f ? a->data[0] : 0.0f, 0);
    } else if (a->ndim == 2) {
        c = tensor_create_matrix_ex(a->shape[0], a->shape[1], 0);
        if (!c) return NULL;
        for (int i = 0; i < a->size; i++) {
            c->data[i] = a->data[i] > 0.0f ? a->data[i] : 0.0f;
        }
    } else {
        report_error("tensor_relu", "only scalars and matrices are supported");
        return NULL;
    }

    if (!attach_unary_parent(c, a, relu_backward, "relu")) {
        tensor_release(c); return NULL;
    }
    return c;
}

Tensor *tensor_softmax(Tensor *a) {
    if (!a || a->ndim != 2) {
        report_error("tensor_softmax", "input must be a 2D matrix");
        return NULL;
    }

    int rows = a->shape[0];
    int cols = a->shape[1];
    Tensor *c = tensor_create_matrix_ex(rows, cols, 0);
    if (!c) return NULL;

    for (int i = 0; i < rows; i++) {
        float max_value = -INFINITY;
        int positive_infinity_count = 0;

        for (int j = 0; j < cols; j++) {
            float value = a->data[i * cols + j];
            if (isnan(value)) {
                report_error("tensor_softmax", "input contains NaN");
                tensor_release(c);
                return NULL;
            }
            if (value > max_value) max_value = value;
            if (value == INFINITY) positive_infinity_count++;
        }

        if (positive_infinity_count > 0) {
            float probability = 1.0f / (float)positive_infinity_count;
            for (int j = 0; j < cols; j++) {
                c->data[i * cols + j] =
                    a->data[i * cols + j] == INFINITY ? probability : 0.0f;
            }
            continue;
        }

        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            sum += expf(a->data[i * cols + j] - max_value);
        }

        if (!(sum > 0.0f) || !isfinite(sum)) {
            report_error("tensor_softmax", "input contains invalid values");
            tensor_release(c);
            return NULL;
        }

        for (int j = 0; j < cols; j++) {
            c->data[i * cols + j] =
                expf(a->data[i * cols + j] - max_value) / sum;
        }
    }

    if (!attach_unary_parent(c, a, softmax_backward, "softmax")) {
        tensor_release(c); return NULL;
    }
    return c;
}

Tensor *tensor_matmul(Tensor *a, Tensor *b) {
    if (!a || !b || a->ndim != 2 || b->ndim != 2) {
        report_error("tensor_matmul", "both inputs must be 2D matrices");
        return NULL;
    }

    int m = a->shape[0];
    int n = a->shape[1];
    int b_rows = b->shape[0];
    int k = b->shape[1];

    if (b_rows != n) {
        report_error("tensor_matmul", "inner dimensions do not match");
        return NULL;
    }

    Tensor *c = tensor_create_matrix_ex(m, k, 0);
    if (!c) return NULL;
    mg_matmul_forward(a->data, b->data, c->data, m, n, k);

    if (!attach_binary_parents(c, a, b, matmul_backward, "matmul")) {
        tensor_release(c); return NULL;
    }
    return c;
}

// ============================================================
// Autograd engine
// ============================================================

int tensor_backward_with_grad(Tensor *tensor, const Tensor *upstream) {
    if (!tensor || !tensor->requires_grad || !tensor->grad ||
        (upstream && (!upstream->data || tensor->ndim != upstream->ndim ||
                      tensor->size != upstream->size ||
                      (tensor->ndim == 2 &&
                       (!tensor->shape || !upstream->shape ||
                        tensor->shape[0] != upstream->shape[0] ||
                        tensor->shape[1] != upstream->shape[1]))))) return 0;

    // The seed may alias an intermediate or leaf gradient, including the
    // root's own gradient. Snapshot it before zeroing any intermediate.
    float *seed = NULL;
    if (upstream) {
        seed = checked_malloc((size_t)tensor->size * sizeof(float));
        for (int i = 0; i < tensor->size; i++) seed[i] = upstream->data[i];
    }
    TensorList topo;
    list_init(&topo);
    build_topo(tensor, &topo);
    for (int i = 0; i < topo.size; i++) {
        if (!topo.items[i]->is_leaf) tensor_zero_grad(topo.items[i]);
    }
    for (int i = 0; i < tensor->size; i++) {
        if (tensor->is_leaf) tensor->grad[i] += seed ? seed[i] : 1.0f;
        else tensor->grad[i] = seed ? seed[i] : 1.0f;
    }
    for (int i = topo.size - 1; i >= 0; i--) {
        Tensor *node = topo.items[i];
        if (node->backward && node->grad) node->backward(node);
    }
    list_free(&topo);
    free(seed);
    return 1;
}

void tensor_backward(Tensor *tensor) {
    if (!tensor_backward_with_grad(tensor, NULL))
        report_error("tensor_backward", "invalid or non-trainable output");
}

// ============================================================
// Utilities
// ============================================================

void tensor_zero_grad(Tensor *tensor) {
    if (!tensor || !tensor->grad) return;
    for (int i = 0; i < tensor->size; i++) tensor->grad[i] = 0.0f;
}

void tensor_print(Tensor *tensor, char *name) {
    if (!tensor) {
        printf("%s: <null>\n", name ? name : "tensor");
        return;
    }

    printf("%s:\n", name ? name : "tensor");
    if (tensor->ndim == 0) {
        printf("%f\n", tensor->data[0]);
        return;
    }

    int rows = tensor->shape[0];
    int cols = tensor->shape[1];
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%f ", tensor->data[i * cols + j]);
        }
        printf("\n");
    }
}
