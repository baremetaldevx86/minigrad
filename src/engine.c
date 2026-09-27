#include "engine.h"

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

Tensor *tensor_create(float x) {
    Tensor *tensor = checked_malloc(sizeof(Tensor));

    tensor->ndim = 0;
    tensor->shape = NULL;
    tensor->size = 1;
    tensor->data = checked_malloc(sizeof(float));
    tensor->grad = checked_calloc(1, sizeof(float));

    tensor->data[0] = x;
    tensor->parents = NULL;
    tensor->n_parents = 0;
    tensor->backward = NULL;
    tensor->ref_count = 1;

    return tensor;
}

Tensor *tensor_create_matrix(int rows, int cols) {
    int size;
    if (!valid_matrix_size(rows, cols, &size)) {
        report_error("tensor_create_matrix", "dimensions must be positive and fit in int");
        return NULL;
    }

    Tensor *tensor = checked_malloc(sizeof(Tensor));
    tensor->ndim = 2;
    tensor->shape = checked_malloc(2 * sizeof(int));
    tensor->shape[0] = rows;
    tensor->shape[1] = cols;
    tensor->size = size;

    // Zero data as well as gradients. This makes partially constructed tensors
    // deterministic if a caller fills only part of an input buffer.
    tensor->data = checked_calloc((size_t)size, sizeof(float));
    tensor->grad = checked_calloc((size_t)size, sizeof(float));

    tensor->parents = NULL;
    tensor->n_parents = 0;
    tensor->backward = NULL;
    tensor->ref_count = 1;

    return tensor;
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

typedef enum {
    BINARY_INVALID,
    BINARY_SCALAR,
    BINARY_SAME_SHAPE,
    BINARY_A_SCALAR,
    BINARY_B_SCALAR,
    BINARY_A_BIAS,
    BINARY_B_BIAS
} BinaryKind;

static BinaryKind binary_kind(const Tensor *a, const Tensor *b,
                              int *rows_out, int *cols_out) {
    if (!a || !b || !rows_out || !cols_out) return BINARY_INVALID;

    if (a->ndim == 0 && b->ndim == 0) {
        *rows_out = 1;
        *cols_out = 1;
        return BINARY_SCALAR;
    }

    if (a->ndim == 0 && b->ndim == 2) {
        *rows_out = b->shape[0];
        *cols_out = b->shape[1];
        return BINARY_A_SCALAR;
    }

    if (a->ndim == 2 && b->ndim == 0) {
        *rows_out = a->shape[0];
        *cols_out = a->shape[1];
        return BINARY_B_SCALAR;
    }

    if (a->ndim != 2 || b->ndim != 2) return BINARY_INVALID;

    if (a->shape[0] == b->shape[0] && a->shape[1] == b->shape[1]) {
        *rows_out = a->shape[0];
        *cols_out = a->shape[1];
        return BINARY_SAME_SHAPE;
    }

    // A row vector is the bias representation used by the neural-network
    // layer: (batch, features) op (1, features).
    if (b->shape[0] == 1 && b->shape[1] == a->shape[1]) {
        *rows_out = a->shape[0];
        *cols_out = a->shape[1];
        return BINARY_B_BIAS;
    }

    if (a->shape[0] == 1 && a->shape[1] == b->shape[1]) {
        *rows_out = b->shape[0];
        *cols_out = b->shape[1];
        return BINARY_A_BIAS;
    }

    return BINARY_INVALID;
}

static float broadcast_value(const Tensor *tensor, int row, int col,
                             int output_rows, int output_cols) {
    if (tensor->ndim == 0) return tensor->data[0];
    if (tensor->shape[0] == 1 && output_rows > 1) {
        return tensor->data[col];
    }
    return tensor->data[row * output_cols + col];
}

static void accumulate_broadcast_grad(Tensor *tensor, int row, int col,
                                      int output_rows, int output_cols,
                                      float value) {
    if (tensor->ndim == 0) {
        tensor->grad[0] += value;
    } else if (tensor->shape[0] == 1 && output_rows > 1) {
        tensor->grad[col] += value;
    } else {
        tensor->grad[row * output_cols + col] += value;
    }
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
        output = tensor_create(0.0f);
    } else {
        output = tensor_create_matrix(*rows_out, *cols_out);
    }

    if (!output) return NULL;
    *kind_out = kind;
    return output;
}

static void attach_binary_parents(Tensor *output, Tensor *a, Tensor *b,
                                  void (*backward)(Tensor *)) {
    output->parents = checked_malloc(2 * sizeof(Tensor *));
    output->parents[0] = a;
    output->parents[1] = b;
    output->n_parents = 2;
    tensor_retain(a);
    tensor_retain(b);
    output->backward = backward;
}

static void attach_unary_parent(Tensor *output, Tensor *parent,
                                void (*backward)(Tensor *)) {
    output->parents = checked_malloc(sizeof(Tensor *));
    output->parents[0] = parent;
    output->n_parents = 1;
    tensor_retain(parent);
    output->backward = backward;
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

            // d(a^b)/da is undefined at a=0 for some exponents. Avoid
            // inventing a gradient in that case.
            if (av != 0.0f) {
                float da = bv * powf(av, bv - 1.0f) * grad;
                accumulate_broadcast_grad(a, i, j, rows, cols, da);
            }

            // The derivative with respect to a continuous exponent is only
            // defined for positive bases.
            if (av > 0.0f) {
                float db = output_value * logf(av) * grad;
                accumulate_broadcast_grad(b, i, j, rows, cols, db);
            }
        }
    }
}

static void exp_backward(Tensor *self) {
    Tensor *a = self->parents[0];
    for (int i = 0; i < self->size; i++) {
        a->grad[i] += self->data[i] * self->grad[i];
    }
}

static void tanh_backward(Tensor *self) {
    Tensor *a = self->parents[0];
    for (int i = 0; i < self->size; i++) {
        float value = self->data[i];
        a->grad[i] += (1.0f - value * value) * self->grad[i];
    }
}

static void relu_backward(Tensor *self) {
    Tensor *a = self->parents[0];
    for (int i = 0; i < self->size; i++) {
        if (self->data[i] > 0.0f) {
            a->grad[i] += self->grad[i];
        }
    }
}

static void sqrt_backward(Tensor *self) {
    Tensor *a = self->parents[0];
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

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++) {
                sum += self->grad[i * k + p] * b->data[j * k + p];
            }
            a->grad[i * n + j] += sum;
        }
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            float sum = 0.0f;
            for (int p = 0; p < m; p++) {
                sum += a->data[p * n + i] * self->grad[p * k + j];
            }
            b->grad[i * k + j] += sum;
        }
    }
}

static void softmax_backward(Tensor *self) {
    Tensor *a = self->parents[0];
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

    attach_binary_parents(c, a, b, add_backward);
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

    attach_binary_parents(c, a, b, sub_backward);
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

    attach_binary_parents(c, a, b, mul_backward);
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

    attach_binary_parents(c, a, b, div_backward);
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

    attach_binary_parents(c, a, b, pow_backward);
    return c;
}

Tensor *tensor_mean(Tensor *a) {
    if (!a || a->size <= 0) {
        report_error("tensor_mean", "input must be non-empty");
        return NULL;
    }

    float sum = 0.0f;
    for (int i = 0; i < a->size; i++) sum += a->data[i];

    Tensor *c = tensor_create(sum / (float)a->size);
    attach_unary_parent(c, a, mean_backward);
    return c;
}

Tensor *tensor_sqrt(Tensor *a) {
    if (!a) {
        report_error("tensor_sqrt", "input is NULL");
        return NULL;
    }

    Tensor *c;
    if (a->ndim == 0) {
        c = tensor_create(sqrtf(a->data[0]));
    } else if (a->ndim == 2) {
        c = tensor_create_matrix(a->shape[0], a->shape[1]);
        if (!c) return NULL;
        for (int i = 0; i < a->size; i++) c->data[i] = sqrtf(a->data[i]);
    } else {
        report_error("tensor_sqrt", "only scalars and matrices are supported");
        return NULL;
    }

    attach_unary_parent(c, a, sqrt_backward);
    return c;
}

Tensor *tensor_expn(Tensor *a) {
    if (!a) {
        report_error("tensor_expn", "input is NULL");
        return NULL;
    }

    Tensor *c;
    if (a->ndim == 0) {
        c = tensor_create(expf(a->data[0]));
    } else if (a->ndim == 2) {
        c = tensor_create_matrix(a->shape[0], a->shape[1]);
        if (!c) return NULL;
        for (int i = 0; i < a->size; i++) c->data[i] = expf(a->data[i]);
    } else {
        report_error("tensor_expn", "only scalars and matrices are supported");
        return NULL;
    }

    attach_unary_parent(c, a, exp_backward);
    return c;
}

Tensor *tensor_Tanh(Tensor *a) {
    if (!a) {
        report_error("tensor_Tanh", "input is NULL");
        return NULL;
    }

    Tensor *c;
    if (a->ndim == 0) {
        c = tensor_create(tanhf(a->data[0]));
    } else if (a->ndim == 2) {
        c = tensor_create_matrix(a->shape[0], a->shape[1]);
        if (!c) return NULL;
        for (int i = 0; i < a->size; i++) c->data[i] = tanhf(a->data[i]);
    } else {
        report_error("tensor_Tanh", "only scalars and matrices are supported");
        return NULL;
    }

    attach_unary_parent(c, a, tanh_backward);
    return c;
}

Tensor *tensor_relu(Tensor *a) {
    if (!a) {
        report_error("tensor_relu", "input is NULL");
        return NULL;
    }

    Tensor *c;
    if (a->ndim == 0) {
        c = tensor_create(a->data[0] > 0.0f ? a->data[0] : 0.0f);
    } else if (a->ndim == 2) {
        c = tensor_create_matrix(a->shape[0], a->shape[1]);
        if (!c) return NULL;
        for (int i = 0; i < a->size; i++) {
            c->data[i] = a->data[i] > 0.0f ? a->data[i] : 0.0f;
        }
    } else {
        report_error("tensor_relu", "only scalars and matrices are supported");
        return NULL;
    }

    attach_unary_parent(c, a, relu_backward);
    return c;
}

Tensor *tensor_softmax(Tensor *a) {
    if (!a || a->ndim != 2) {
        report_error("tensor_softmax", "input must be a 2D matrix");
        return NULL;
    }

    int rows = a->shape[0];
    int cols = a->shape[1];
    Tensor *c = tensor_create_matrix(rows, cols);
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

    attach_unary_parent(c, a, softmax_backward);
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

    Tensor *c = tensor_create_matrix(m, k);
    if (!c) return NULL;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            float sum = 0.0f;
            for (int p = 0; p < n; p++) {
                sum += a->data[i * n + p] * b->data[p * k + j];
            }
            c->data[i * k + j] = sum;
        }
    }

    attach_binary_parents(c, a, b, matmul_backward);
    return c;
}

// ============================================================
// Autograd engine
// ============================================================

void tensor_backward(Tensor *tensor) {
    if (!tensor) {
        report_error("tensor_backward", "input is NULL");
        return;
    }

    TensorList topo;
    list_init(&topo);
    build_topo(tensor, &topo);

    // Each call computes a fresh vector-Jacobian product. Gradients from a
    // previous call are cleared before seeding the output gradient.
    for (int i = 0; i < topo.size; i++) {
        tensor_zero_grad(topo.items[i]);
    }

    if (tensor->size == 1) {
        tensor->grad[0] = 1.0f;
    } else {
        // For a non-scalar output this computes the gradient of sum(output).
        for (int i = 0; i < tensor->size; i++) tensor->grad[i] = 1.0f;
    }

    for (int i = topo.size - 1; i >= 0; i--) {
        Tensor *node = topo.items[i];
        if (node->backward) node->backward(node);
    }

    list_free(&topo);
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
