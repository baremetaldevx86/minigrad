#include "tensor_ops.h"
#include "engine_internal.h"

#include <limits.h>
#include <math.h>

static int valid_input(const Tensor *a) {
    if (!a || !a->data || a->size <= 0) return 0;
    if (a->ndim == 0) return a->size == 1;
    if (a->ndim != 2 || !a->shape || a->shape[0] <= 0 || a->shape[1] <= 0)
        return 0;
    return (size_t)a->shape[0] * (size_t)a->shape[1] == (size_t)a->size;
}

static Tensor *new_like(const Tensor *a) {
    return a->ndim == 0 ? tensor_create_ex(0.0f, 0) :
        tensor_create_matrix_ex(a->shape[0], a->shape[1], 0);
}

static Tensor *attach_unary(Tensor *out, Tensor *a,
                            void (*backward)(Tensor *), const char *name) {
    if (!out) return NULL;
    Tensor *parents[] = {a};
    if (!tensor_attach_operation(out, parents, 1, backward, name)) {
        tensor_release(out);
        return NULL;
    }
    return out;
}

static int tracks_parent(const Tensor *a) {
    return a->requires_grad && a->grad;
}

static float stable_sigmoid(float x) {
    if (x >= 0.0f) return 1.0f / (1.0f + expf(-x));
    float z = expf(x);
    return z / (1.0f + z);
}

static void backward_log(Tensor *out) {
    Tensor *a = out->parents[0];
    if (!tracks_parent(a)) return;
    for (int i = 0; i < a->size; i++)
        a->grad[i] += out->grad[i] / a->data[i];
}

Tensor *tensor_log(Tensor *a) {
    if (!valid_input(a)) return NULL;
    Tensor *out = new_like(a);
    if (!out) return NULL;
    for (int i = 0; i < a->size; i++) out->data[i] = logf(a->data[i]);
    return attach_unary(out, a, backward_log, "log");
}

static void backward_sigmoid(Tensor *out) {
    Tensor *a = out->parents[0];
    if (!tracks_parent(a)) return;
    for (int i = 0; i < a->size; i++)
        a->grad[i] += out->grad[i] * out->data[i] * (1.0f - out->data[i]);
}

Tensor *tensor_sigmoid(Tensor *a) {
    if (!valid_input(a)) return NULL;
    Tensor *out = new_like(a);
    if (!out) return NULL;
    for (int i = 0; i < a->size; i++) out->data[i] = stable_sigmoid(a->data[i]);
    return attach_unary(out, a, backward_sigmoid, "sigmoid");
}

static void backward_softplus(Tensor *out) {
    Tensor *a = out->parents[0];
    if (!tracks_parent(a)) return;
    for (int i = 0; i < a->size; i++)
        a->grad[i] += out->grad[i] * stable_sigmoid(a->data[i]);
}

Tensor *tensor_softplus(Tensor *a) {
    if (!valid_input(a)) return NULL;
    Tensor *out = new_like(a);
    if (!out) return NULL;
    for (int i = 0; i < a->size; i++) {
        float x = a->data[i];
        out->data[i] = fmaxf(x, 0.0f) + log1pf(expf(-fabsf(x)));
    }
    return attach_unary(out, a, backward_softplus, "softplus");
}

/* The parent's dimensions uniquely identify the reduced axis because the
 * output retains the reduced dimension (and axis=-1 yields a scalar).
 * A 1x1 input is ambiguous, but either axis has the same mapping/factor.
 */
static void backward_reduction(Tensor *out, float scale) {
    Tensor *a = out->parents[0];
    if (!tracks_parent(a)) return;
    if (out->ndim == 0) {
        for (int i = 0; i < a->size; i++)
            a->grad[i] += out->grad[0] * scale;
    } else if (out->shape[0] == 1 && out->shape[1] == a->shape[1]) {
        int cols = a->shape[1];
        for (int i = 0; i < a->size; i++)
            a->grad[i] += out->grad[i % cols] * scale;
    } else {
        int cols = a->shape[1];
        for (int i = 0; i < a->size; i++)
            a->grad[i] += out->grad[i / cols] * scale;
    }
}

static void backward_sum(Tensor *out) {
    backward_reduction(out, 1.0f);
}

static void backward_mean(Tensor *out) {
    Tensor *a = out->parents[0];
    int count = out->ndim == 0 ? a->size : a->size / out->size;
    backward_reduction(out, 1.0f / (float)count);
}

static Tensor *reduce_axis(Tensor *a, int axis, int mean) {
    if (!valid_input(a) || (axis != -1 && (a->ndim != 2 || (axis != 0 && axis != 1))))
        return NULL;
    Tensor *out = axis == -1 ? tensor_create_ex(0.0f, 0) :
        tensor_create_matrix_ex(axis == 0 ? 1 : a->shape[0],
                                axis == 0 ? a->shape[1] : 1, 0);
    if (!out) return NULL;
    if (axis == -1) {
        for (int i = 0; i < a->size; i++) out->data[0] += a->data[i];
    } else if (axis == 0) {
        int cols = a->shape[1];
        for (int i = 0; i < a->size; i++) out->data[i % cols] += a->data[i];
    } else {
        int cols = a->shape[1];
        for (int i = 0; i < a->size; i++) out->data[i / cols] += a->data[i];
    }
    if (mean) {
        int count = a->size / out->size;
        for (int i = 0; i < out->size; i++) out->data[i] /= (float)count;
    }
    return attach_unary(out, a, mean ? backward_mean : backward_sum,
                        mean ? "mean_axis" : "sum_axis");
}

Tensor *tensor_sum_axis(Tensor *a, int axis) {
    return reduce_axis(a, axis, 0);
}

Tensor *tensor_mean_axis(Tensor *a, int axis) {
    return reduce_axis(a, axis, 1);
}

static void backward_transpose(Tensor *out) {
    Tensor *a = out->parents[0];
    if (!tracks_parent(a)) return;
    int rows = a->shape[0], cols = a->shape[1];
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            a->grad[r * cols + c] += out->grad[c * rows + r];
}

Tensor *tensor_transpose(Tensor *a) {
    if (!valid_input(a) || a->ndim != 2) return NULL;
    int rows = a->shape[0], cols = a->shape[1];
    Tensor *out = tensor_create_matrix_ex(cols, rows, 0);
    if (!out) return NULL;
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            out->data[c * rows + r] = a->data[r * cols + c];
    return attach_unary(out, a, backward_transpose, "transpose");
}

static void backward_reshape(Tensor *out) {
    Tensor *a = out->parents[0];
    if (!tracks_parent(a)) return;
    for (int i = 0; i < a->size; i++) a->grad[i] += out->grad[i];
}

Tensor *tensor_reshape(Tensor *a, int rows, int cols) {
    if (!valid_input(a) || rows <= 0 || cols <= 0 ||
        (size_t)rows * (size_t)cols != (size_t)a->size) return NULL;
    Tensor *out = tensor_create_matrix_ex(rows, cols, 0);
    if (!out) return NULL;
    for (int i = 0; i < a->size; i++) out->data[i] = a->data[i];
    return attach_unary(out, a, backward_reshape, "reshape");
}
