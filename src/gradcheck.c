#include "gradcheck.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    float *data;
    float *grad;
    float *analytic;
} InputSnapshot;

typedef struct {
    int ndim;
    int size;
    int rows;
    int cols;
} OutputShape;

static int valid_shape(const Tensor *t) {
    if (!t || !t->data || t->size <= 0) return 0;
    if (t->ndim == 0) return t->size == 1;
    return t->ndim == 2 && t->shape && t->shape[0] > 0 &&
           t->shape[1] > 0 && (size_t)t->shape[0] * (size_t)t->shape[1] == (size_t)t->size;
}

static int same_shape(const Tensor *t, const OutputShape *shape) {
    return valid_shape(t) && t->ndim == shape->ndim && t->size == shape->size &&
           (t->ndim == 0 || (t->shape[0] == shape->rows && t->shape[1] == shape->cols));
}

/* Always release the callback's owned output, even if its shape or data is bad. */
static int evaluate(GradcheckFunction fn, Tensor **inputs, int n_inputs,
                    void *context, const OutputShape *shape, const float *seed,
                    double *objective) {
    Tensor *out = fn(inputs, n_inputs, context);
    int ok = same_shape(out, shape);
    double sum = 0.0;
    if (ok) {
        for (int i = 0; i < out->size; ++i) {
            if (!isfinite(out->data[i])) { ok = 0; break; }
            sum += (double)out->data[i] * (seed ? (double)seed[i] : 1.0);
        }
        if (!isfinite(sum)) ok = 0;
    }
    tensor_release(out);
    if (ok) *objective = sum;
    return ok;
}

int tensor_gradcheck(GradcheckFunction fn, Tensor **inputs, int n_inputs,
                     void *context, const Tensor *upstream,
                     float epsilon, float atol, float rtol) {
    if (!fn || !inputs || n_inputs <= 0 || !isfinite(epsilon) || epsilon <= 0.0f ||
        !isfinite(atol) || atol < 0.0f || !isfinite(rtol) || rtol < 0.0f)
        return 0;
    if ((size_t)n_inputs > SIZE_MAX / sizeof(InputSnapshot)) return 0;

    InputSnapshot *saved = calloc((size_t)n_inputs, sizeof(*saved));
    if (!saved) return 0;
    float *seed = NULL;
    Tensor *out = NULL;
    int old_mode = grad_is_enabled();
    int success = 0;
    OutputShape shape = {0};

    for (int i = 0; i < n_inputs; ++i) {
        Tensor *t = inputs[i];
        if (!valid_shape(t) || !t->requires_grad || !t->is_leaf || !t->grad ||
            t->ref_count != 1 || (size_t)t->size > SIZE_MAX / sizeof(float)) goto cleanup;
        for (int j = 0; j < i; ++j) if (inputs[j] == t) goto cleanup;
        size_t bytes = (size_t)t->size * sizeof(float);
        saved[i].data = malloc(bytes);
        if (!saved[i].data) goto cleanup;
        memcpy(saved[i].data, t->data, bytes);
        saved[i].grad = malloc(bytes);
        if (!saved[i].grad) goto cleanup;
        memcpy(saved[i].grad, t->grad, bytes);
        saved[i].analytic = malloc(bytes);
        if (!saved[i].analytic) goto cleanup;
        for (int j = 0; j < t->size; ++j) if (!isfinite(t->data[j])) goto cleanup;
    }

    grad_set_enabled(1);
    out = fn(inputs, n_inputs, context);
    if (!valid_shape(out) || !out->requires_grad || !out->grad) goto cleanup;
    shape.ndim = out->ndim;
    shape.size = out->size;
    if (out->ndim == 2) {
        shape.rows = out->shape[0];
        shape.cols = out->shape[1];
    }
    if (upstream) {
        if (!valid_shape(upstream) || !same_shape(upstream, &shape) ||
            (size_t)shape.size > SIZE_MAX / sizeof(float)) goto cleanup;
        seed = malloc((size_t)shape.size * sizeof(float));
        if (!seed) goto cleanup;
        memcpy(seed, upstream->data, (size_t)shape.size * sizeof(float));
        for (int j = 0; j < shape.size; ++j) if (!isfinite(seed[j])) goto cleanup;
    }
    for (int j = 0; j < shape.size; ++j) if (!isfinite(out->data[j])) goto cleanup;

    /* Preserve the seed even if it aliases an input gradient being cleared. */
    for (int i = 0; i < n_inputs; ++i)
        memset(inputs[i]->grad, 0, (size_t)inputs[i]->size * sizeof(float));
    Tensor seed_tensor = {0};
    if (upstream) {
        seed_tensor = *upstream;
        seed_tensor.data = seed;
    }
    if (!tensor_backward_with_grad(out, upstream ? &seed_tensor : NULL)) goto cleanup;
    for (int i = 0; i < n_inputs; ++i) {
        Tensor *t = inputs[i];
        for (int j = 0; j < t->size; ++j)
            if (!isfinite(t->grad[j])) goto cleanup;
        memcpy(saved[i].analytic, t->grad, (size_t)t->size * sizeof(float));
        /* Numerical callbacks see the original gradients, not scratch VJPs. */
        memcpy(t->grad, saved[i].grad, (size_t)t->size * sizeof(float));
    }
    tensor_release(out);
    out = NULL;
    grad_set_enabled(0);

    for (int i = 0; i < n_inputs; ++i) {
        Tensor *t = inputs[i];
        for (int j = 0; j < t->size; ++j) {
            float x = saved[i].data[j];
            float plus = (float)((double)x + (double)epsilon);
            float minus = (float)((double)x - (double)epsilon);
            if (!isfinite(plus) || !isfinite(minus) || plus == minus) goto cleanup;
            double fplus, fminus;
            t->data[j] = plus;
            if (!evaluate(fn, inputs, n_inputs, context, &shape, seed, &fplus)) goto cleanup;
            grad_set_enabled(0);
            t->data[j] = minus;
            if (!evaluate(fn, inputs, n_inputs, context, &shape, seed, &fminus)) goto cleanup;
            grad_set_enabled(0);
            t->data[j] = x;
            double numerical = (fplus - fminus) / ((double)plus - (double)minus);
            double analytical = (double)saved[i].analytic[j];
            if (!isfinite(numerical) ||
                fabs(numerical - analytical) > (double)atol +
                    (double)rtol * fmax(fabs(numerical), fabs(analytical))) goto cleanup;
        }
    }
    success = 1;

cleanup:
    tensor_release(out);
    for (int i = 0; i < n_inputs; ++i) {
        if (inputs[i] && saved[i].data)
            memcpy(inputs[i]->data, saved[i].data, (size_t)inputs[i]->size * sizeof(float));
        if (inputs[i] && saved[i].grad)
            memcpy(inputs[i]->grad, saved[i].grad, (size_t)inputs[i]->size * sizeof(float));
        free(saved[i].data);
        free(saved[i].grad);
        free(saved[i].analytic);
    }
    free(seed);
    free(saved);
    grad_set_enabled(old_mode);
    return success;
}
