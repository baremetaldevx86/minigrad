#include "optim.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

struct Adam {
    Tensor **params; /* Borrowed, including the array itself. */
    int n_params;
    float lr;
    float weight_decay;
    uint64_t *steps; /* per-parameter: freezing does not advance its moments */
    double **first;
    double **second;
};

static int valid_lr(float lr) { return isfinite(lr) && lr >= 0.0f; }

static int valid_params(Tensor **params, int n_params) {
    if (n_params < 0 || (n_params && !params)) return 0;
    for (int i = 0; i < n_params; i++) {
        Tensor *p = params[i];
        if (!p || p->size <= 0 || !p->data ||
            (p->requires_grad && !p->grad)) return 0;
        /* Reject aliases, otherwise a parameter would be updated twice. */
        for (int j = 0; j < i; j++) if (p == params[j]) return 0;
    }
    return 1;
}

static int active(const Tensor *p) {
    return p && p->requires_grad && p->grad && p->data && p->size > 0;
}

static void free_buffers(double **buffers, int n) {
    if (!buffers) return;
    for (int i = 0; i < n; i++) free(buffers[i]);
    free(buffers);
}

SGD *sgd_create_momentum(Tensor **params, int n_params, float lr,
                         float momentum) {
    if (!valid_params(params, n_params) || !valid_lr(lr) ||
        !isfinite(momentum) || momentum < 0.0f || momentum >= 1.0f)
        return NULL;
    SGD *opt = calloc(1, sizeof(*opt));
    if (!opt) return NULL;
    opt->params = params;
    opt->n_params = n_params;
    opt->lr = lr;
    opt->momentum = momentum;
    if (momentum != 0.0f && n_params > 0) {
        if ((size_t)n_params > SIZE_MAX / sizeof(*opt->velocity)) goto fail;
        opt->velocity = calloc((size_t)n_params, sizeof(*opt->velocity));
        if (!opt->velocity) goto fail;
        for (int i = 0; i < n_params; i++) {
            if ((size_t)params[i]->size > SIZE_MAX / sizeof(float)) goto fail;
            opt->velocity[i] = calloc((size_t)params[i]->size, sizeof(float));
            if (!opt->velocity[i]) goto fail;
        }
    }
    return opt;
fail:
    sgd_free(opt);
    return NULL;
}

SGD *sgd_create(Tensor **params, int n_params, float lr) {
    return sgd_create_momentum(params, n_params, lr, 0.0f);
}

void sgd_step(SGD *opt) {
    if (!opt || !valid_lr(opt->lr) || opt->n_params < 0 ||
        (opt->n_params && !opt->params)) return;
    for (int i = 0; i < opt->n_params; i++) {
        Tensor *p = opt->params[i];
        if (!active(p)) continue;
        for (int j = 0; j < p->size; j++) {
            float update = p->grad[j];
            if (opt->momentum != 0.0f) {
                update = opt->velocity[i][j] =
                    opt->momentum * opt->velocity[i][j] + update;
            }
            p->data[j] -= opt->lr * update;
        }
    }
}

void sgd_zero_grad(SGD *opt) {
    if (!opt || opt->n_params < 0 || (opt->n_params && !opt->params)) return;
    for (int i = 0; i < opt->n_params; i++) {
        if (active(opt->params[i])) tensor_zero_grad(opt->params[i]);
    }
}

void sgd_set_lr(SGD *opt, float lr) {
    if (opt && valid_lr(lr)) opt->lr = lr;
}

void sgd_free(SGD *opt) {
    if (!opt) return;
    if (opt->velocity) {
        for (int i = 0; i < opt->n_params; i++) free(opt->velocity[i]);
        free(opt->velocity);
    }
    free(opt);
}

static Adam *adam_create_impl(Tensor **params, int n_params, float lr,
                              float weight_decay) {
    if (!valid_params(params, n_params) || !valid_lr(lr) ||
        !isfinite(weight_decay) || weight_decay < 0.0f) return NULL;
    Adam *opt = calloc(1, sizeof(*opt));
    if (!opt) return NULL;
    opt->params = params;
    opt->n_params = n_params;
    opt->lr = lr;
    opt->weight_decay = weight_decay;
    if (n_params) {
        if ((size_t)n_params > SIZE_MAX / sizeof(*opt->first) ||
            (size_t)n_params > SIZE_MAX / sizeof(*opt->steps)) goto fail;
        opt->first = calloc((size_t)n_params, sizeof(*opt->first));
        opt->second = calloc((size_t)n_params, sizeof(*opt->second));
        opt->steps = calloc((size_t)n_params, sizeof(*opt->steps));
        if (!opt->first || !opt->second || !opt->steps) goto fail;
        for (int i = 0; i < n_params; i++) {
            if ((size_t)params[i]->size > SIZE_MAX / sizeof(double)) goto fail;
            opt->first[i] = calloc((size_t)params[i]->size, sizeof(double));
            opt->second[i] = calloc((size_t)params[i]->size, sizeof(double));
            if (!opt->first[i] || !opt->second[i]) goto fail;
        }
    }
    return opt;
fail:
    adam_free(opt);
    return NULL;
}

Adam *adam_create(Tensor **params, int n_params, float lr) {
    return adam_create_impl(params, n_params, lr, 0.0f);
}

Adam *adamw_create(Tensor **params, int n_params, float lr, float weight_decay) {
    return adam_create_impl(params, n_params, lr, weight_decay);
}

void adam_step(Adam *opt) {
    if (!opt || !valid_lr(opt->lr) || opt->n_params < 0 ||
        (opt->n_params && !opt->params)) return;
    for (int i = 0; i < opt->n_params; i++) {
        Tensor *p = opt->params[i];
        if (!active(p) || opt->steps[i] == UINT64_MAX) continue;
        ++opt->steps[i];
        /* expm1 improves bias correction at small t. */
        const double b1_correction = -expm1((double)opt->steps[i] * log(0.9));
        const double b2_correction = -expm1((double)opt->steps[i] * log(0.999));
        for (int j = 0; j < p->size; j++) {
            double g = p->grad[j];
            double m = opt->first[i][j] = 0.9 * opt->first[i][j] + 0.1 * g;
            double v = opt->second[i][j] = 0.999 * opt->second[i][j] + 0.001 * g * g;
            double update = (m / b1_correction) /
                (sqrt(v / b2_correction) + 1e-8);
            double value = p->data[j];
            p->data[j] = (float)(value - (double)opt->lr *
                (update + (double)opt->weight_decay * value));
        }
    }
}

void adam_zero_grad(Adam *opt) {
    if (!opt || opt->n_params < 0 || (opt->n_params && !opt->params)) return;
    for (int i = 0; i < opt->n_params; i++) {
        if (active(opt->params[i])) tensor_zero_grad(opt->params[i]);
    }
}

void adam_set_lr(Adam *opt, float lr) {
    if (opt && valid_lr(lr)) opt->lr = lr;
}

void adam_free(Adam *opt) {
    if (!opt) return;
    free_buffers(opt->first, opt->n_params);
    free_buffers(opt->second, opt->n_params);
    free(opt->steps);
    free(opt);
}

float optim_clip_grad_norm(Tensor **params, int n_params, float max_norm) {
    if (!valid_params(params, n_params) || !isfinite(max_norm) || max_norm < 0.0f)
        return NAN;
    double squared = 0.0;
    for (int i = 0; i < n_params; i++) {
        Tensor *p = params[i];
        if (!active(p)) continue;
        for (int j = 0; j < p->size; j++) {
            double g = p->grad[j];
            if (!isfinite(g)) return NAN;
            squared += g * g;
        }
    }
    double norm = sqrt(squared);
    if (norm > (double)max_norm) {
        double factor = (double)max_norm / norm;
        for (int i = 0; i < n_params; i++) {
            Tensor *p = params[i];
            if (!active(p)) continue;
            for (int j = 0; j < p->size; j++)
                p->grad[j] = (float)((double)p->grad[j] * factor);
        }
    }
    return (float)norm;
}
