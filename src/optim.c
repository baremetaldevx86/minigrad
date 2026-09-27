#include "optim.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

SGD *sgd_create(Tensor **params, int n_params, float lr) {
    if (n_params < 0 || (n_params > 0 && !params) || !isfinite(lr) || lr < 0.0f) {
        fprintf(stderr, "minigrad: sgd_create: invalid parameters or learning rate\n");
        return NULL;
    }
    for (int i = 0; i < n_params; i++) {
        if (!params[i] || params[i]->size <= 0 || !params[i]->data ||
            !params[i]->grad) {
            fprintf(stderr, "minigrad: sgd_create: invalid tensor parameter\n");
            return NULL;
        }
    }

    SGD *opt = (SGD *)malloc(sizeof(*opt));
    if (!opt) {
        fprintf(stderr, "minigrad: sgd_create: out of memory\n");
        return NULL;
    }
    opt->params = params; /* Borrowed: caller owns array and tensors. */
    opt->n_params = n_params;
    opt->lr = lr;
    return opt;
}

void sgd_step(SGD *opt) {
    if (!opt || opt->n_params < 0 ||
        (opt->n_params > 0 && !opt->params) || !isfinite(opt->lr) ||
        opt->lr < 0.0f) return;

    for (int i = 0; i < opt->n_params; i++) {
        Tensor *p = opt->params[i];
        if (!p || p->size <= 0 || !p->data || !p->grad) continue;
        for (int j = 0; j < p->size; j++) {
            p->data[j] -= opt->lr * p->grad[j];
        }
    }
}

void sgd_zero_grad(SGD *opt) {
    if (!opt || opt->n_params < 0 ||
        (opt->n_params > 0 && !opt->params)) return;
    for (int i = 0; i < opt->n_params; i++) {
        if (opt->params[i]) tensor_zero_grad(opt->params[i]);
    }
}

void sgd_free(SGD *opt) {
    /* The parameter array and tensors are borrowed, not owned by SGD. */
    free(opt);
}

void sgd_set_lr(SGD *opt, float lr) {
    if (opt && isfinite(lr) && lr >= 0.0f) opt->lr = lr;
}
