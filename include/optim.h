#ifndef OPTIM_H
#define OPTIM_H

#include "engine.h"

typedef struct {
    Tensor** params; // borrowed, NOT owned; caller frees array and owns tensors
    int n_params;
    float lr;
} SGD;

// Returns NULL if parameters or learning rate are invalid, or allocation fails.
// Learning rates must be finite and non-negative.
SGD* sgd_create(Tensor** params, int n_params, float lr);
void sgd_step(SGD* opt);
void sgd_zero_grad(SGD* opt);
void sgd_free(SGD* opt);
void sgd_set_lr(SGD* opt, float lr);

#endif
