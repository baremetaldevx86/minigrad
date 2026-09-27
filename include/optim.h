#ifndef OPTIM_H
#define OPTIM_H

#include "engine.h"

/* Parameter arrays and tensors are borrowed: keep both alive until free.
 * Tensor metadata (including size) must remain unchanged while registered.
 * Repeated parameter pointers are rejected. Frozen/no-grad tensors are skipped.
 */
typedef struct {
    Tensor **params;
    int n_params;
    float lr;
    float momentum;   /* [0, 1); zero for ordinary SGD */
    float **velocity; /* owned; NULL for ordinary SGD */
} SGD;

/* Empty parameter lists are allowed. Learning rates must be finite and >= 0.
 * Momentum must be finite and in [0, 1). On invalid input or allocation failure
 * creation returns NULL. Momentum uses v = momentum*v + grad, p -= lr*v.
 */
SGD *sgd_create(Tensor **params, int n_params, float lr);
SGD *sgd_create_momentum(Tensor **params, int n_params, float lr, float momentum);
void sgd_step(SGD *opt);
void sgd_zero_grad(SGD *opt);
void sgd_set_lr(SGD *opt, float lr);
void sgd_free(SGD *opt);

typedef struct Adam Adam;
/* Standard Adam: beta1=0.9, beta2=0.999, eps=1e-8. AdamW additionally
 * applies decoupled p -= lr*weight_decay*p to active parameters each step.
 * Decay must be finite and nonnegative. Internal moments are owned by Adam.
 */
Adam *adam_create(Tensor **params, int n_params, float lr);
Adam *adamw_create(Tensor **params, int n_params, float lr, float weight_decay);
void adam_step(Adam *opt);
void adam_zero_grad(Adam *opt);
void adam_set_lr(Adam *opt, float lr);
void adam_free(Adam *opt);

/* Returns the pre-clipping L2 norm. Skips frozen/no-grad tensors. Invalid
 * arguments, duplicate params or nonfinite active gradients return NAN without
 * modifying gradients. max_norm must be finite and >= 0. A norm too large for
 * float may return INFINITY even when every gradient was finite.
 */
float optim_clip_grad_norm(Tensor **params, int n_params, float max_norm);

#endif
