/* Differentiable MSE and fused, row-wise softmax cross-entropy. */
#ifndef LOSS_H
#define LOSS_H

#include "engine.h"

/* Mean squared error. NULL for NULL inputs or incompatible shapes. */
Tensor* mse_loss(Tensor* y_pred, Tensor* y_true);

/* Logits/targets must be matching nonempty (batch, classes) matrices.
 * Targets may be nonnegative finite weights (normalization is not required).
 * Returns the mean of -sum_j target_j * logsoftmax(logits)_j. NULL on
 * nonfinite logits/targets or a loss that cannot fit in float.
 * When tracked, retains both operands and differentiates each trainable one:
 * dlogits = (softmax * row_sum(targets) - targets) / batch,
 * dtargets = -logsoftmax / batch, multiplied by the upstream seed.
 * No-grad/untracked results retain no parents. Caller owns the returned loss.
 */
Tensor* cross_entropy_loss(Tensor* logits, Tensor* targets);

#endif
