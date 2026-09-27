#ifndef MINIGRAD_TENSOR_OPS_H
#define MINIGRAD_TENSOR_OPS_H

#include "engine.h"

/* Unary elementwise operations on scalars and 2D matrices. log follows libm:
 * log(0) is -infinity and log(negative) is NaN (and gradients follow 1/x).
 */
Tensor *tensor_log(Tensor *a);
Tensor *tensor_sigmoid(Tensor *a);
Tensor *tensor_softplus(Tensor *a);

/* axis=-1 reduces all elements to a scalar. On 2D inputs, axis=0/1
 * preserves two dimensions (1, cols)/(rows, 1), respectively.
 */
Tensor *tensor_sum_axis(Tensor *a, int axis);
Tensor *tensor_mean_axis(Tensor *a, int axis);

/* Return independent, contiguous data copies; neither operation makes a view.
 * Transpose requires a 2D input; reshape accepts scalar or 2D input and
 * produces a 2D tensor with the same number of elements.
 */
Tensor *tensor_transpose(Tensor *a);
Tensor *tensor_reshape(Tensor *a, int rows, int cols);

#endif
