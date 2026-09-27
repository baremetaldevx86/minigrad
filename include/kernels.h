#ifndef MINIGRAD_KERNELS_H
#define MINIGRAD_KERNELS_H

/* Internal contiguous row-major kernels: A[rows, inner] @ B[inner, cols].
 * Dimensions must be positive, buffer sizes checked by callers, and outputs
 * must not overlap any input. Parameter order is rows, inner, cols. */
void mg_matmul_forward(const float *a, const float *b, float *out,
                       int rows, int inner, int cols);
/* Accumulate into da/db, either of which may be NULL to skip that derivative. */
void mg_matmul_backward(const float *a, const float *b, const float *upstream,
                        float *da, float *db, int rows, int inner, int cols);

#endif
