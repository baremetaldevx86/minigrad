#ifndef GRADCHECK_H
#define GRADCHECK_H

#include "engine.h"

/* Return one owned reference to the output, which may be non-scalar. */
typedef Tensor *(*GradcheckFunction)(Tensor **inputs, int n_inputs, void *context);

/* Compare the VJP of fn against centered finite differences of
 * dot(fn(inputs), upstream). NULL upstream represents an all-ones seed.
 * Inputs must be distinct, trainable leaves with no outstanding graph edges
 * (ref_count == 1). Neither their data nor their pre-existing gradients are
 * changed, including on failure. Callback outputs are always released.
 * Returns 1 on agreement, 0 on invalid arguments or mismatch. */
int tensor_gradcheck(GradcheckFunction fn, Tensor **inputs, int n_inputs,
                     void *context, const Tensor *upstream,
                     float epsilon, float atol, float rtol);

#endif
