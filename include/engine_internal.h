#ifndef MINIGRAD_ENGINE_INTERNAL_H
#define MINIGRAD_ENGINE_INTERNAL_H

#include "engine.h"

/* Attach a freshly allocated output to parents only when grad mode is enabled
 * and at least one input requires gradients. Retains all parents if tracking.
 * Callers own output on both success (1) and failure (0).
 */
int tensor_attach_operation(Tensor *out, Tensor *const *parents, int n_parents,
                            void (*backward)(Tensor *), const char *op_name);

#endif
