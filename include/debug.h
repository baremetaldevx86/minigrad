#ifndef MINIGRAD_DEBUG_H
#define MINIGRAD_DEBUG_H

#include <stdio.h>
#include "engine.h"

/* Export the reachable graph (parents point toward results) as Graphviz DOT.
 * Returns 1 on success, 0 on invalid inputs, allocation or I/O errors.
 * Does not retain tensors or require the Graphviz executable.
 */
int tensor_dump_dot(const Tensor *root, const char *path);

/* Check every reachable tensor's data and, when requested, every allocated
 * gradient buffer. Missing gradients on frozen/no-grad tensors are normal.
 * Returns 0 for NULL, malformed tensors or nonfinite values.
 */
int tensor_check_finite(const Tensor *root, int check_gradients);

/* Print a human-readable snapshot; NULL stream defaults to stdout. */
void tensor_summary(const Tensor *tensor, FILE *stream);

#endif
