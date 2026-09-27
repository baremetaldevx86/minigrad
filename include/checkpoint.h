#ifndef CHECKPOINT_H
#define CHECKPOINT_H

#include "mlp.h"

/* Version 1 checkpoint: "MGMLP001" magic, little-endian u32 version (1),
 * u32 layer count, u64 scalar parameter count, then (layers + 1) little-endian
 * u32 dimensions, followed by each layer's row-major weights and biases as
 * little-endian IEEE-754 binary32. No optimizer, RNG or training state is
 * stored. Only finite parameters are supported. Loading produces trainable
 * parameters. Files exceeding 65536 layers or 16 million parameters are
 * rejected as a resource-safety limit. */
int mlp_save(const MLP *model, const char *path); /* 1 success, 0 failure */
MLP *mlp_load(const char *path);                   /* owned result or NULL */

#endif
