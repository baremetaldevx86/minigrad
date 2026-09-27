#ifndef RNG_H
#define RNG_H

#include <stdint.h>

/* SplitMix64 with a cached Box-Muller second sample. The full state is public
 * so callers may save/copy an RNG; seed before first use. This stream is
 * independent of libc's srand/rand (and is not cryptographically secure).
 */
typedef struct {
    uint64_t state;
    float spare;
    int has_spare;
} MinigradRNG;

void rng_seed(MinigradRNG *rng, uint64_t seed);
float rng_uniform(MinigradRNG *rng); /* [0, 1), 24 random mantissa bits */
float rng_normal(MinigradRNG *rng);  /* standard normal; caches second sample */

#endif
