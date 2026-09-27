#include "rng.h"

#include <math.h>

/* Fixed unsigned arithmetic makes the generator independent of libc's rand.
 * Each draw advances by the golden-ratio increment and scrambles the result.
 */
static uint64_t splitmix64(MinigradRNG *rng) {
    uint64_t z = (rng->state += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

void rng_seed(MinigradRNG *rng, uint64_t seed) {
    if (!rng) return;
    rng->state = seed;
    rng->spare = 0.0f;
    rng->has_spare = 0;
}

float rng_uniform(MinigradRNG *rng) {
    if (!rng) return 0.0f;
    return (float)(splitmix64(rng) >> 40) * (1.0f / 16777216.0f);
}

float rng_normal(MinigradRNG *rng) {
    if (!rng) return 0.0f;
    if (rng->has_spare) {
        rng->has_spare = 0;
        return rng->spare;
    }
    /* 1-u is in (0,1], including when the uniform draw is exactly zero. */
    double u1 = 1.0 - (double)rng_uniform(rng);
    double u2 = (double)rng_uniform(rng);
    double radius = sqrt(-2.0 * log(u1));
    double angle = 6.283185307179586476925286766559 * u2;
    rng->spare = (float)(radius * sin(angle));
    rng->has_spare = 1;
    return (float)(radius * cos(angle));
}
