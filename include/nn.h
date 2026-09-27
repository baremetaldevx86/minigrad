#ifndef NN_H
#define NN_H

#include "engine.h"
#include "rng.h"

typedef enum { INIT_HE, INIT_XAVIER } WeightInit;

typedef struct {
    Tensor* W;
    Tensor* b;
    int in_features;  
    int out_features;
} Linearlayer;

// Returns NULL for invalid sizes or layer allocation failure.
// Tensor allocations follow the fail-fast policy documented in engine.h.
Linearlayer* linear_create(int in_features, int out_features);

// Initializers use N(0, sqrt(2 / fan_in)) for He and
// N(0, sqrt(2 / (fan_in + fan_out))) for Xavier. Biases start at zero.
// An explicit rng makes the weights reproducible without touching rand();
// rng == NULL uses the legacy global rand() stream (also used by linear_create).
// Returns NULL for invalid dimensions/initializer or allocation failure.
Linearlayer* linear_create_ex(int in_features, int out_features,
                             WeightInit init, MinigradRNG* rng);

// Forward pass; returns NULL if the input is invalid or an operation fails.
Tensor* linear_forward(Linearlayer* layer, Tensor* x);

// Access parameters (for optimizer). The returned array is caller-owned;
// the tensors remain owned by the layer. Returns NULL and sets *n_params to 0
// on failure.
Tensor** linear_params(Linearlayer* layer, int* n_params);
void linear_free(Linearlayer* layer);

#endif
