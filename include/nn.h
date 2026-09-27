#ifndef NN_H
#define NN_H

#include "engine.h"

typedef struct {
    Tensor* W;
    Tensor* b;
    int in_features;  
    int out_features;
} Linearlayer;

// Returns NULL for invalid sizes or layer allocation failure.
// Tensor allocations follow the fail-fast policy documented in engine.h.
Linearlayer* linear_create(int in_features, int out_features);

// Forward pass; returns NULL if the input is invalid or an operation fails.
Tensor* linear_forward(Linearlayer* layer, Tensor* x);

// Access parameters (for optimizer). The returned array is caller-owned;
// the tensors remain owned by the layer. Returns NULL and sets *n_params to 0
// on failure.
Tensor** linear_params(Linearlayer* layer, int* n_params);
void linear_free(Linearlayer* layer);

#endif
