#ifndef NN_H
#define NN_H

#include "engine.h"

typedef struct {
    Tensor* W;
    Tensor* b;
    int in_features;  
    int out_features;
} Linearlayer;

// Create layer
Linearlayer* linear_create(int in_features, int out_features);

// Forward Pass
Tensor* linear_forward(Linearlayer* layer, Tensor* x);

// Access parameters (for optimizer). The returned array is caller-owned;
// the tensors remain owned by the layer.
Tensor** linear_params(Linearlayer* layer, int* n_params);
void linear_free(Linearlayer* layer);

#endif
