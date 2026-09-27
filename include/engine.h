#ifndef ENGINE_H
#define ENGINE_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//
// Tensor object. Create through tensor_create*(); do not modify metadata,
// graph links, or ref_count directly. Data/grad buffers are caller-accessible.
// Keep data unchanged between forward and backward.
//
typedef struct Tensor {
    float* data;      // data buffer (scalar or matrix)
    float* grad;      // gradient buffer

    int ndim;         // number of dimensions (0 = scalar, 2 = matrix)
    int* shape;       // shape array (e.g. [rows, cols])
    int size;         // total number of elements


    struct Tensor** parents;   // computation graph parents
    int n_parents;             // number of parents

    int ref_count;             // caller owns one reference; graph edges retain parents

    void (*backward)(struct Tensor*);   // backward function
} Tensor;

//
// Tensor creation. Returned tensors own one caller reference. Matrix buffers
// are zero-initialized. Invalid dimensions return NULL. The core engine uses
// a fail-fast policy for allocation failure/graph-size/reference-count overflow:
// it prints an error and exits rather than continuing with partial objects.
//
Tensor* tensor_create(float x);                 // scalar tensor
Tensor* tensor_create_matrix(int rows, int cols); // positive dimensions; NULL on invalid shape

//
// Memory management. NULL is accepted. Release each owned reference once;
// a released/dangling pointer must never be reused (including double release).
//
void tensor_retain(Tensor* t);
void tensor_release(Tensor* t);

//
// Core ops (forward). Invalid shapes/NULL inputs return NULL.
// Binary elementwise ops accept equal shapes, scalar/matrix pairs, and
// (rows, cols) with (1, cols) row-bias broadcasting in either order.
//
Tensor* tensor_add(Tensor* a, Tensor* b);
Tensor* tensor_mul(Tensor* a, Tensor* b);
Tensor* tensor_pow(Tensor* a, Tensor* b);
Tensor* tensor_expn(Tensor* a);
Tensor* tensor_relu(Tensor* a);
Tensor* tensor_Tanh(Tensor* a);
Tensor* tensor_matmul(Tensor* A, Tensor* B);
Tensor* tensor_sub(Tensor* a, Tensor* b);
Tensor* tensor_mean(Tensor* a);
Tensor* tensor_div(Tensor* a, Tensor* b);
Tensor* tensor_sqrt(Tensor* a);

//
// Backward engine
//
void tensor_backward(Tensor* t); // clears graph gradients; seeds sum(t) with ones

// Row-wise softmax of a 2D matrix; returns NULL for invalid input.
Tensor* tensor_softmax(Tensor* t);

//
// Utility
//
void tensor_zero_grad(Tensor* t);
void tensor_print(Tensor* t, char* name); // Added for debugging

#endif

