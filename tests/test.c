#include <assert.h>
#include <math.h>
#include <stdio.h>
#include "engine.h"

int main(void) {
    Tensor* a = tensor_create(4.0f);
    Tensor* b = tensor_sqrt(a);

    tensor_backward(b);

    assert(fabsf(*b->data - 2.0f) < 1e-6f);
    assert(fabsf(*a->grad - 0.25f) < 1e-6f);
    printf("value = %f\n", *b->data);
    printf("grad  = %f\n", *a->grad);

    tensor_release(b);
    tensor_release(a);
}
