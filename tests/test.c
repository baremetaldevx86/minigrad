#include <stdio.h>
#include "engine.h"

int main(){
    Tensor* a = tensor_create(4.0f);
    Tensor* b = tensor_sqrt(a);

    tensor_backward(b);

    printf("value = %f\n", *b->data); // 2.0
    printf("grad  = %f\n", *a->grad); // 0.25

    tensor_release(b);
    tensor_release(a);
}
