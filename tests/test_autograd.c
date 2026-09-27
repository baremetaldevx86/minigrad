#include <assert.h>
#include <math.h>
#include <stdio.h>

#include "engine.h"

static void close_to(float got, float expected) {
    assert(fabsf(got - expected) <= 2e-5f * (1.0f + fabsf(expected)));
}

static void test_tracking_and_mode(void) {
    int initial_mode = grad_is_enabled();
    grad_set_enabled(1);
    Tensor *x = tensor_create(3.0f);
    Tensor *constant = tensor_create_ex(4.0f, 0);
    Tensor *matrix = tensor_create_matrix_ex(1, 2, 0);
    assert(x && constant && matrix);
    assert(x->requires_grad && x->is_leaf && x->grad);
    assert(!constant->requires_grad && constant->is_leaf && !constant->grad);
    assert(!matrix->requires_grad && !matrix->grad);

    int old_mode = grad_set_enabled(0);
    assert(old_mode == 1 && !grad_is_enabled());
    Tensor *detached = tensor_mul(x, constant);
    Tensor *fresh = tensor_create(5.0f); // Leaf constructors ignore grad mode.
    assert(detached && !detached->requires_grad && !detached->grad);
    assert(!detached->is_leaf && detached->n_parents == 0);
    assert(!tensor_backward_with_grad(detached, NULL));
    assert(x->ref_count == 1 && constant->ref_count == 1);
    assert(fresh && fresh->requires_grad && fresh->grad);
    assert(grad_set_enabled(old_mode) == 0);

    Tensor *constant_only = tensor_add(constant, constant);
    assert(constant_only && !constant_only->requires_grad && !constant_only->grad);
    assert(constant_only->n_parents == 0 && !constant_only->is_leaf);
    tensor_release(constant_only);

    Tensor *tracked = tensor_mul(x, constant);
    assert(tracked && tracked->requires_grad && tracked->grad);
    assert(tracked->n_parents == 2 && tracked->parents[1] == constant);
    assert(x->ref_count == 2 && constant->ref_count == 2);
    tensor_release(constant); // Graph must retain nontrainable data for d(x*c).
    tensor_backward(tracked);
    close_to(x->grad[0], 4.0f);
    tensor_release(tracked);
    tensor_release(detached);
    tensor_release(matrix);
    tensor_release(fresh);
    tensor_release(x);
    grad_set_enabled(initial_mode);
}

static void test_vjp_and_accumulation(void) {
    Tensor *x = tensor_create_matrix(1, 2);
    Tensor *factor = tensor_create_ex(2.0f, 0);
    x->data[0] = 3.0f;
    x->data[1] = -4.0f;
    Tensor *intermediate = tensor_mul(x, factor);
    Tensor *result = tensor_add(intermediate, intermediate);
    Tensor *seed = tensor_create_matrix_ex(1, 2, 0);
    Tensor *wrong = tensor_create_ex(1.0f, 0);
    assert(intermediate && result && seed && wrong);
    seed->data[0] = 1.0f;
    seed->data[1] = -3.0f;
    assert(!tensor_backward_with_grad(result, wrong));
    close_to(x->grad[0], 0.0f);
    close_to(x->grad[1], 0.0f);
    assert(tensor_backward_with_grad(result, seed));
    close_to(x->grad[0], 4.0f);
    close_to(x->grad[1], -12.0f);
    close_to(intermediate->grad[0], 2.0f);
    close_to(intermediate->grad[1], -6.0f);
    assert(tensor_backward_with_grad(result, seed));
    close_to(x->grad[0], 8.0f);
    close_to(x->grad[1], -24.0f);
    close_to(intermediate->grad[0], 2.0f); // Not stale/doubled.
    tensor_zero_grad(x);
    assert(tensor_backward_with_grad(result, NULL));
    close_to(x->grad[0], 4.0f);
    close_to(x->grad[1], 4.0f);

    Tensor *second = tensor_mul(x, factor);
    assert(second && tensor_backward_with_grad(second, seed));
    close_to(x->grad[0], 6.0f);
    close_to(x->grad[1], -2.0f);
    tensor_release(second);
    tensor_release(wrong);
    tensor_release(seed);
    tensor_release(result);
    tensor_release(intermediate);
    tensor_release(factor);
    tensor_release(x);

    Tensor *leaf = tensor_create(1.0f);
    Tensor *upstream = tensor_create_ex(2.5f, 0);
    assert(leaf && upstream);
    assert(tensor_backward_with_grad(leaf, upstream));
    assert(tensor_backward_with_grad(leaf, upstream));
    close_to(leaf->grad[0], 5.0f);
    tensor_backward(leaf);
    close_to(leaf->grad[0], 6.0f);
    tensor_release(upstream);
    tensor_release(leaf);
}

static void test_frozen_and_guards(void) {
    Tensor *x = tensor_create(2.0f);
    Tensor *frozen = tensor_create_ex(3.0f, 0);
    assert(x && frozen && tensor_set_requires_grad(frozen, 1));
    assert(frozen->requires_grad && frozen->grad);
    assert(tensor_set_requires_grad(frozen, 0) && !frozen->grad);
    assert(!tensor_set_requires_grad(NULL, 1));
    Tensor *product = tensor_mul(x, frozen);
    assert(product && product->requires_grad);
    assert(!tensor_set_requires_grad(frozen, 1)); // Referenced by graph.
    assert(!tensor_set_requires_grad(product, 0)); // Nonleaf.
    tensor_backward(product);
    close_to(x->grad[0], 3.0f);
    assert(frozen->grad == NULL);
    tensor_release(product);
    assert(tensor_set_requires_grad(frozen, 1));
    Tensor *sum = tensor_add(x, frozen);
    assert(sum && sum->requires_grad);
    tensor_backward(sum);
    close_to(x->grad[0], 4.0f);
    close_to(frozen->grad[0], 1.0f);
    tensor_release(sum);
    tensor_release(frozen);
    tensor_release(x);
}

static void test_general_broadcast(void) {
    Tensor *row = tensor_create_matrix(1, 3);
    Tensor *column = tensor_create_matrix(2, 1);
    Tensor *seed = tensor_create_matrix_ex(2, 3, 0);
    assert(row && column && seed);
    row->data[0] = 2.0f;
    row->data[1] = 3.0f;
    row->data[2] = 4.0f;
    column->data[0] = 5.0f;
    column->data[1] = 7.0f;
    const float s[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    for (int i = 0; i < 6; i++) seed->data[i] = s[i];
    Tensor *product = tensor_mul(row, column);
    assert(product && product->ndim == 2 && product->shape[0] == 2 && product->shape[1] == 3);
    const float values[] = {10, 15, 20, 14, 21, 28};
    for (int i = 0; i < 6; i++) close_to(product->data[i], values[i]);
    assert(tensor_backward_with_grad(product, seed));
    close_to(row->grad[0], 33.0f);
    close_to(row->grad[1], 45.0f);
    close_to(row->grad[2], 57.0f);
    close_to(column->grad[0], 20.0f);
    close_to(column->grad[1], 47.0f);
    tensor_release(product);

    tensor_zero_grad(row);
    tensor_zero_grad(column);
    Tensor *sum = tensor_add(column, row);
    assert(sum && tensor_backward_with_grad(sum, seed));
    close_to(row->grad[0], 5.0f);
    close_to(row->grad[1], 7.0f);
    close_to(row->grad[2], 9.0f);
    close_to(column->grad[0], 6.0f);
    close_to(column->grad[1], 15.0f);
    tensor_release(sum);
    tensor_release(seed);
    tensor_release(column);
    tensor_release(row);
}

int main(void) {
    TensorMemoryStats baseline = tensor_memory_stats();
    test_tracking_and_mode();
    test_vjp_and_accumulation();
    test_frozen_and_guards();
    test_general_broadcast();
    TensorMemoryStats after = tensor_memory_stats();
    assert(after.live_tensors == baseline.live_tensors);
    assert(after.live_bytes == baseline.live_bytes);
    assert(after.peak_bytes >= after.live_bytes);
    puts("All autograd tests passed!");
    return 0;
}
