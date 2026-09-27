#include <assert.h>
#include <math.h>
#include <stdio.h>

#include "tensor_ops.h"

static void near(float actual, float expected) {
    assert(isfinite(actual));
    assert(fabsf(actual - expected) <= 2e-5f * (1.0f + fabsf(expected)));
}

static void test_elementwise(void) {
    Tensor *a = tensor_create_matrix_ex(1, 5, 1);
    assert(a);
    const float values[] = {-1000.0f, -2.0f, 0.0f, 2.0f, 1000.0f};
    for (int i = 0; i < 5; i++) a->data[i] = values[i];
    Tensor *sig = tensor_sigmoid(a);
    Tensor *sp = tensor_softplus(a);
    assert(sig && sp && sig->ndim == 2 && sig->requires_grad);
    near(sig->data[0], 0.0f);
    near(sig->data[1], 1.0f / (1.0f + expf(2.0f)));
    near(sig->data[2], 0.5f);
    near(sig->data[3], 1.0f / (1.0f + expf(-2.0f)));
    near(sig->data[4], 1.0f);
    near(sp->data[0], 0.0f);
    near(sp->data[2], logf(2.0f));
    near(sp->data[4], 1000.0f);
    for (int i = 0; i < 5; i++) near(sp->data[i], fmaxf(values[i], 0.0f) + log1pf(expf(-fabsf(values[i]))));
    tensor_backward(sig);
    for (int i = 0; i < 5; i++)
        near(a->grad[i], sig->data[i] * (1.0f - sig->data[i]));
    tensor_backward(sp); // Leaf gradients accumulate across separate graphs.
    for (int i = 0; i < 5; i++)
        near(a->grad[i], sig->data[i] * (1.0f - sig->data[i]) + sig->data[i]);
    tensor_release(sig);
    tensor_release(sp);
    tensor_release(a);

    Tensor *x = tensor_create(2.0f);
    Tensor *y = tensor_log(x);
    Tensor *z = tensor_softplus(y);
    assert(y && z && z->ndim == 0);
    near(y->data[0], logf(2.0f));
    near(z->data[0], logf(3.0f));
    tensor_backward(z);
    near(x->grad[0], 1.0f / 3.0f);
    tensor_release(z);
    tensor_release(y);
    tensor_release(x);

    x = tensor_create(0.0f);
    y = tensor_log(x);
    assert(y && isinf(y->data[0]) && signbit(y->data[0]));
    tensor_release(y);
    x->data[0] = -1.0f;
    y = tensor_log(x);
    assert(y && isnan(y->data[0])); // libm domain semantics.
    tensor_release(y);
    tensor_release(x);
}

static void test_reductions(void) {
    Tensor *a = tensor_create_matrix_ex(2, 3, 1);
    assert(a);
    for (int i = 0; i < 6; i++) a->data[i] = (float)(i + 1);
    Tensor *row = tensor_sum_axis(a, 0);
    Tensor *col = tensor_mean_axis(a, 1);
    Tensor *total = tensor_mean_axis(a, -1);
    assert(row && row->ndim == 2 && row->shape[0] == 1 && row->shape[1] == 3);
    assert(col && col->ndim == 2 && col->shape[0] == 2 && col->shape[1] == 1);
    assert(total && total->ndim == 0);
    for (int i = 0; i < 3; i++) near(row->data[i], (float)(5 + 2 * i));
    near(col->data[0], 2.0f);
    near(col->data[1], 5.0f);
    near(total->data[0], 3.5f);
    Tensor *row_seed = tensor_create_matrix_ex(1, 3, 0);
    row_seed->data[0] = 2.0f;
    row_seed->data[1] = 3.0f;
    row_seed->data[2] = 4.0f;
    assert(tensor_backward_with_grad(row, row_seed));
    for (int i = 0; i < 6; i++) near(a->grad[i], row_seed->data[i % 3]);
    tensor_backward(col);
    for (int i = 0; i < 6; i++) near(a->grad[i], row_seed->data[i % 3] + 1.0f / 3.0f);
    tensor_backward(total);
    for (int i = 0; i < 6; i++) near(a->grad[i], row_seed->data[i % 3] + 0.5f);
    tensor_release(row_seed);
    tensor_release(total);
    tensor_release(col);
    tensor_release(row);
    tensor_release(a);

    /* Degenerate shapes must not confuse reduction-axis backward mapping. */
    a = tensor_create_matrix_ex(1, 3, 1);
    for (int i = 0; i < 3; i++) a->data[i] = (float)(i + 1);
    row = tensor_sum_axis(a, 0);
    assert(row && row->shape[0] == 1 && row->shape[1] == 3);
    tensor_backward(row);
    for (int i = 0; i < 3; i++) near(a->grad[i], 1.0f);
    tensor_release(row);
    tensor_zero_grad(a);
    col = tensor_sum_axis(a, 1);
    assert(col && col->shape[0] == 1 && col->shape[1] == 1);
    near(col->data[0], 6.0f);
    tensor_backward(col);
    for (int i = 0; i < 3; i++) near(a->grad[i], 1.0f);
    tensor_release(col);
    tensor_release(a);

    a = tensor_create_matrix_ex(3, 1, 1);
    a->data[0] = 2.0f;
    a->data[1] = 4.0f;
    a->data[2] = 6.0f;
    row = tensor_mean_axis(a, 0);
    col = tensor_sum_axis(a, 1);
    total = tensor_sum_axis(a, -1);
    assert(row && row->shape[0] == 1 && row->shape[1] == 1);
    assert(col && col->shape[0] == 3 && col->shape[1] == 1);
    near(row->data[0], 4.0f);
    near(total->data[0], 12.0f);
    for (int i = 0; i < 3; i++) near(col->data[i], a->data[i]);
    tensor_backward(row);
    tensor_backward(col);
    tensor_backward(total);
    for (int i = 0; i < 3; i++) near(a->grad[i], 2.0f + 1.0f / 3.0f);
    tensor_release(row);
    tensor_release(col);
    tensor_release(total);
    tensor_release(a);

    Tensor *scalar = tensor_create(7.0f);
    total = tensor_sum_axis(scalar, -1);
    assert(total && total->ndim == 0);
    near(total->data[0], 7.0f);
    tensor_backward(total);
    near(scalar->grad[0], 1.0f);
    tensor_release(total);
    tensor_release(scalar);
}

static void test_copies(void) {
    Tensor *a = tensor_create_matrix_ex(2, 3, 1);
    for (int i = 0; i < 6; i++) a->data[i] = (float)(i + 1);
    Tensor *t = tensor_transpose(a);
    assert(t && t->shape[0] == 3 && t->shape[1] == 2 && t->data != a->data);
    const float expected[] = {1, 4, 2, 5, 3, 6};
    for (int i = 0; i < 6; i++) near(t->data[i], expected[i]);
    Tensor *b = tensor_reshape(t, 1, 6);
    assert(b && b->shape[0] == 1 && b->shape[1] == 6 && b->data != t->data);
    for (int i = 0; i < 6; i++) near(b->data[i], expected[i]);
    b->data[0] = 99.0f;
    near(a->data[0], 1.0f);
    near(t->data[0], 1.0f);
    Tensor *seed = tensor_create_matrix_ex(1, 6, 0);
    for (int i = 0; i < 6; i++) seed->data[i] = (float)(i + 1);
    assert(tensor_backward_with_grad(b, seed));
    const float expected_grad[] = {1, 3, 5, 2, 4, 6};
    for (int i = 0; i < 6; i++) near(a->grad[i], expected_grad[i]);
    tensor_release(seed);
    tensor_release(b);
    tensor_release(t);
    tensor_release(a);

    Tensor *scalar = tensor_create(4.0f);
    b = tensor_reshape(scalar, 1, 1);
    assert(b && b->ndim == 2 && b->data != scalar->data);
    tensor_backward(b);
    near(scalar->grad[0], 1.0f);
    tensor_release(b);
    tensor_release(scalar);
}

static void test_validation_and_detachment(void) {
    Tensor *a = tensor_create_matrix_ex(2, 3, 0);
    assert(a && !a->grad);
    assert(!tensor_log(NULL) && !tensor_sigmoid(NULL) && !tensor_softplus(NULL));
    assert(!tensor_sum_axis(NULL, -1) && !tensor_mean_axis(NULL, 0));
    assert(!tensor_sum_axis(a, 2) && !tensor_mean_axis(a, -2));
    assert(!tensor_transpose(NULL));
    assert(!tensor_reshape(a, 1, 5) && !tensor_reshape(a, 0, 6));
    Tensor *scalar = tensor_create(1.0f);
    assert(!tensor_sum_axis(scalar, 0) && !tensor_mean_axis(scalar, 1));
    assert(!tensor_transpose(scalar) && !tensor_reshape(scalar, 1, 2));
    assert(a->ref_count == 1 && scalar->ref_count == 1);

    Tensor *out = tensor_sigmoid(a);
    assert(out && !out->requires_grad && !out->grad && out->n_parents == 0);
    tensor_release(out);
    int previous = grad_set_enabled(0);
    out = tensor_log(scalar);
    assert(out && !out->requires_grad && !out->grad && out->n_parents == 0);
    tensor_release(out);
    out = tensor_mean_axis(scalar, -1);
    assert(out && !out->requires_grad && !out->grad && out->n_parents == 0);
    tensor_release(out);
    grad_set_enabled(previous);
    tensor_release(scalar);
    tensor_release(a);
}

int main(void) {
    test_elementwise();
    test_reductions();
    test_copies();
    test_validation_and_detachment();
    puts("tensor operations: passed");
    return 0;
}
