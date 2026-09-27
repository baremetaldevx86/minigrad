#include "engine.h"
#include "loss.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>

static void near(float actual, double expected, double tolerance) {
    assert(isfinite(actual));
    assert(fabs((double)actual - expected) <= tolerance);
}

static void test_weighted_targets_and_seed(void) {
    Tensor *z = tensor_create_matrix(2, 2);
    Tensor *y = tensor_create_matrix(2, 2);
    assert(z && y);
    z->data[0] = 0; z->data[1] = 0;
    z->data[2] = logf(3.0f); z->data[3] = 0;
    y->data[0] = 2; y->data[1] = 1;
    y->data[2] = 0; y->data[3] = 4;
    Tensor *loss = cross_entropy_loss(z, y);
    Tensor *seed = tensor_create_ex(3.0f, 0);
    assert(loss && seed && loss->requires_grad && loss->n_parents == 2);
    near(loss->data[0], (3.0 * log(2.0) + 4.0 * log(4.0)) / 2.0, 1e-6);
    assert(tensor_backward_with_grad(loss, seed));
    near(z->grad[0], -0.75, 1e-6);
    near(z->grad[1], 0.75, 1e-6);
    near(z->grad[2], 4.5, 1e-6);
    near(z->grad[3], -4.5, 1e-6);
    near(y->grad[0], 1.5 * log(2.0), 1e-6);
    near(y->grad[1], 1.5 * log(2.0), 1e-6);
    near(y->grad[2], 1.5 * log(4.0 / 3.0), 1e-6);
    near(y->grad[3], 1.5 * log(4.0), 1e-6);
    /* Repeated backward must clear intermediates, but accumulate into leaves. */
    assert(tensor_backward_with_grad(loss, seed));
    near(z->grad[0], -1.5, 1e-6);
    near(y->grad[3], 3.0 * log(4.0), 1e-6);
    tensor_release(seed); tensor_release(loss);
    tensor_release(z); tensor_release(y);
}

static void test_tracking_and_ownership(void) {
    Tensor *z = tensor_create_matrix_ex(1, 2, 0);
    Tensor *y = tensor_create_matrix(1, 2);
    assert(z && y && !z->grad);
    z->data[0] = 3; z->data[1] = 1; y->data[0] = 1;
    Tensor *loss = cross_entropy_loss(z, y);
    assert(loss && loss->requires_grad && loss->n_parents == 2);
    tensor_release(z); /* the graph retains even frozen parents' values */
    tensor_backward(loss);
    double logsum = log1p(exp(-2.0));
    near(y->grad[0], logsum, 1e-6);
    near(y->grad[1], 2.0 + logsum, 1e-6);
    tensor_release(loss); tensor_release(y);

    z = tensor_create_matrix(1, 2);
    y = tensor_create_matrix_ex(1, 2, 0);
    assert(z && y && !y->grad);
    z->data[0] = 0; z->data[1] = 0; y->data[0] = 2;
    loss = cross_entropy_loss(z, y);
    assert(loss && loss->requires_grad);
    tensor_backward(loss);
    near(z->grad[0], -1.0, 1e-6);
    near(z->grad[1], 1.0, 1e-6);
    tensor_release(loss); tensor_release(z); tensor_release(y);

    z = tensor_create_matrix_ex(1, 2, 0);
    y = tensor_create_matrix_ex(1, 2, 0);
    assert(z && y);
    loss = cross_entropy_loss(z, y);
    assert(loss && !loss->requires_grad && !loss->grad && !loss->parents &&
           loss->n_parents == 0);
    tensor_release(loss); tensor_release(z); tensor_release(y);

    z = tensor_create_matrix(1, 2);
    y = tensor_create_matrix(1, 2);
    assert(z && y);
    int previous = grad_set_enabled(0);
    loss = cross_entropy_loss(z, y);
    assert(loss && !loss->requires_grad && !loss->grad && !loss->parents);
    assert(grad_set_enabled(previous) == 0);
    tensor_release(loss); tensor_release(z); tensor_release(y);
}

static void test_extremes(void) {
    Tensor *z = tensor_create_matrix(1, 2);
    Tensor *y = tensor_create_matrix(1, 2);
    assert(z && y);
    z->data[0] = FLT_MAX; z->data[1] = -FLT_MAX;
    y->data[1] = 0.25f;
    Tensor *loss = cross_entropy_loss(z, y);
    assert(loss);
    near(loss->data[0] / FLT_MAX, 0.5, 1e-6);
    tensor_backward(loss);
    near(z->grad[0], 0.25, 1e-6);
    near(z->grad[1], -0.25, 1e-6);
    near(y->grad[0] / FLT_MAX, 0.0, 1e-6);
    assert(isinf(y->grad[1]) && y->grad[1] > 0); /* exact derivative > FLT_MAX */
    tensor_release(loss);
    z->data[0] = -FLT_MAX; z->data[1] = -FLT_MAX;
    tensor_zero_grad(z); tensor_zero_grad(y);
    loss = cross_entropy_loss(z, y);
    assert(loss && isfinite(loss->data[0]));
    tensor_backward(loss);
    near(y->grad[0], log(2.0), 1e-6);
    tensor_release(loss); tensor_release(z); tensor_release(y);
}

static void test_invalid_inputs_and_mse(void) {
    Tensor *z = tensor_create_matrix(1, 2);
    Tensor *y = tensor_create_matrix(1, 2);
    Tensor *bad_shape = tensor_create_matrix(2, 1);
    Tensor *scalar = tensor_create(1.0f);
    assert(z && y && bad_shape && scalar);
    assert(!cross_entropy_loss(NULL, y));
    assert(!cross_entropy_loss(z, NULL));
    assert(!cross_entropy_loss(z, bad_shape));
    assert(!cross_entropy_loss(scalar, y));
    z->data[0] = NAN;
    assert(!cross_entropy_loss(z, y));
    z->data[0] = INFINITY;
    assert(!cross_entropy_loss(z, y));
    z->data[0] = 0; y->data[0] = -0.1f;
    assert(!cross_entropy_loss(z, y));
    y->data[0] = NAN;
    assert(!cross_entropy_loss(z, y));
    y->data[0] = FLT_MAX; y->data[1] = FLT_MAX;
    z->data[0] = FLT_MAX; z->data[1] = -FLT_MAX;
    assert(!cross_entropy_loss(z, y)); /* result cannot fit in float */
    assert(!mse_loss(NULL, y));
    tensor_release(z); tensor_release(y);
    tensor_release(bad_shape); tensor_release(scalar);

    Tensor *a = tensor_create(2);
    Tensor *b = tensor_create_ex(1, 0);
    Tensor *loss = mse_loss(a, b);
    assert(loss);
    near(loss->data[0], 1, 1e-6);
    tensor_backward(loss);
    near(a->grad[0], 2, 1e-6);
    assert(!b->grad);
    tensor_release(loss); tensor_release(a); tensor_release(b);
}

int main(void) {
    test_weighted_targets_and_seed();
    test_tracking_and_ownership();
    test_extremes();
    test_invalid_inputs_and_mse();
    puts("loss tests passed");
    return 0;
}
