#include <assert.h>
#include <math.h>
#include <stdio.h>

#include "engine.h"
#include "loss.h"

static void expect_close(float actual, float expected) {
    float tolerance = 2e-4f * (1.0f + fabsf(expected));
    assert(fabsf(actual - expected) <= tolerance);
}

static void expect_matrix(const Tensor *tensor, const float *expected) {
    assert(tensor != NULL);
    for (int i = 0; i < tensor->size; i++) {
        expect_close(tensor->data[i], expected[i]);
    }
}

static void test_broadcasting_and_bias(void) {
    printf("Test: Scalar and bias broadcasting... ");

    Tensor *matrix = tensor_create_matrix(2, 3);
    Tensor *scalar = tensor_create(10.0f);
    const float matrix_values[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    for (int i = 0; i < matrix->size; i++) matrix->data[i] = matrix_values[i];

    Tensor *sum = tensor_add(matrix, scalar);
    const float sum_expected[] = {11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
    expect_matrix(sum, sum_expected);
    tensor_backward(sum);
    for (int i = 0; i < matrix->size; i++) expect_close(matrix->grad[i], 1.0f);
    expect_close(scalar->grad[0], 6.0f);
    tensor_release(sum);

    Tensor *scalar_left = tensor_create(2.0f);
    Tensor *scalar_sum = tensor_add(scalar_left, matrix);
    const float scalar_sum_expected[] = {3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    expect_matrix(scalar_sum, scalar_sum_expected);
    tensor_backward(scalar_sum);
    expect_close(scalar_left->grad[0], 6.0f);
    for (int i = 0; i < matrix->size; i++) expect_close(matrix->grad[i], 1.0f);
    tensor_release(scalar_sum);

    Tensor *bias = tensor_create_matrix(1, 3);
    bias->data[0] = 0.5f;
    bias->data[1] = -1.0f;
    bias->data[2] = 2.0f;
    Tensor *biased = tensor_add(matrix, bias);
    const float biased_expected[] = {1.5f, 1.0f, 5.0f, 4.5f, 4.0f, 8.0f};
    expect_matrix(biased, biased_expected);
    tensor_backward(biased);
    for (int i = 0; i < matrix->size; i++) expect_close(matrix->grad[i], 1.0f);
    for (int i = 0; i < bias->size; i++) expect_close(bias->grad[i], 2.0f);

    tensor_release(biased);

    Tensor *scaled = tensor_mul(bias, matrix); // Reverse bias broadcasting.
    const float scaled_expected[] = {0.5f, -2.0f, 6.0f,
                                      2.0f, -5.0f, 12.0f};
    expect_matrix(scaled, scaled_expected);
    tensor_backward(scaled);
    const float bias_grad_expected[] = {5.0f, 7.0f, 9.0f};
    for (int i = 0; i < matrix->size; i++) {
        expect_close(matrix->grad[i], bias->data[i % 3]);
    }
    for (int i = 0; i < bias->size; i++) {
        expect_close(bias->grad[i], bias_grad_expected[i]);
    }
    tensor_release(scaled);

    Tensor *quotient = tensor_div(matrix, scalar);
    const float quotient_expected[] = {0.1f, 0.2f, 0.3f,
                                        0.4f, 0.5f, 0.6f};
    expect_matrix(quotient, quotient_expected);
    tensor_backward(quotient);
    for (int i = 0; i < matrix->size; i++) expect_close(matrix->grad[i], 0.1f);
    expect_close(scalar->grad[0], -0.21f);
    tensor_release(quotient);

    tensor_release(bias);
    tensor_release(scalar_left);
    tensor_release(scalar);
    tensor_release(matrix);
    printf("passed\n");
}

static void test_pow_scalar_and_matrix(void) {
    printf("Test: Scalar and matrix power gradients... ");

    Tensor *x = tensor_create(3.0f);
    Tensor *exponent = tensor_create(2.0f);
    Tensor *scalar_power = tensor_pow(x, exponent);
    expect_close(scalar_power->data[0], 9.0f);
    tensor_backward(scalar_power);
    expect_close(x->grad[0], 6.0f);
    expect_close(exponent->grad[0], 9.0f * logf(3.0f));
    tensor_release(scalar_power);
    tensor_release(exponent);
    tensor_release(x);

    Tensor *bases = tensor_create_matrix(1, 3);
    bases->data[0] = 1.0f;
    bases->data[1] = 2.0f;
    bases->data[2] = 4.0f;
    Tensor *matrix_exponent = tensor_create(3.0f);
    Tensor *matrix_power = tensor_pow(bases, matrix_exponent);
    const float matrix_expected[] = {1.0f, 8.0f, 64.0f};
    expect_matrix(matrix_power, matrix_expected);
    tensor_backward(matrix_power);
    expect_close(bases->grad[0], 3.0f);
    expect_close(bases->grad[1], 12.0f);
    expect_close(bases->grad[2], 48.0f);
    expect_close(matrix_exponent->grad[0], 8.0f * logf(2.0f) +
                                            64.0f * logf(4.0f));
    tensor_release(matrix_power);
    tensor_release(matrix_exponent);
    tensor_release(bases);

    Tensor *scalar_base = tensor_create(2.0f);
    Tensor *matrix_exponents = tensor_create_matrix(1, 3);
    matrix_exponents->data[0] = 1.0f;
    matrix_exponents->data[1] = 2.0f;
    matrix_exponents->data[2] = 3.0f;
    Tensor *scalar_base_power = tensor_pow(scalar_base, matrix_exponents);
    const float reverse_expected[] = {2.0f, 4.0f, 8.0f};
    expect_matrix(scalar_base_power, reverse_expected);
    tensor_backward(scalar_base_power);
    expect_close(scalar_base->grad[0], 17.0f);
    for (int i = 0; i < 3; i++) {
        expect_close(matrix_exponents->grad[i], reverse_expected[i] * logf(2.0f));
    }
    tensor_release(scalar_base_power);
    tensor_release(matrix_exponents);
    tensor_release(scalar_base);
    printf("passed\n");
}

static void test_matmul_both_gradients(void) {
    printf("Test: Matrix multiplication gradients... ");

    Tensor *a = tensor_create_matrix(2, 3);
    Tensor *b = tensor_create_matrix(3, 2);
    const float a_values[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const float b_values[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    for (int i = 0; i < a->size; i++) a->data[i] = a_values[i];
    for (int i = 0; i < b->size; i++) b->data[i] = b_values[i];

    Tensor *product = tensor_matmul(a, b);
    const float product_expected[] = {22.0f, 28.0f, 49.0f, 64.0f};
    expect_matrix(product, product_expected);
    tensor_backward(product); // The non-scalar output seeds d(sum(product)).

    const float a_grad_expected[] = {3.0f, 7.0f, 11.0f,
                                     3.0f, 7.0f, 11.0f};
    const float b_grad_expected[] = {5.0f, 5.0f, 7.0f, 7.0f, 9.0f, 9.0f};
    for (int i = 0; i < a->size; i++) expect_close(a->grad[i], a_grad_expected[i]);
    for (int i = 0; i < b->size; i++) expect_close(b->grad[i], b_grad_expected[i]);

    tensor_release(product);
    tensor_release(b);
    tensor_release(a);
    printf("passed\n");
}

static void test_losses(void) {
    printf("Test: MSE and cross-entropy losses... ");

    Tensor *prediction = tensor_create_matrix(1, 3);
    Tensor *target = tensor_create_matrix(1, 3);
    prediction->data[0] = 1.0f;
    prediction->data[1] = 2.0f;
    prediction->data[2] = 3.0f;
    target->data[0] = 0.0f;
    target->data[1] = 1.0f;
    target->data[2] = 2.0f;
    Tensor *mse = mse_loss(prediction, target);
    expect_close(mse->data[0], 1.0f);
    tensor_backward(mse);
    for (int i = 0; i < 3; i++) {
        expect_close(prediction->grad[i], 2.0f / 3.0f);
        expect_close(target->grad[i], -2.0f / 3.0f);
    }
    tensor_release(mse);
    tensor_release(target);
    tensor_release(prediction);

    Tensor *logits = tensor_create_matrix(2, 3);
    Tensor *labels = tensor_create_matrix(2, 3);
    const float logits_values[] = {1.0f, 2.0f, 3.0f, 0.0f, 0.0f, 0.0f};
    const float labels_values[] = {0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f};
    for (int i = 0; i < logits->size; i++) {
        logits->data[i] = logits_values[i];
        labels->data[i] = labels_values[i];
    }
    Tensor *cross_entropy = cross_entropy_loss(logits, labels);
    float expected_loss = (logf(expf(1.0f) + expf(2.0f) + expf(3.0f)) -
                           3.0f + logf(3.0f)) / 2.0f;
    expect_close(cross_entropy->data[0], expected_loss);
    tensor_backward(cross_entropy);

    float normalizer = expf(1.0f) + expf(2.0f) + expf(3.0f);
    const float first_probabilities[] = {expf(1.0f) / normalizer,
                                         expf(2.0f) / normalizer,
                                         expf(3.0f) / normalizer};
    for (int i = 0; i < 3; i++) {
        expect_close(logits->grad[i], first_probabilities[i] / 2.0f -
                                         (i == 2 ? 0.5f : 0.0f));
    }
    expect_close(logits->grad[3], -1.0f / 3.0f);
    expect_close(logits->grad[4], 1.0f / 6.0f);
    expect_close(logits->grad[5], 1.0f / 6.0f);
    for (int i = 0; i < labels->size; i++) expect_close(labels->grad[i], 0.0f);

    tensor_release(cross_entropy);
    tensor_release(labels);
    tensor_release(logits);

    Tensor *large_logits = tensor_create_matrix(1, 2);
    Tensor *large_labels = tensor_create_matrix(1, 2);
    large_logits->data[0] = 1000.0f;
    large_logits->data[1] = 999.0f;
    large_labels->data[0] = 1.0f;
    Tensor *stable_loss = cross_entropy_loss(large_logits, large_labels);
    expect_close(stable_loss->data[0], log1pf(expf(-1.0f)));
    tensor_backward(stable_loss);
    float second_probability = 1.0f / (expf(1.0f) + 1.0f);
    expect_close(large_logits->grad[0], -second_probability);
    expect_close(large_logits->grad[1], second_probability);
    tensor_release(stable_loss);
    tensor_release(large_labels);
    tensor_release(large_logits);
    printf("passed\n");
}

static void test_refcounts_and_graph_cleanup(void) {
    printf("Test: Reference counts and graph cleanup... ");

    Tensor *a = tensor_create(2.0f);
    Tensor *b = tensor_create(3.0f);
    Tensor *exponent = tensor_create(2.0f);
    Tensor *sum = tensor_add(a, b);
    Tensor *product = tensor_mul(sum, a);
    Tensor *power = tensor_pow(product, exponent);
    assert(a->ref_count == 3);
    assert(b->ref_count == 2);
    assert(sum->ref_count == 2);
    assert(product->ref_count == 2);
    assert(exponent->ref_count == 2);

    tensor_retain(a);
    assert(a->ref_count == 4);
    tensor_release(a); // Drop the extra retained reference.
    assert(a->ref_count == 3);
    tensor_backward(power);

    tensor_release(power);
    tensor_release(product);
    tensor_release(sum);
    tensor_release(b);
    tensor_release(exponent);
    tensor_release(a); // Drop the original caller reference.

    // Exercise iterative cleanup repeatedly; sanitizer targets catch leaks and
    // use-after-free while this also covers shared parents in a DAG.
    for (int i = 0; i < 1000; i++) {
        Tensor *x = tensor_create(1.0f);
        Tensor *y = tensor_create(2.0f);
        Tensor *shared = tensor_add(x, y);
        Tensor *left = tensor_mul(shared, x);
        Tensor *right = tensor_sub(shared, y);
        Tensor *root = tensor_add(left, right);
        tensor_release(root);
        tensor_release(left);
        tensor_release(right);
        tensor_release(shared);
        tensor_release(y);
        tensor_release(x);
    }
    printf("passed\n");
}

static void test_shape_and_error_behavior(void) {
    printf("Test: Shape and error handling... ");

    assert(tensor_create_matrix(0, 2) == NULL);
    assert(tensor_create_matrix(2, 0) == NULL);
    assert(tensor_create_matrix(-1, 2) == NULL);
    assert(tensor_create_matrix(50000, 50000) == NULL);

    Tensor *matrix = tensor_create_matrix(2, 2);
    Tensor *wrong = tensor_create_matrix(1, 3);
    Tensor *scalar = tensor_create(1.0f);
    assert(tensor_add(matrix, wrong) == NULL);
    assert(tensor_matmul(matrix, wrong) == NULL);
    assert(tensor_matmul(scalar, matrix) == NULL);
    assert(tensor_softmax(scalar) == NULL);
    assert(tensor_mean(NULL) == NULL);
    assert(tensor_sqrt(NULL) == NULL);
    assert(tensor_add(NULL, matrix) == NULL);
    tensor_release(scalar);
    tensor_release(wrong);
    tensor_release(matrix);
    printf("passed\n");
}

int main(void) {
    setbuf(stdout, NULL);
    test_broadcasting_and_bias();
    test_pow_scalar_and_matrix();
    test_matmul_both_gradients();
    test_losses();
    test_refcounts_and_graph_cleanup();
    test_shape_and_error_behavior();
    printf("All engine regression tests passed!\n");
    return 0;
}
