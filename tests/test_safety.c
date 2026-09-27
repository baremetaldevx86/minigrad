#define _POSIX_C_SOURCE 200809L

#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "engine.h"
#include "loss.h"
#include "mlp.h"
#include "mnist_loader.h"
#include "optim.h"

static void test_deep_graph(void) {
    // Exceeds the old 10,000-node buffer and typical recursive stack limits.
    Tensor *x = tensor_create(1.0f);
    Tensor *root = x;
    tensor_retain(root);
    for (int i = 0; i < 100000; i++) {
        Tensor *next = tensor_add(root, x);
        assert(next);
        tensor_release(root);
        root = next;
    }
    tensor_backward(root);
    assert(x->grad[0] == 100001.0f);
    tensor_release(x); // Leave only the graph's references alive.
    tensor_release(root);
}

static void test_operation_errors(void) {
    Tensor *a = tensor_create_matrix(2, 3);
    Tensor *b = tensor_create_matrix(1, 6);
    Tensor *scalar = tensor_create(2.0f);
    Tensor *(*ops[])(Tensor *, Tensor *) = {
        tensor_add, tensor_sub, tensor_mul, tensor_div, tensor_pow
    };
    for (size_t i = 0; i < sizeof(ops) / sizeof(ops[0]); i++) {
        // Equal element counts are not equivalent shapes.
        assert(!ops[i](a, b));
        assert(!ops[i](b, a));
        assert(!ops[i](a, NULL));
        assert(!ops[i](NULL, a));
        assert(a->ref_count == 1 && b->ref_count == 1);
    }
    assert(!mse_loss(a, b));
    assert(!mse_loss(NULL, b));
    assert(!cross_entropy_loss(a, b));
    assert(!cross_entropy_loss(NULL, b));
    assert(!cross_entropy_loss(scalar, b));

    Tensor *difference = tensor_sub(scalar, a);
    assert(difference);
    tensor_backward(difference);
    assert(scalar->grad[0] == 6.0f);
    for (int i = 0; i < a->size; i++) {
        assert(difference->data[i] == 2.0f);
        assert(a->grad[i] == -1.0f);
    }
    tensor_release(difference);
    tensor_release(scalar);
    tensor_release(b);
    tensor_release(a);
    tensor_backward(NULL);
    tensor_zero_grad(NULL);
    tensor_retain(NULL);
    tensor_release(NULL);
}

static void test_softmax_and_loss_errors(void) {
    Tensor *logits = tensor_create_matrix(2, 2);
    Tensor *targets = tensor_create_matrix(2, 2);
    for (int i = 0; i < logits->size; i++) logits->data[i] = -2e9f;
    targets->data[0] = targets->data[2] = 1.0f;
    Tensor *loss = cross_entropy_loss(logits, targets);
    assert(loss && fabsf(loss->data[0] - logf(2.0f)) < 1e-5f);
    tensor_backward(loss);
    assert(logits->grad[0] == -0.25f && logits->grad[1] == 0.25f);
    tensor_release(loss);

    tensor_zero_grad(logits); // Leaf gradients accumulate across separate graphs.
    Tensor *probabilities = tensor_softmax(logits);
    Tensor *weighted = tensor_mul(probabilities, targets);
    assert(probabilities && weighted);
    tensor_backward(weighted);
    for (int i = 0; i < logits->size; i++) {
        assert(probabilities->data[i] == 0.5f);
        assert(logits->grad[i] == (i % 2 == 0 ? 0.25f : -0.25f));
    }
    tensor_release(probabilities);
    tensor_release(weighted);

    // Fail after an earlier row has been computed; no partially built graph leaks.
    logits->data[2] = NAN;
    logits->data[3] = INFINITY;
    assert(!tensor_softmax(logits));
    assert(!cross_entropy_loss(logits, targets));
    logits->data[2] = logits->data[3] = -INFINITY;
    assert(!tensor_softmax(logits));
    logits->data[2] = logits->data[3] = 0.0f;
    targets->data[2] = -1.0f;
    assert(!cross_entropy_loss(logits, targets));
    assert(logits->ref_count == 1 && targets->ref_count == 1);
    tensor_release(targets);
    tensor_release(logits);
}

static void test_model_lifetimes(void) {
    assert(!linear_create(0, 1));
    assert(!linear_create(INT_MAX, 2));
    assert(!mlp_create(NULL, 2));
    int partial_sizes[] = {2, 3, 0};
    assert(!mlp_create(partial_sizes, 2)); // Cleans up the first layer.
    int count = -1;
    assert(!linear_params(NULL, &count) && count == 0);
    assert(!mlp_params(NULL, &count) && count == 0);
    assert(!sgd_create(NULL, 1, 0.1f));
    assert(!sgd_create(NULL, -1, 0.1f));
    assert(!sgd_create(NULL, 0, NAN));

    int sizes[] = {2, 4, 1};
    MLP *model = mlp_create(sizes, 2);
    assert(model && mlp_count_scalar_params(model) == 17);
    Tensor **params = mlp_params(model, &count);
    assert(params && count == 4);
    SGD *opt = sgd_create(params, count, 0.1f);
    assert(opt);
    Tensor *input = tensor_create_matrix_ex(3, 2, 0);
    Tensor *target = tensor_create_matrix_ex(3, 1, 0);
    Tensor *wrong = tensor_create_matrix_ex(2, 3, 0);
    for (int i = 0; i < 20; i++) {
        assert(!mlp_forward(model, wrong, 1));
        assert(wrong->ref_count == 1);
        Tensor *prediction = mlp_forward(model, input, i % 2);
        Tensor *loss = mse_loss(prediction, target);
        assert(prediction && loss);
        sgd_zero_grad(opt);
        tensor_backward(loss);
        sgd_step(opt);
        tensor_release(prediction);
        tensor_release(loss);
        assert(input->ref_count == 1 && target->ref_count == 1);
        for (int j = 0; j < count; j++) assert(params[j]->ref_count == 1);
    }
    params[0]->grad[0] = 1.0f;
    float before = params[0]->data[0];
    sgd_step(opt);
    assert(fabsf(params[0]->data[0] - (before - 0.1f)) < 1e-6f);
    sgd_set_lr(opt, NAN);
    assert(opt->lr == 0.1f);
    sgd_free(opt);
    free(params);

    // Graph edges own parameters, so a graph can outlive the model wrapper.
    Tensor *out = mlp_forward(model, input, 0);
    assert(out);
    mlp_free(model);
    tensor_release(input);
    tensor_backward(out);
    tensor_release(out);
    tensor_release(wrong);
    tensor_release(target);
    linear_free(NULL);
    mlp_free(NULL);
    sgd_step(NULL);
    sgd_zero_grad(NULL);
    sgd_free(NULL);
}

static void write_u32(FILE *file, uint32_t value) {
    unsigned char bytes[] = {
        (unsigned char)(value >> 24), (unsigned char)(value >> 16),
        (unsigned char)(value >> 8), (unsigned char)value
    };
    assert(fwrite(bytes, 1, sizeof(bytes), file) == sizeof(bytes));
}

static void write_fixture(const char *images, const char *labels,
                          uint32_t image_count, uint32_t label_count,
                          uint32_t rows, uint32_t cols,
                          size_t image_bytes, size_t label_bytes, int bad_label) {
    FILE *file = fopen(images, "wb");
    assert(file);
    write_u32(file, 2051);
    write_u32(file, image_count);
    write_u32(file, rows);
    write_u32(file, cols);
    for (size_t i = 0; i < image_bytes; i++) assert(fputc(255, file) != EOF);
    assert(fclose(file) == 0);
    file = fopen(labels, "wb");
    assert(file);
    write_u32(file, 2049);
    write_u32(file, label_count);
    for (size_t i = 0; i < label_bytes; i++) {
        assert(fputc(bad_label ? 255 : 3, file) != EOF);
    }
    assert(fclose(file) == 0);
}

static void test_mnist_loader(void) {
    char directory[] = "/tmp/minigrad-test-XXXXXX";
    assert(mkdtemp(directory));
    char images[256], labels[256], missing[256];
    assert(snprintf(images, sizeof(images), "%s/images", directory) > 0);
    assert(snprintf(labels, sizeof(labels), "%s/labels", directory) > 0);
    assert(snprintf(missing, sizeof(missing), "%s/missing", directory) > 0);

    write_fixture(images, labels, 2, 2, 2, 2, 8, 2, 0);
    MNISTData *data = load_mnist(images, labels);
    assert(data && data->n_samples == 2 && data->input_dim == 4);
    for (int i = 0; i < 8; i++) assert(data->images[i] == 1.0f);
    assert(data->labels[0] == 3 && data->labels[1] == 3);
    mnist_free(data);

    assert(!load_mnist(NULL, labels));
    assert(!load_mnist(missing, labels));
    assert(!load_mnist(images, missing));
    write_fixture(images, labels, 2, 1, 2, 2, 8, 2, 0);
    assert(!load_mnist(images, labels));
    write_fixture(images, labels, 2, 2, 2, 2, 7, 2, 0);
    assert(!load_mnist(images, labels));
    write_fixture(images, labels, 2, 2, 2, 2, 8, 1, 0);
    assert(!load_mnist(images, labels));
    write_fixture(images, labels, 2, 2, 2, 2, 8, 2, 1);
    assert(!load_mnist(images, labels));
    write_fixture(images, labels, 0, 0, 2, 2, 0, 0, 0);
    assert(!load_mnist(images, labels));
    write_fixture(images, labels, 2, 2, 0, 2, 0, 0, 0);
    assert(!load_mnist(images, labels));
    write_fixture(images, labels, 2, 2, UINT32_MAX, UINT32_MAX, 0, 0, 0);
    assert(!load_mnist(images, labels));
    write_fixture(images, labels, UINT32_MAX, UINT32_MAX, 2, 2, 0, 0, 0);
    assert(!load_mnist(images, labels));
    FILE *file = fopen(images, "wb");
    assert(file && fputc(0, file) != EOF && fclose(file) == 0);
    assert(!load_mnist(images, labels));

    assert(remove(images) == 0);
    assert(remove(labels) == 0);
    assert(rmdir(directory) == 0);
    mnist_free(NULL);
}

int main(void) {
    test_deep_graph();
    test_operation_errors();
    test_softmax_and_loss_errors();
    test_model_lifetimes();
    test_mnist_loader();
    puts("All safety and lifetime tests passed!");
    return 0;
}
