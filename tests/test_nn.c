#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#include "mlp.h"
#include "nn.h"
#include "rng.h"

static void test_rng(void) {
    MinigradRNG a, b;
    rng_seed(&a, 0);
    assert(rng_uniform(&a) == 0xe220a8 * (1.0f / 16777216.0f));
    assert(rng_uniform(&a) == 0x6e789e * (1.0f / 16777216.0f));
    rng_seed(&a, UINT64_C(0xffffffffffffffff));
    rng_seed(&b, UINT64_C(0xffffffffffffffff));
    for (int i = 0; i < 1000; i++) {
        float u = rng_uniform(&a);
        assert(u >= 0.0f && u < 1.0f);
        assert(u == rng_uniform(&b));
    }
    rng_seed(&a, 17);
    rng_seed(&b, 17);
    for (int i = 0; i < 21; i++) {
        float normal = rng_normal(&a);
        assert(isfinite(normal));
        assert(normal == rng_normal(&b));
    }
    /* Copying state also copies the cached second normal variate. */
    b = a;
    for (int i = 0; i < 10; i++) assert(rng_normal(&a) == rng_normal(&b));
    rng_seed(&a, 17);
    float initial = rng_normal(&a);
    (void)rng_normal(&a);
    (void)rng_normal(&a); /* Leave a cached normal that must be discarded. */
    rng_seed(&a, 17);
    assert(rng_normal(&a) == initial);
    rng_seed(NULL, 0);
    assert(rng_uniform(NULL) == 0.0f && rng_normal(NULL) == 0.0f);
}

static float legacy_normal(void) {
    float u1 = (float)(((double)rand() + 1.0) / ((double)RAND_MAX + 1.0));
    float u2 = (float)(((double)rand() + 1.0) / ((double)RAND_MAX + 1.0));
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265358979323846f * u2);
}

static void test_legacy_compatibility(void) {
    srand(2024);
    float expected[6];
    for (int i = 0; i < 6; i++) expected[i] = legacy_normal() * sqrtf(2.0f / 2.0f);
    int next_rand = rand();
    srand(2024);
    Linearlayer *legacy = linear_create(2, 3);
    assert(legacy);
    for (int i = 0; i < 6; i++) assert(legacy->W->data[i] == expected[i]);
    assert(rand() == next_rand);
    linear_free(legacy);

    srand(2024);
    Linearlayer *nullable = linear_create_ex(2, 3, INIT_HE, NULL);
    assert(nullable);
    for (int i = 0; i < 6; i++) assert(nullable->W->data[i] == expected[i]);
    assert(rand() == next_rand);
    linear_free(nullable);
}

static void test_linear_initializers(void) {
    MinigradRNG a, b, c;
    rng_seed(&a, 423);
    rng_seed(&b, 423);
    rng_seed(&c, 423);
    srand(17);
    int expected_rand = rand();
    srand(17);
    Linearlayer *he = linear_create_ex(3, 5, INIT_HE, &a);
    Linearlayer *again = linear_create_ex(3, 5, INIT_HE, &b);
    Linearlayer *xavier = linear_create_ex(3, 5, INIT_XAVIER, &c);
    assert(he && again && xavier);
    assert(rand() == expected_rand); /* explicit RNG never touches global rand */
    float ratio = (float)(sqrt(2.0 / (3.0 + 5.0)) / sqrtf(2.0f / 3.0f));
    for (int i = 0; i < 15; i++) {
        assert(he->W->data[i] == again->W->data[i]);
        assert(fabsf(xavier->W->data[i] - he->W->data[i] * ratio) < 2e-7f);
    }
    for (int i = 0; i < 5; i++) {
        assert(he->b->data[i] == 0.0f && xavier->b->data[i] == 0.0f);
    }
    assert(he->W->ref_count == 1 && he->b->ref_count == 1);
    int count = 0;
    Tensor **params = linear_params(he, &count);
    assert(params && count == 2 && params[0] == he->W && params[1] == he->b);
    free(params);
    linear_free(he);
    linear_free(again);
    linear_free(xavier);

    rng_seed(&a, 32);
    uint64_t state = a.state;
    assert(!linear_create_ex(0, 3, INIT_HE, &a));
    assert(!linear_create_ex(2, 0, INIT_HE, &a));
    assert(!linear_create_ex(INT_MAX, 2, INIT_HE, &a));
    assert(!linear_create_ex(2, 3, (WeightInit)-1, &a));
    assert(!linear_create_ex(2, 3, (WeightInit)200, &a));
    assert(a.state == state);
}

static void test_mlp_initializers(void) {
    int sizes[] = {3, 4, 2};
    MinigradRNG a, b, c;
    rng_seed(&a, 12);
    rng_seed(&b, 12);
    rng_seed(&c, 12);
    srand(99);
    int expected_rand = rand();
    srand(99);
    MLP *model = mlp_create_ex(sizes, 2, INIT_XAVIER, &a);
    MLP *copy = mlp_create_ex(sizes, 2, INIT_XAVIER, &b);
    Linearlayer *one = linear_create_ex(3, 4, INIT_XAVIER, &c);
    Linearlayer *two = linear_create_ex(4, 2, INIT_XAVIER, &c);
    assert(model && copy && one && two);
    assert(rand() == expected_rand);
    assert(model->n_layers == 2 && mlp_count_scalar_params(model) == 26);
    for (int layer = 0; layer < 2; layer++) {
        Linearlayer *standalone = layer == 0 ? one : two;
        for (int i = 0; i < model->layers[layer]->W->size; i++) {
            assert(model->layers[layer]->W->data[i] == copy->layers[layer]->W->data[i]);
            assert(model->layers[layer]->W->data[i] == standalone->W->data[i]);
        }
    }
    int n_params = -1;
    Tensor **params = mlp_params(model, &n_params);
    assert(params && n_params == 4 && params[0] == model->layers[0]->W &&
           params[3] == model->layers[1]->b);
    free(params);

    Tensor *input = tensor_create_matrix(2, 3);
    assert(input);
    Tensor *output = mlp_forward(model, input, 1);
    assert(output && output->ndim == 2 && output->shape[0] == 2 && output->shape[1] == 2);
    mlp_free(model); /* graph retains the model's parameters */
    tensor_backward(output);
    tensor_release(output);
    tensor_release(input);
    mlp_free(copy);
    linear_free(one);
    linear_free(two);

    int bad[] = {3, 4, 0};
    rng_seed(&a, 123);
    assert(!mlp_create_ex(NULL, 2, INIT_HE, &a));
    assert(!mlp_create_ex(sizes, 0, INIT_HE, &a));
    assert(!mlp_create_ex(sizes, 2, (WeightInit)100, &a));
    assert(!mlp_create_ex(bad, 2, INIT_XAVIER, &a));
    /* Invalid second layer frees the first and does not corrupt the RNG. */
    rng_seed(&b, 123);
    Linearlayer *first = linear_create_ex(3, 4, INIT_XAVIER, &b);
    assert(first && a.state == b.state && a.has_spare == b.has_spare);
    linear_free(first);
}

int main(void) {
    test_rng();
    test_legacy_compatibility();
    test_linear_initializers();
    test_mlp_initializers();
    return 0;
}
