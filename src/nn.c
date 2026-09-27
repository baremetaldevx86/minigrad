#include "nn.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* Box-Muller. The +1 keeps u1 nonzero; double arithmetic avoids overflow. */
static float rand_normal(void) {
    const float pi = 3.14159265358979323846f;
    float u1 = (float)(((double)rand() + 1.0) / ((double)RAND_MAX + 1.0));
    float u2 = (float)(((double)rand() + 1.0) / ((double)RAND_MAX + 1.0));
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * pi * u2);
}

static void init_weights(Tensor *weights, int in_features, int out_features,
                         WeightInit init, MinigradRNG *rng) {
    /* Compute fan sums in double to avoid overflowing a signed int. */
    float scale = init == INIT_HE ? sqrtf(2.0f / (float)in_features)
        : (float)sqrt(2.0 / ((double)in_features + (double)out_features));
    for (int i = 0; i < weights->size; i++) {
        weights->data[i] = (rng ? rng_normal(rng) : rand_normal()) * scale;
    }
}

Linearlayer *linear_create(int in_features, int out_features) {
    return linear_create_ex(in_features, out_features, INIT_HE, NULL);
}

Linearlayer *linear_create_ex(int in_features, int out_features,
                             WeightInit init, MinigradRNG *rng) {
    if (in_features <= 0 || out_features <= 0 ||
        (init != INIT_HE && init != INIT_XAVIER)) {
        fprintf(stderr, "minigrad: linear_create_ex: invalid dimensions or initializer\n");
        return NULL;
    }

    Linearlayer *layer = (Linearlayer *)calloc(1, sizeof(*layer));
    if (!layer) {
        fprintf(stderr, "minigrad: linear_create: out of memory\n");
        return NULL;
    }
    layer->in_features = in_features;
    layer->out_features = out_features;
    layer->W = tensor_create_matrix(in_features, out_features);
    if (!layer->W) {
        linear_free(layer);
        return NULL;
    }
    layer->b = tensor_create_matrix(1, out_features);
    if (!layer->b) {
        linear_free(layer);
        return NULL;
    }

    init_weights(layer->W, in_features, out_features, init, rng);
    /* tensor_create_matrix initializes the bias to zero. */
    return layer;
}

Tensor *linear_forward(Linearlayer *layer, Tensor *x) {
    if (!layer || !layer->W || !layer->b || !x || x->ndim != 2 ||
        !x->shape || x->shape[1] != layer->in_features) {
        fprintf(stderr, "minigrad: linear_forward: invalid layer or input shape\n");
        return NULL;
    }

    Tensor *product = tensor_matmul(x, layer->W);
    if (!product) return NULL;
    Tensor *result = tensor_add(product, layer->b);
    tensor_release(product);
    return result;
}

Tensor **linear_params(Linearlayer *layer, int *n_params) {
    if (n_params) *n_params = 0;
    if (!layer || !layer->W || !layer->b || !n_params) return NULL;

    Tensor **params = (Tensor **)malloc(2 * sizeof(*params));
    if (!params) {
        fprintf(stderr, "minigrad: linear_params: out of memory\n");
        return NULL;
    }
    params[0] = layer->W;
    params[1] = layer->b;
    *n_params = 2;
    return params;
}

void linear_free(Linearlayer *layer) {
    if (!layer) return;
    tensor_release(layer->W);
    tensor_release(layer->b);
    free(layer);
}
