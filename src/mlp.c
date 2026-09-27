#include "mlp.h"

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

MLP *mlp_create(int *layer_sizes, int n_layers) {
    if (!layer_sizes || n_layers <= 0 ||
        (size_t)n_layers > SIZE_MAX / sizeof(Linearlayer *)) {
        fprintf(stderr, "minigrad: mlp_create: invalid layer configuration\n");
        return NULL;
    }

    MLP *mlp = (MLP *)calloc(1, sizeof(*mlp));
    if (!mlp) {
        fprintf(stderr, "minigrad: mlp_create: out of memory\n");
        return NULL;
    }
    mlp->n_layers = n_layers;
    mlp->layers = (Linearlayer **)calloc((size_t)n_layers, sizeof(*mlp->layers));
    if (!mlp->layers) {
        fprintf(stderr, "minigrad: mlp_create: out of memory\n");
        mlp_free(mlp);
        return NULL;
    }

    for (int i = 0; i < n_layers; i++) {
        if (layer_sizes[i] <= 0 || layer_sizes[i + 1] <= 0) {
            fprintf(stderr, "minigrad: mlp_create: layer sizes must be positive\n");
            mlp_free(mlp);
            return NULL;
        }
        mlp->layers[i] = linear_create(layer_sizes[i], layer_sizes[i + 1]);
        if (!mlp->layers[i]) {
            mlp_free(mlp);
            return NULL;
        }
    }
    return mlp;
}

Tensor *mlp_forward(MLP *mlp, Tensor *x, int use_relu) {
    if (!mlp || !mlp->layers || mlp->n_layers <= 0 || !x) {
        fprintf(stderr, "minigrad: mlp_forward: invalid model or input\n");
        return NULL;
    }

    Tensor *current = x;
    tensor_retain(current);
    for (int i = 0; i < mlp->n_layers; i++) {
        Tensor *z = linear_forward(mlp->layers[i], current);
        tensor_release(current);
        if (!z) return NULL;

        if (i == mlp->n_layers - 1) return z;

        current = use_relu ? tensor_relu(z) : tensor_Tanh(z);
        tensor_release(z);
        if (!current) return NULL;
    }
    return NULL;
}

Tensor **mlp_params(MLP *mlp, int *n_params) {
    if (n_params) *n_params = 0;
    if (!n_params || !mlp || !mlp->layers || mlp->n_layers <= 0 ||
        mlp->n_layers > INT_MAX / 2 ||
        (size_t)mlp->n_layers > SIZE_MAX / (2 * sizeof(Tensor *))) {
        return NULL;
    }

    int count = mlp->n_layers * 2;
    Tensor **params = (Tensor **)malloc((size_t)count * sizeof(*params));
    if (!params) {
        fprintf(stderr, "minigrad: mlp_params: out of memory\n");
        return NULL;
    }
    for (int i = 0; i < mlp->n_layers; i++) {
        if (!mlp->layers[i] || !mlp->layers[i]->W || !mlp->layers[i]->b) {
            free(params);
            return NULL;
        }
        params[i * 2] = mlp->layers[i]->W;
        params[i * 2 + 1] = mlp->layers[i]->b;
    }
    *n_params = count;
    return params;
}

int mlp_count_scalar_params(MLP *mlp) {
    if (!mlp || !mlp->layers || mlp->n_layers <= 0) return 0;
    int total = 0;
    for (int i = 0; i < mlp->n_layers; i++) {
        Linearlayer *layer = mlp->layers[i];
        if (!layer || !layer->W || !layer->b || layer->W->size < 0 ||
            layer->b->size < 0 || layer->W->size > INT_MAX - total ||
            layer->b->size > INT_MAX - total - layer->W->size) {
            fprintf(stderr, "minigrad: mlp_count_scalar_params: count overflow or invalid model\n");
            return -1;
        }
        total += layer->W->size + layer->b->size;
    }
    return total;
}

void mlp_free(MLP *mlp) {
    if (!mlp) return;
    if (mlp->layers) {
        for (int i = 0; i < mlp->n_layers; i++) {
            linear_free(mlp->layers[i]);
        }
    }
    free(mlp->layers);
    free(mlp);
}
