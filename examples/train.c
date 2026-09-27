#include <stdio.h>
#include <stdlib.h>
#include "engine.h"
#include "nn.h"
#include "loss.h"
#include "optim.h"

// Create a dataset: y = 3x + 2
void create_dataset(Tensor** X, Tensor** Y, int n) {
    *X = tensor_create_matrix_ex(n, 1, 0);
    *Y = tensor_create_matrix_ex(n, 1, 0);
    if (!*X || !*Y) return;

    for (int i = 0; i < n; i++) {
        float x = (float)i;
        float y = 3.0f * x + 2.0f;

        (*X)->data[i] = x;
        (*Y)->data[i] = y;
    }
}

int main() {
    int n_samples = 50;
    int epochs = 10000;
    float lr = 0.001f;

    int status = EXIT_FAILURE;
    int eval_mode = -1;
    Tensor* X = NULL;
    Tensor* Y = NULL;
    Tensor* tx = NULL;
    Tensor* pred = NULL;
    Tensor** params = NULL;
    SGD* opt = NULL;
    Linearlayer* model = NULL;

    create_dataset(&X, &Y, n_samples);
    if (!X || !Y) goto cleanup;

    // Create model: Linear(1 → 1)
    model = linear_create(1, 1);
    if (!model) goto cleanup;

    // Get parameters for optimizer
    int n_params;
    params = linear_params(model, &n_params);
    if (!params) goto cleanup;

    // Create optimizer
    opt = sgd_create(params, n_params, lr);
    if (!opt) goto cleanup;

    // Training loop
    for (int epoch = 0; epoch < epochs; epoch++) {
        // Forward
        Tensor* y_pred = linear_forward(model, X);
        if (!y_pred) goto cleanup;
        Tensor* loss = mse_loss(y_pred, Y);
        if (!loss) {
            tensor_release(y_pred);
            goto cleanup;
        }

        // Zero gradients
        sgd_zero_grad(opt);

        // Backward
        tensor_backward(loss);

        // Update weights
        sgd_step(opt);

        // Print loss every 50 epochs
        if (epoch % 50 == 0) {
            printf("Epoch %d | Loss = %f\n", epoch, *(loss->data));
        }

        // Release graph
        tensor_release(y_pred);
        tensor_release(loss);
    }

    // Final learned parameters
    printf("\nLearned parameters:\n");
    printf("W = %f\n", model->W->data[0]);
    printf("b = %f\n", model->b->data[0]);

    // Test prediction
    float test_x = 10.0f;
    tx = tensor_create_matrix_ex(1, 1, 0);
    if (!tx) goto cleanup;
    tx->data[0] = test_x;

    eval_mode = grad_set_enabled(0);
    pred = linear_forward(model, tx);
    if (!pred) goto cleanup;
    printf("\nPrediction for x=10: %f (expected ~32)\n", pred->data[0]);

    status = EXIT_SUCCESS;

cleanup:
    if (eval_mode >= 0) grad_set_enabled(eval_mode);
    tensor_release(tx);
    tensor_release(pred);
    tensor_release(X);
    tensor_release(Y);
    sgd_free(opt);
    free(params);
    linear_free(model);

    return status;
}

