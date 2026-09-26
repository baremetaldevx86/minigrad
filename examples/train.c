#include <stdio.h>
#include <stdlib.h>
#include "engine.h"
#include "nn.h"
#include "loss.h"
#include "optim.h"

// Create a dataset: y = 3x + 2
void create_dataset(Tensor** X, Tensor** Y, int n) {
    *X = tensor_create_matrix(n, 1);
    *Y = tensor_create_matrix(n, 1);

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

    // Create dataset
    Tensor* X;
    Tensor* Y;
    create_dataset(&X, &Y, n_samples);

    // Create model: Linear(1 → 1)
    Linearlayer* model = linear_create(1, 1);

    // Get parameters for optimizer
    int n_params;
    Tensor** params = linear_params(model, &n_params);

    // Create optimizer
    SGD* opt = sgd_create(params, n_params, lr);

    // Training loop
    for (int epoch = 0; epoch < epochs; epoch++) {
        // Forward
        Tensor* y_pred = linear_forward(model, X);
        Tensor* loss = mse_loss(y_pred, Y);

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
    Tensor* tx = tensor_create_matrix(1,1);
    tx->data[0] = test_x;

    Tensor* pred = linear_forward(model, tx);
    printf("\nPrediction for x=10: %f (expected ~32)\n", pred->data[0]);

    // Cleanup
    tensor_release(tx);
    tensor_release(pred);
    tensor_release(X);
    tensor_release(Y);
    sgd_free(opt);
    free(params);
    linear_free(model);

    return 0;
}

