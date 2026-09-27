#include <stdio.h>
#include <stdlib.h>
#include "engine.h"
#include "nn.h"
#include "mlp.h"
#include "loss.h"
#include "optim.h"

// XOR Dataset
// Input: (4x2)
// 0 0 -> 0
// 0 1 -> 1
// 1 0 -> 1
// 1 1 -> 0

int main(void) {
    int status = EXIT_FAILURE;
    int eval_mode = -1;
    MLP* model = NULL;
    Tensor** params = NULL;
    SGD* opt = NULL;
    Tensor* final_pred = NULL;
    // 1. Prepare Data
    Tensor* X = tensor_create_matrix_ex(4, 2, 0);
    Tensor* Y = tensor_create_matrix_ex(4, 1, 0);
    if (!X || !Y) goto cleanup;
    
    // Inputs
    X->data[0] = 0.0f; X->data[1] = 0.0f;
    X->data[2] = 0.0f; X->data[3] = 1.0f;
    X->data[4] = 1.0f; X->data[5] = 0.0f;
    X->data[6] = 1.0f; X->data[7] = 1.0f;
    
    // Targets
    Y->data[0] = 0.0f;
    Y->data[1] = 1.0f;
    Y->data[2] = 1.0f;
    Y->data[3] = 0.0f;
    
    // 2. Define Model
    // 2 inputs -> 4 hidden (Relu) -> 1 output (Linear/Sigmoid)
    // Note: We use raw linear output with MSE. A 4-neuron hidden layer is enough for XOR.
    int layer_sizes[] = {2, 4, 1};
    model = mlp_create(layer_sizes, 2); // 2 layers: {2->4, 4->1}
    if (!model) goto cleanup;
    
    // 3. Optimizer
    int n_params;
    params = mlp_params(model, &n_params);
    if (!params) goto cleanup;
    opt = sgd_create(params, n_params, 0.1f); // Learning rate 0.1
    if (!opt) goto cleanup;
    
    printf("Training XOR with MLP (2->4->1)...\n");
    
    // 4. Training Loop
    int epochs = 5000;
    for (int i = 0; i < epochs; i++) {
        // Forward
        // Use Tanh (0) for hidden layers (better convergence for small XOR net)
        Tensor* y_pred = mlp_forward(model, X, 0);
        if (!y_pred) goto cleanup;

        // Loss (MSE for regression-style training on 0/1)
        Tensor* loss = mse_loss(y_pred, Y);
        if (!loss) {
            tensor_release(y_pred);
            goto cleanup;
        }
        
        // Zero Grad
        sgd_zero_grad(opt);
        
        // Backward
        tensor_backward(loss);
        
        // Update
        sgd_step(opt);
        
        if (i % 500 == 0) {
            printf("Epoch %d, Loss: %f\n", i, *loss->data);
        }
        
        // Cleanup graph
        tensor_release(y_pred);
        tensor_release(loss);
    }
    
    // 5. Validation
    printf("\nOptimization Finished!\n");
    printf("Predictions:\n");
    eval_mode = grad_set_enabled(0);
    final_pred = mlp_forward(model, X, 0);
    if (!final_pred) goto cleanup;
    for (int i = 0; i < 4; i++) {
        float input1 = X->data[i*2];
        float input2 = X->data[i*2+1];
        float target = Y->data[i];
        float pred = final_pred->data[i];
        printf("In: [%.0f, %.0f] Target: %.0f Pred: %.4f\n", input1, input2, target, pred);
    }
    
    status = EXIT_SUCCESS;

cleanup:
    if (eval_mode >= 0) grad_set_enabled(eval_mode);
    tensor_release(final_pred);
    tensor_release(X);
    tensor_release(Y);
    sgd_free(opt);
    free(params);
    mlp_free(model);

    return status;
}
