#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "engine.h"
#include "nn.h"
#include "mlp.h"
#include "loss.h"
#include "optim.h"
#include "mnist_loader.h"

/* ================= CONFIG ================= */

#define BATCH_SIZE     32
#define BASE_LR        0.02f
#define EPOCHS         20
#define EVAL_BATCH     100
#define SEED           42

/* ========================================= */

/* Shuffle index array (Fisher–Yates) */
static void shuffle_indices(int* idx, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = idx[i];
        idx[i] = idx[j];
        idx[j] = tmp;
    }
}

/* Compute accuracy on arbitrary dataset */
static float compute_accuracy(MLP* model, MNISTData* data) {
    int correct = 0;
    int n = data->n_samples;
    int prior_mode = grad_set_enabled(0);
    float accuracy = NAN;

    Tensor* X = tensor_create_matrix_ex(EVAL_BATCH, 784, 0);
    if (!X) goto cleanup;
    for (int i = 0; i < n; ) {
        int bs = (n - i < EVAL_BATCH) ? (n - i) : EVAL_BATCH;

        /* Fill batch */
        for (int b = 0; b < bs; b++) {
            for (int k = 0; k < 784; k++) {
                X->data[b * 784 + k] =
                    data->images[(size_t)(i + b) * 784 + (size_t)k];
            }
        }

        /* Forward in no-grad mode: evaluation builds no parameter graph. */
        Tensor* logits = mlp_forward(model, X, 1);
        if (!logits) goto cleanup;

        for (int b = 0; b < bs; b++) {
            int pred = 0;
            float maxv = logits->data[b * 10];

            for (int k = 1; k < 10; k++) {
                float v = logits->data[b * 10 + k];
                if (v > maxv) {
                    maxv = v;
                    pred = k;
                }
            }

            if (pred == data->labels[i + b])
                correct++;
        }

        tensor_release(logits);
        i += bs;
    }

    accuracy = (float)correct / (float)n;
cleanup:
    tensor_release(X);
    grad_set_enabled(prior_mode);
    return accuracy;
}

/* =================== MAIN =================== */

int main(void) {
    int status = EXIT_FAILURE;
    MLP* model = NULL;
    Tensor** params = NULL;
    SGD* opt = NULL;
    Tensor* X = NULL;
    Tensor* Y = NULL;
    int* indices = NULL;
    srand(SEED);

    printf("Loading MNIST data...\n");

    MNISTData* train =
        load_mnist("data/train-images-idx3-ubyte",
                   "data/train-labels-idx1-ubyte");

    MNISTData* test =
        load_mnist("data/t10k-images-idx3-ubyte",
                   "data/t10k-labels-idx1-ubyte");

    if (!train || !test) {
        fprintf(stderr, "Failed to load MNIST data\n");
        goto cleanup;
    }
    // The loader supports arbitrary IDX dimensions; this model requires 784.
    if (train->input_dim != 784 || test->input_dim != 784 ||
        train->n_samples < BATCH_SIZE || test->n_samples <= 0) {
        fprintf(stderr, "Expected 28x28 data and at least one full training batch\n");
        goto cleanup;
    }

    printf("Loaded MNIST data: %d train, %d test\n",
           train->n_samples, test->n_samples);

    /* Model: 784 → 256 → 128 → 10 */
    int layers[] = {784, 256, 128, 10};
    model = mlp_create(layers, 3);
    if (!model) goto cleanup;

    int n_params = 0;
    params = mlp_params(model, &n_params);
    if (!params) goto cleanup;

    opt = sgd_create(params, n_params, BASE_LR);
    if (!opt) goto cleanup;

    printf("Model created. Parameters: %d\n", mlp_count_scalar_params(model));
    printf("Training for %d epochs | batch=%d | lr=%.4f\n",
           EPOCHS, BATCH_SIZE, BASE_LR);

    /* Training buffers */
    X = tensor_create_matrix_ex(BATCH_SIZE, 784, 0);
    Y = tensor_create_matrix_ex(BATCH_SIZE, 10, 0);
    if (!X || !Y) goto cleanup;

    /* Loader already verified n_samples * sizeof(int) fits size_t. */
    indices = malloc((size_t)train->n_samples * sizeof(*indices));
    if (!indices) {
        fprintf(stderr, "Failed to allocate shuffle indices\n");
        goto cleanup;
    }
    for (int i = 0; i < train->n_samples; i++)
        indices[i] = i;

    /* ================= TRAIN LOOP ================= */

    for (int epoch = 0; epoch < EPOCHS; epoch++) {

        // LR decay
        float lr = BASE_LR;
        if (epoch >= 25)
            lr *= 0.01f;
        else if (epoch >= 15)
            lr *= 0.1f;
        sgd_set_lr(opt, lr);

        shuffle_indices(indices, train->n_samples);

        float epoch_loss = 0.0f;
        int n_batches = train->n_samples / BATCH_SIZE;

        for (int b = 0; b < n_batches; b++) {
            int base = b * BATCH_SIZE;

            /* X */
            for (int i = 0; i < BATCH_SIZE; i++) {
                int idx = indices[base + i];
                for (int k = 0; k < 784; k++) {
                    X->data[i * 784 + k] =
                        train->images[(size_t)idx * 784 + (size_t)k];
                }
            }

            /* Y (one-hot) */
            for (int i = 0; i < BATCH_SIZE * 10; i++)
                Y->data[i] = 0.0f;

            for (int i = 0; i < BATCH_SIZE; i++) {
                int idx = indices[base + i];
                Y->data[i * 10 + train->labels[idx]] = 1.0f;
            }

            Tensor* logits = mlp_forward(model, X, 1);
            if (!logits) goto cleanup;
            Tensor* loss = cross_entropy_loss(logits, Y);
            if (!loss) {
                tensor_release(logits);
                goto cleanup;
            }

            epoch_loss += loss->data[0] * BATCH_SIZE;

            sgd_zero_grad(opt);
            tensor_backward(loss);
            sgd_step(opt);

            tensor_release(logits);
            tensor_release(loss);

            if (b % 100 == 0) {
                printf("\rEpoch %d [%d/%d] loss=%.4f lr=%.5f",
                       epoch + 1, b, n_batches,
                       epoch_loss / ((b + 1) * BATCH_SIZE),
                       lr);
                fflush(stdout);
            }
        }

        epoch_loss /= (n_batches * BATCH_SIZE);
        printf("\nEpoch %d finished. Avg loss: %.4f\n",
               epoch + 1, epoch_loss);
    }

    /* ================= EVAL ================= */

    printf("Computing final accuracy...\n");
    float acc = compute_accuracy(model, test);
    if (!isfinite(acc)) goto cleanup;
    printf("Final Test Accuracy: %.2f%%\n", acc * 100.0f);
    status = EXIT_SUCCESS;

    /* ================= CLEANUP ================= */
cleanup:
    free(indices);
    tensor_release(X);
    tensor_release(Y);
    sgd_free(opt);
    free(params);
    mlp_free(model);
    mnist_free(train);
    mnist_free(test);

    return status;
}
