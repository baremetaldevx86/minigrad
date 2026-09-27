#include "loss.h"

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/*
 * Mean squared error is built from engine operations so it participates in
 * the normal computation graph.  The engine returns NULL for invalid shapes;
 * preserve that result rather than dereferencing an intermediate.
 */
Tensor *mse_loss(Tensor *y_pred, Tensor *y_true) {
    if (!y_pred || !y_true) {
        fprintf(stderr, "minigrad: mse_loss: inputs must not be NULL\n");
        return NULL;
    }

    Tensor *diff = tensor_sub(y_pred, y_true);
    if (!diff) return NULL;

    Tensor *squared = tensor_mul(diff, diff);
    if (!squared) {
        tensor_release(diff);
        return NULL;
    }

    Tensor *loss = tensor_mean(squared);
    tensor_release(diff);
    tensor_release(squared);
    return loss;
}

/* ============================================================
 * Cross entropy loss (fused stable softmax + log)
 * ============================================================ */

static void cross_entropy_backward(Tensor *loss) {
    if (!loss || !loss->parents || loss->n_parents != 2 ||
        !loss->parents[0] || !loss->parents[1] || !loss->grad) {
        return;
    }

    Tensor *logits = loss->parents[0];
    Tensor *targets = loss->parents[1];
    if (logits->ndim != 2 || targets->ndim != 2 || !logits->shape ||
        !targets->shape || !logits->data || !targets->data || !logits->grad ||
        logits->shape[0] <= 0 || logits->shape[1] <= 0 ||
        targets->shape[0] != logits->shape[0] ||
        targets->shape[1] != logits->shape[1] ||
        logits->shape[0] > INT_MAX / logits->shape[1] ||
        logits->size != logits->shape[0] * logits->shape[1] ||
        targets->size != logits->size) {
        return;
    }

    int batch_size = logits->shape[0];
    int n_classes = logits->shape[1];

    for (int i = 0; i < batch_size; i++) {
        float max_value = -INFINITY;
        for (int j = 0; j < n_classes; j++) {
            float value = logits->data[i * n_classes + j];
            if (!isfinite(value)) return;
            if (value > max_value) max_value = value;
        }

        float sum_exp = 0.0f;
        for (int j = 0; j < n_classes; j++) {
            sum_exp += expf(logits->data[i * n_classes + j] - max_value);
        }
        if (!(sum_exp > 0.0f) || !isfinite(sum_exp)) return;

        float target_sum = 0.0f;
        for (int j = 0; j < n_classes; j++) {
            float target = targets->data[i * n_classes + j];
            if (!isfinite(target)) return;
            target_sum += target;
        }
        for (int j = 0; j < n_classes; j++) {
            float target = targets->data[i * n_classes + j];
            float probability =
                expf(logits->data[i * n_classes + j] - max_value) / sum_exp;
            float update = (probability * target_sum - target) * loss->grad[0] /
                           (float)batch_size;
            logits->grad[i * n_classes + j] += update;
        }
    }
}

Tensor *cross_entropy_loss(Tensor *logits, Tensor *targets) {
    if (!logits || !targets || logits->ndim != 2 || targets->ndim != 2 ||
        !logits->shape || !targets->shape || !logits->data ||
        !targets->data || logits->shape[0] <= 0 || logits->shape[1] <= 0 ||
        targets->shape[0] != logits->shape[0] ||
        targets->shape[1] != logits->shape[1] ||
        logits->shape[0] > INT_MAX / logits->shape[1] ||
        logits->size != logits->shape[0] * logits->shape[1] ||
        targets->size != logits->size) {
        fprintf(stderr,
                "minigrad: cross_entropy_loss: expected matching non-empty 2D inputs\n");
        return NULL;
    }

    int batch_size = logits->shape[0];
    int n_classes = logits->shape[1];
    float total_loss = 0.0f;

    for (int i = 0; i < batch_size; i++) {
        float max_value = -INFINITY;
        for (int j = 0; j < n_classes; j++) {
            float value = logits->data[i * n_classes + j];
            if (!isfinite(value)) {
                fprintf(stderr,
                        "minigrad: cross_entropy_loss: logits must be finite\n");
                return NULL;
            }
            if (value > max_value) max_value = value;
        }

        float sum_exp = 0.0f;
        for (int j = 0; j < n_classes; j++) {
            sum_exp += expf(logits->data[i * n_classes + j] - max_value);
        }
        if (!(sum_exp > 0.0f) || !isfinite(sum_exp)) {
            fprintf(stderr,
                    "minigrad: cross_entropy_loss: invalid softmax normalization\n");
            return NULL;
        }

        float log_sum_exp = logf(sum_exp);
        for (int j = 0; j < n_classes; j++) {
            float target = targets->data[i * n_classes + j];
            if (!isfinite(target) || target < 0.0f) {
                fprintf(stderr,
                        "minigrad: cross_entropy_loss: targets must be finite and non-negative\n");
                return NULL;
            }
            float log_probability =
                (logits->data[i * n_classes + j] - max_value) - log_sum_exp;
            total_loss -= target * log_probability;
        }
    }

    if (!isfinite(total_loss)) {
        fprintf(stderr, "minigrad: cross_entropy_loss: non-finite loss\n");
        return NULL;
    }
    Tensor *loss = tensor_create(total_loss / (float)batch_size);
    if (!loss) return NULL;

    loss->parents = (Tensor **)malloc(2 * sizeof(*loss->parents));
    if (!loss->parents) {
        tensor_release(loss);
        fprintf(stderr, "minigrad: cross_entropy_loss: out of memory\n");
        return NULL;
    }
    loss->parents[0] = logits;
    loss->parents[1] = targets;
    loss->n_parents = 2;
    tensor_retain(logits);
    tensor_retain(targets);
    loss->backward = cross_entropy_backward;
    return loss;
}
