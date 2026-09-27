#include "loss.h"
#include "engine_internal.h"

#include <float.h>
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
    Tensor *logits = loss->parents[0];
    Tensor *targets = loss->parents[1];
    int batch_size = logits->shape[0];
    int n_classes = logits->shape[1];
    double seed = (double)loss->grad[0] / (double)batch_size;

    for (int i = 0; i < batch_size; i++) {
        int row = i * n_classes;
        double max_value = -INFINITY;
        double target_sum = 0.0;
        for (int j = 0; j < n_classes; j++) {
            double value = logits->data[row + j];
            if (value > max_value) max_value = value;
            target_sum += targets->data[row + j];
        }
        double sum_exp = 0.0;
        for (int j = 0; j < n_classes; j++)
            sum_exp += exp((double)logits->data[row + j] - max_value);
        double log_sum_exp = log(sum_exp);

        for (int j = 0; j < n_classes; j++) {
            int k = row + j;
            if (logits->requires_grad && logits->grad) {
                double probability = exp((double)logits->data[k] - max_value) / sum_exp;
                logits->grad[k] += (float)(seed *
                    (probability * target_sum - (double)targets->data[k]));
            }
            if (targets->requires_grad && targets->grad) {
                double log_probability =
                    (double)logits->data[k] - max_value - log_sum_exp;
                targets->grad[k] -= (float)(seed * log_probability);
            }
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
    double total_loss = 0.0;

    for (int i = 0; i < batch_size; i++) {
        int row = i * n_classes;
        double max_value = -INFINITY;
        for (int j = 0; j < n_classes; j++) {
            double value = logits->data[row + j];
            if (!isfinite(value)) {
                fprintf(stderr,
                        "minigrad: cross_entropy_loss: logits must be finite\n");
                return NULL;
            }
            if (value > max_value) max_value = value;
        }

        double sum_exp = 0.0;
        for (int j = 0; j < n_classes; j++)
            sum_exp += exp((double)logits->data[row + j] - max_value);
        double log_sum_exp = log(sum_exp);
        for (int j = 0; j < n_classes; j++) {
            double target = targets->data[row + j];
            if (!isfinite(target) || target < 0.0) {
                fprintf(stderr,
                        "minigrad: cross_entropy_loss: targets must be finite and non-negative\n");
                return NULL;
            }
            double log_probability =
                (double)logits->data[row + j] - max_value - log_sum_exp;
            total_loss -= target * log_probability;
        }
    }

    double mean_loss = total_loss / (double)batch_size;
    if (!isfinite(mean_loss) || mean_loss > FLT_MAX || mean_loss < -FLT_MAX) {
        fprintf(stderr, "minigrad: cross_entropy_loss: loss exceeds float range\n");
        return NULL;
    }
    Tensor *loss = tensor_create_ex((float)mean_loss, 0);
    if (!loss) return NULL;
    Tensor *parents[] = {logits, targets};
    if (!tensor_attach_operation(loss, parents, 2, cross_entropy_backward,
                                 "cross_entropy")) {
        tensor_release(loss);
        return NULL;
    }
    return loss;
}
