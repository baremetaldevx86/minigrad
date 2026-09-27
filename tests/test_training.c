#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "debug.h"
#include "loss.h"
#include "mlp.h"
#include "optim.h"

static void test_microbatch_accumulation(void) {
    MinigradRNG ra, rb;
    rng_seed(&ra, 11);
    rng_seed(&rb, 11);
    int sizes[] = {2, 3, 1};
    MLP *full = mlp_create_ex(sizes, 2, INIT_XAVIER, &ra);
    MLP *micro = mlp_create_ex(sizes, 2, INIT_XAVIER, &rb);
    assert(full && micro);
    Tensor *x = tensor_create_matrix_ex(2, 2, 0);
    Tensor *y = tensor_create_matrix_ex(2, 1, 0);
    x->data[0] = 0.25f; x->data[1] = 1.0f;
    x->data[2] = -0.5f; x->data[3] = 0.75f;
    y->data[0] = 0.3f; y->data[1] = -0.4f;
    Tensor *out = mlp_forward(full, x, 0);
    Tensor *loss = mse_loss(out, y);
    assert(out && loss);
    tensor_backward(loss);
    tensor_release(loss);
    tensor_release(out);

    // Two mean-reduced microbatches, each weighted by its share of the batch.
    Tensor *seed = tensor_create_ex(0.5f, 0);
    for (int row = 0; row < 2; row++) {
        Tensor *mx = tensor_create_matrix_ex(1, 2, 0);
        Tensor *my = tensor_create_matrix_ex(1, 1, 0);
        mx->data[0] = x->data[row * 2];
        mx->data[1] = x->data[row * 2 + 1];
        my->data[0] = y->data[row];
        out = mlp_forward(micro, mx, 0);
        loss = mse_loss(out, my);
        assert(out && loss && tensor_backward_with_grad(loss, seed));
        tensor_release(loss);
        tensor_release(out);
        tensor_release(mx);
        tensor_release(my);
    }
    int nfull, nmicro;
    Tensor **pf = mlp_params(full, &nfull);
    Tensor **pm = mlp_params(micro, &nmicro);
    assert(pf && pm && nfull == nmicro);
    for (int i = 0; i < nfull; i++) {
        for (int j = 0; j < pf[i]->size; j++) {
            assert(fabsf(pf[i]->grad[j] - pm[i]->grad[j]) < 1e-6f);
        }
    }
    free(pf); free(pm);
    tensor_release(seed);
    tensor_release(x); tensor_release(y);
    mlp_free(full); mlp_free(micro);
}

static void test_xor_with_adam(void) {
    MinigradRNG rng;
    rng_seed(&rng, 42);
    int sizes[] = {2, 8, 1};
    MLP *model = mlp_create_ex(sizes, 2, INIT_XAVIER, &rng);
    assert(model);
    int count;
    Tensor **params = mlp_params(model, &count);
    assert(params);
    Adam *opt = adam_create(params, count, 0.03f);
    assert(opt);
    Tensor *x = tensor_create_matrix_ex(4, 2, 0);
    Tensor *y = tensor_create_matrix_ex(4, 1, 0);
    const float inputs[] = {0, 0, 0, 1, 1, 0, 1, 1};
    const float targets[] = {0, 1, 1, 0};
    for (int i = 0; i < 8; i++) x->data[i] = inputs[i];
    for (int i = 0; i < 4; i++) y->data[i] = targets[i];
    float final_loss = INFINITY;
    for (int step = 0; step < 1500; step++) {
        Tensor *out = mlp_forward(model, x, 0);
        Tensor *loss = mse_loss(out, y);
        assert(out && loss && isfinite(loss->data[0]));
        final_loss = loss->data[0];
        adam_zero_grad(opt);
        tensor_backward(loss);
        assert(tensor_check_finite(loss, 1));
        assert(isfinite(optim_clip_grad_norm(params, count, 1.0f)));
        adam_step(opt);
        tensor_release(loss);
        tensor_release(out);
        assert(x->ref_count == 1 && !x->grad && !y->grad);
    }
    assert(final_loss < 1e-4f);
    int previous = grad_set_enabled(0);
    Tensor *prediction = mlp_forward(model, x, 0);
    grad_set_enabled(previous);
    assert(prediction && !prediction->grad && !prediction->parents);
    for (int i = 0; i < 4; i++) {
        assert(fabsf(prediction->data[i] - targets[i]) < 0.05f);
    }
    tensor_release(prediction);
    tensor_release(x); tensor_release(y);
    adam_free(opt);
    free(params);
    mlp_free(model);
}

int main(void) {
    TensorMemoryStats before = tensor_memory_stats();
    test_microbatch_accumulation();
    test_xor_with_adam();
    TensorMemoryStats after = tensor_memory_stats();
    assert(before.live_bytes == after.live_bytes);
    assert(before.live_tensors == after.live_tensors);
    puts("Microbatch accumulation and Adam XOR training passed");
    return 0;
}
