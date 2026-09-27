#include "optim.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int near(float actual, double expected, double tolerance) {
    return fabs((double)actual - expected) <= tolerance;
}

static void test_validation_and_borrowing(void) {
    assert(!sgd_create(NULL, -1, 0.1f));
    assert(!adam_create(NULL, 1, 0.1f));
    assert(!adamw_create(NULL, 0, -1.0f, 0.0f));
    assert(!adamw_create(NULL, 0, 0.1f, NAN));
    assert(!adamw_create(NULL, 0, 0.1f, -1.0f));
    assert(!sgd_create(NULL, 0, INFINITY));
    assert(!sgd_create_momentum(NULL, 0, 0.1f, 1.0f));
    assert(!sgd_create_momentum(NULL, 0, 0.1f, -0.1f));
    assert(!sgd_create_momentum(NULL, 0, 0.1f, NAN));
    assert(isnan(optim_clip_grad_norm(NULL, 1, 1.0f)));
    assert(isnan(optim_clip_grad_norm(NULL, 0, -1.0f)));
    assert(isnan(optim_clip_grad_norm(NULL, 0, INFINITY)));
    assert(optim_clip_grad_norm(NULL, 0, 0.0f) == 0.0f);

    SGD *empty = sgd_create(NULL, 0, 0.0f);
    Adam *empty_adam = adam_create(NULL, 0, 0.0f);
    assert(empty && empty_adam);
    sgd_step(empty);
    adam_step(empty_adam);
    sgd_free(empty);
    adam_free(empty_adam);

    Tensor *p = tensor_create(2.0f);
    Tensor *params[] = {p};
    Tensor *dupes[] = {p, p};
    Tensor *invalid[] = {NULL};
    assert(!sgd_create(invalid, 1, 0.1f));
    assert(!adam_create(invalid, 1, 0.1f));
    assert(!sgd_create(dupes, 2, 0.1f));
    assert(!sgd_create_momentum(dupes, 2, 0.1f, 0.5f));
    assert(!adam_create(dupes, 2, 0.1f));
    assert(isnan(optim_clip_grad_norm(dupes, 2, 1.0f)));
    SGD *sgd = sgd_create(params, 1, 0.5f);
    Adam *adam = adam_create(params, 1, 0.01f);
    assert(sgd && adam && sgd->params == params);
    p->grad[0] = 3.0f;
    sgd_set_lr(sgd, NAN);
    assert(sgd->lr == 0.5f);
    sgd_step(sgd);
    assert(near(p->data[0], 0.5, 1e-7));
    sgd_zero_grad(sgd);
    assert(p->grad[0] == 0.0f);
    adam_zero_grad(adam);
    adam_set_lr(adam, NAN); /* invalid change ignored */
    adam_free(adam);
    sgd_free(sgd);
    assert(params[0] == p && p->ref_count == 1);
    tensor_release(p);
    sgd_step(NULL);
    sgd_zero_grad(NULL);
    sgd_free(NULL);
    adam_step(NULL);
    adam_zero_grad(NULL);
    adam_free(NULL);
}

static void test_momentum_and_frozen(void) {
    Tensor *p = tensor_create_matrix(1, 2);
    Tensor *frozen = tensor_create_ex(7.0f, 0);
    Tensor *params[] = {p, frozen};
    SGD *sgd = sgd_create_momentum(params, 2, 0.1f, 0.5f);
    assert(sgd && frozen->grad == NULL);
    p->data[0] = 1.0f;
    p->data[1] = -2.0f;
    p->grad[0] = 2.0f;
    p->grad[1] = -4.0f;
    sgd_step(sgd);
    assert(near(p->data[0], 0.8, 1e-6));
    assert(near(p->data[1], -1.6, 1e-6));
    assert(frozen->data[0] == 7.0f);
    p->grad[0] = -1.0f;
    p->grad[1] = 1.0f;
    sgd_step(sgd); /* velocity = (0,-1) */
    assert(near(p->data[0], 0.8, 1e-6));
    assert(near(p->data[1], -1.5, 1e-6));
    sgd_set_lr(sgd, 0.2f);
    sgd_zero_grad(sgd);
    assert(p->grad[0] == 0.0f && p->grad[1] == 0.0f);
    sgd_step(sgd); /* momentum continues even if gradient is zero */
    assert(near(p->data[1], -1.4, 1e-6));
    assert(tensor_set_requires_grad(frozen, 1));
    frozen->grad[0] = 2.0f;
    sgd_step(sgd);
    assert(near(frozen->data[0], 6.6, 1e-6));
    assert(tensor_set_requires_grad(frozen, 0));
    sgd_step(sgd);
    assert(near(frozen->data[0], 6.6, 1e-6));
    sgd_free(sgd);
    tensor_release(p);
    tensor_release(frozen);
}

static void test_adam_and_adamw(void) {
    Tensor *a = tensor_create(3.0f);
    Tensor *w = tensor_create(3.0f);
    Tensor *frozen = tensor_create_ex(-4.0f, 0);
    Tensor *ap[] = {a, frozen};
    Tensor *wp[] = {w, frozen};
    Adam *adam = adam_create(ap, 2, 0.01f);
    Adam *adamw = adamw_create(wp, 2, 0.01f, 0.2f);
    assert(adam && adamw);
    const float grads[] = {2.0f, -4.0f, 0.0f, 3.0f};
    double ref_a = 3.0, ref_w = 3.0, m = 0.0, v = 0.0;
    for (int t = 1; t <= 4; t++) {
        double g = grads[t - 1];
        a->grad[0] = w->grad[0] = grads[t - 1];
        m = 0.9 * m + 0.1 * g;
        v = 0.999 * v + 0.001 * g * g;
        double update = (m / (1.0 - pow(0.9, t))) /
                        (sqrt(v / (1.0 - pow(0.999, t))) + 1e-8);
        ref_a -= 0.01 * update;
        ref_w -= 0.01 * (update + 0.2 * ref_w);
        adam_step(adam);
        adam_step(adamw);
        assert(near(a->data[0], ref_a, 2e-6));
        assert(near(w->data[0], ref_w, 2e-6));
        assert(frozen->data[0] == -4.0f);
    }
    assert(tensor_set_requires_grad(frozen, 1));
    frozen->grad[0] = 2.0f;
    adam_step(adam); /* first update of newly unfrozen parameter */
    assert(near(frozen->data[0], -4.01, 1e-6));
    assert(tensor_set_requires_grad(frozen, 0));
    adam_set_lr(adamw, 0.0f);
    adam_zero_grad(adamw);
    assert(w->grad[0] == 0.0f && frozen->grad == NULL);
    float previous = w->data[0];
    adam_step(adamw);
    assert(w->data[0] == previous);
    adam_free(adam);
    adam_free(adamw);
    tensor_release(a);
    tensor_release(w);
    tensor_release(frozen);
}

static void test_clip(void) {
    Tensor *a = tensor_create_matrix(1, 2);
    Tensor *b = tensor_create(1.0f);
    Tensor *frozen = tensor_create_ex(1.0f, 0);
    Tensor *params[] = {a, b, frozen};
    a->grad[0] = 3.0f;
    a->grad[1] = 4.0f;
    b->grad[0] = 12.0f;
    assert(near(optim_clip_grad_norm(params, 3, 5.0f), 13.0, 1e-6));
    assert(near(a->grad[0], 15.0 / 13.0, 2e-7));
    assert(near(a->grad[1], 20.0 / 13.0, 2e-7));
    assert(near(b->grad[0], 60.0 / 13.0, 2e-7));
    float prior = a->grad[0];
    assert(near(optim_clip_grad_norm(params, 3, 20.0f), 5.0, 1e-6));
    assert(a->grad[0] == prior);
    assert(near(optim_clip_grad_norm(params, 3, 0.0f), 5.0, 1e-6));
    assert(a->grad[0] == 0.0f && a->grad[1] == 0.0f && b->grad[0] == 0.0f);
    a->grad[0] = FLT_MAX;
    a->grad[1] = FLT_MAX;
    assert(isinf(optim_clip_grad_norm(params, 3, 1.0f)));
    assert(near(a->grad[0], 1.0 / sqrt(2.0), 1e-6));
    a->grad[0] = 3.0f;
    b->grad[0] = NAN;
    assert(isnan(optim_clip_grad_norm(params, 3, 1.0f)));
    assert(a->grad[0] == 3.0f && isnan(b->grad[0]));
    b->grad[0] = INFINITY;
    assert(isnan(optim_clip_grad_norm(params, 3, 1.0f)));
    assert(a->grad[0] == 3.0f);
    tensor_release(a);
    tensor_release(b);
    tensor_release(frozen);
}

int main(void) {
    test_validation_and_borrowing();
    test_momentum_and_frozen();
    test_adam_and_adamw();
    test_clip();
    puts("optimizer tests passed");
    return 0;
}
