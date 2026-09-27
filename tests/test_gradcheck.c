#include <assert.h>
#include <math.h>
#include <stdio.h>

#include "gradcheck.h"
#include "loss.h"
#include "tensor_ops.h"

typedef Tensor *(*UnaryOp)(Tensor *);
typedef Tensor *(*BinaryOp)(Tensor *, Tensor *);

static Tensor *unary(Tensor **in, int n, void *ctx) {
    assert(n == 1);
    return (*(UnaryOp *)ctx)(in[0]);
}
static Tensor *binary(Tensor **in, int n, void *ctx) {
    assert(n == 2);
    return (*(BinaryOp *)ctx)(in[0], in[1]);
}
#define U(op) ((void *)&(UnaryOp){op})
#define B(op) ((void *)&(BinaryOp){op})

static Tensor *mse(Tensor **in, int n, void *ctx) {
    (void)ctx;
    assert(n == 2);
    return mse_loss(in[0], in[1]);
}
static Tensor *ce(Tensor **in, int n, void *ctx) {
    (void)ctx;
    assert(n == 2);
    return cross_entropy_loss(in[0], in[1]);
}
static Tensor *sum_axis(Tensor **in, int n, void *ctx) {
    assert(n == 1);
    return tensor_sum_axis(in[0], *(int *)ctx);
}
static Tensor *mean_axis(Tensor **in, int n, void *ctx) {
    assert(n == 1);
    return tensor_mean_axis(in[0], *(int *)ctx);
}
static Tensor *reshape(Tensor **in, int n, void *ctx) {
    assert(n == 1);
    int *dims = ctx;
    return tensor_reshape(in[0], dims[0], dims[1]);
}
static Tensor *shared(Tensor **in, int n, void *ctx) {
    (void)ctx;
    assert(n == 2);
    Tensor *s = tensor_mul(in[0], in[1]);
    if (!s) return NULL;
    Tensor *left = tensor_mul(s, s);
    Tensor *right = tensor_add(s, in[0]);
    Tensor *out = (left && right) ? tensor_add(left, right) : NULL;
    tensor_release(left);
    tensor_release(right);
    tensor_release(s);
    return out;
}

static Tensor *matrix(int rows, int cols, const float *data) {
    Tensor *t = tensor_create_matrix(rows, cols);
    assert(t);
    for (int i = 0; i < t->size; ++i) t->data[i] = data[i];
    return t;
}

/* Gradcheck must leave BOTH the data and nonzero preexisting gradients alone. */
static void check(GradcheckFunction fn, Tensor **in, int n, void *ctx,
                  const Tensor *seed) {
    float before_data[4][16], before_grad[4][16];
    assert(n > 0 && n <= 4);
    for (int i = 0; i < n; ++i) {
        assert(in[i] && in[i]->size <= 16);
        for (int j = 0; j < in[i]->size; ++j) {
            in[i]->grad[j] = (float)(17 + 3 * i + j) / 9.0f;
            before_data[i][j] = in[i]->data[j];
            before_grad[i][j] = in[i]->grad[j];
        }
    }
    int mode = grad_is_enabled();
    assert(tensor_gradcheck(fn, in, n, ctx, seed, 0.003f, 0.003f, 0.012f));
    assert(grad_is_enabled() == mode);
    for (int i = 0; i < n; ++i) for (int j = 0; j < in[i]->size; ++j) {
        assert(in[i]->data[j] == before_data[i][j]);
        assert(in[i]->grad[j] == before_grad[i][j]);
    }
}

static void old_ops(void) {
    Tensor *a = tensor_create(1.7f);
    Tensor *b = tensor_create(2.3f);
    Tensor *ab[] = {a, b};
    const BinaryOp old_binary[] = {tensor_add, tensor_sub, tensor_mul,
                                   tensor_div, tensor_pow};
    for (size_t i = 0; i < sizeof(old_binary)/sizeof(old_binary[0]); ++i)
        check(binary, ab, 2, (void *)&old_binary[i], NULL);
    const UnaryOp old_unary[] = {tensor_expn, tensor_relu, tensor_Tanh,
                                 tensor_sqrt, tensor_mean, tensor_softmax};
    /* softmax is matrix-only. */
    for (size_t i = 0; i + 1 < sizeof(old_unary)/sizeof(old_unary[0]); ++i)
        check(unary, ab, 1, (void *)&old_unary[i], NULL);
    tensor_release(a);
    tensor_release(b);

    const float xdata[] = {0.5f, 1.3f, 2.0f, 0.8f, 1.8f, 1.1f};
    Tensor *x = matrix(2, 3, xdata);
    Tensor *xs[] = {x};
    Tensor *seed = tensor_create_matrix_ex(2, 3, 0);
    assert(seed);
    const float weights[] = {-1.2f, 0.4f, 0.7f, 2.1f, -0.5f, 0.3f};
    for (int i = 0; i < 6; ++i) seed->data[i] = weights[i];
    check(unary, xs, 1, U(tensor_softmax), seed);
    check(unary, xs, 1, U(tensor_Tanh), seed);
    check(unary, xs, 1, U(tensor_mean), NULL);
    tensor_release(seed);
    tensor_release(x);
}

static void broadcast_matmul_and_dags(void) {
    const float xdata[] = {1.1f, 1.6f, 2.0f, 1.3f, 1.8f, 2.4f};
    const float rowdata[] = {0.5f, 1.1f, 1.7f};
    const float coldata[] = {0.8f, 1.2f};
    Tensor *x = matrix(2, 3, xdata);
    Tensor *row = matrix(1, 3, rowdata);
    Tensor *col = matrix(2, 1, coldata);
    Tensor *scalar = tensor_create(2.0f);
    Tensor *xr[] = {x, row};
    Tensor *xc[] = {x, col};
    Tensor *rc[] = {row, col};
    Tensor *xs[] = {x, scalar};
    check(binary, xr, 2, B(tensor_add), NULL);
    check(binary, xr, 2, B(tensor_mul), NULL);
    check(binary, xr, 2, B(tensor_div), NULL);
    check(binary, xc, 2, B(tensor_mul), NULL);
    check(binary, rc, 2, B(tensor_pow), NULL);
    check(binary, xs, 2, B(tensor_pow), NULL);
    check(binary, xs, 2, B(tensor_add), NULL);
    check(binary, xs, 2, B(tensor_div), NULL);
    check(shared, xr, 2, NULL, NULL);
    tensor_release(scalar);
    tensor_release(col);
    tensor_release(row);
    tensor_release(x);

    const float ad[] = {0.2f, 1.1f, -0.4f, 0.7f, 1.5f, -0.8f};
    const float bd[] = {0.4f, -0.6f, 1.1f, 0.3f, -0.2f, 0.8f};
    Tensor *a = matrix(2, 3, ad);
    Tensor *b = matrix(3, 2, bd);
    Tensor *inputs[] = {a, b};
    Tensor *seed = tensor_create_matrix_ex(2, 2, 0);
    assert(seed);
    seed->data[0] = 0.3f; seed->data[1] = -1.5f;
    seed->data[2] = 2.4f; seed->data[3] = 0.7f;
    check(binary, inputs, 2, B(tensor_matmul), seed);
    tensor_release(seed);
    tensor_release(a);
    tensor_release(b);
}

static void new_ops_and_losses(void) {
    const float xd[] = {0.6f, 1.1f, 1.7f, 2.0f, 0.8f, 1.3f};
    Tensor *x = matrix(2, 3, xd);
    Tensor *one[] = {x};
    const UnaryOp ops[] = {tensor_log, tensor_sigmoid, tensor_softplus,
                           tensor_transpose};
    for (size_t i = 0; i < sizeof(ops)/sizeof(ops[0]); ++i)
        check(unary, one, 1, (void *)&ops[i], NULL);
    int dims[] = {3, 2};
    check(reshape, one, 1, dims, NULL);
    for (int axis = -1; axis <= 1; ++axis) {
        check(sum_axis, one, 1, &axis, NULL);
        check(mean_axis, one, 1, &axis, NULL);
    }
    const float yd[] = {0.1f, 1.5f, 0.5f, 1.1f, 0.4f, 1.7f};
    Tensor *y = matrix(2, 3, yd);
    Tensor *both[] = {x, y};
    Tensor *loss_seed = tensor_create_ex(-0.75f, 0);
    assert(loss_seed);
    check(mse, both, 2, NULL, loss_seed);
    tensor_release(loss_seed);
    tensor_release(y);
    tensor_release(x);

    Tensor *scalar = tensor_create(1.2f);
    Tensor *single[] = {scalar};
    check(unary, single, 1, U(tensor_log), NULL);
    check(unary, single, 1, U(tensor_sigmoid), NULL);
    check(unary, single, 1, U(tensor_softplus), NULL);
    int scalar_dims[] = {1, 1};
    check(reshape, single, 1, scalar_dims, NULL);
    tensor_release(scalar);

    const float logits_data[] = {0.4f, -0.3f, 1.2f, -1.0f, 0.8f, 0.3f};
    /* Strictly positive weighted targets permit central differences on both inputs. */
    const float target_data[] = {0.2f, 0.6f, 0.4f, 0.7f, 0.1f, 0.3f};
    Tensor *logits = matrix(2, 3, logits_data);
    Tensor *targets = matrix(2, 3, target_data);
    Tensor *pair[] = {logits, targets};
    Tensor *ce_seed = tensor_create_ex(1.8f, 0);
    assert(ce_seed);
    check(ce, pair, 2, NULL, ce_seed);
    tensor_release(ce_seed);
    tensor_release(targets);
    tensor_release(logits);
}

typedef struct { int calls; int fail_at; } Failure;
static Tensor *failure_callback(Tensor **in, int n, void *ctx) {
    Failure *f = ctx;
    assert(n == 1);
    if (++f->calls == f->fail_at) return NULL;
    return tensor_mul(in[0], in[0]);
}
static Tensor *bad_derivative(Tensor **in, int n, void *ctx) {
    (void)n; (void)ctx;
    /* Deliberately disagree between tracked and finite-difference evaluations. */
    return grad_is_enabled() ? tensor_mul(in[0], in[0]) : tensor_add(in[0], in[0]);
}
static Tensor *wrong_shape(Tensor **in, int n, void *ctx) {
    (void)n;
    Failure *f = ctx;
    if (++f->calls > 1) return tensor_create_matrix_ex(1, 1, 0);
    return tensor_add(in[0], in[0]);
}
static void errors_and_restoration(void) {
    Tensor *x = tensor_create(1.6f);
    Tensor *one[] = {x};
    x->grad[0] = 9.25f;
    Failure f = {0, 2};
    int old_mode = grad_set_enabled(0);
    assert(!tensor_gradcheck(failure_callback, one, 1, &f, NULL, .003f, .001f, .01f));
    assert(f.calls == 2 && x->data[0] == 1.6f && x->grad[0] == 9.25f);
    assert(grad_is_enabled() == 0);
    check(unary, one, 1, U(tensor_expn), NULL);
    x->grad[0] = 9.25f;
    grad_set_enabled(1);
    assert(!tensor_gradcheck(bad_derivative, one, 1, NULL, NULL, .003f, .001f, .01f));
    assert(x->data[0] == 1.6f && x->grad[0] == 9.25f);
    assert(!tensor_gradcheck(NULL, one, 1, NULL, NULL, .003f, .001f, .01f));
    Tensor *duplicate[] = {x, x};
    assert(!tensor_gradcheck(binary, duplicate, 2, B(tensor_mul), NULL, .003f, .001f, .01f));
    Tensor *wrong_seed = tensor_create_matrix_ex(1, 1, 0);
    assert(wrong_seed);
    assert(!tensor_gradcheck(unary, one, 1, U(tensor_expn), wrong_seed, .003f, .001f, .01f));
    tensor_release(wrong_seed);
    Tensor *nan_seed = tensor_create_ex(NAN, 0);
    assert(nan_seed);
    assert(!tensor_gradcheck(unary, one, 1, U(tensor_expn), nan_seed, .003f, .001f, .01f));
    tensor_release(nan_seed);
    f.calls = 0;
    assert(!tensor_gradcheck(wrong_shape, one, 1, &f, NULL, .003f, .001f, .01f));
    Tensor alias_seed = *x;
    alias_seed.data = x->grad;
    check(unary, one, 1, U(tensor_expn), &alias_seed);
    x->grad[0] = 9.25f;
    assert(x->data[0] == 1.6f && x->grad[0] == 9.25f);
    assert(!tensor_gradcheck(unary, one, 1, U(tensor_expn), NULL, 0, .001f, .01f));
    assert(!tensor_gradcheck(unary, one, 1, U(tensor_expn), NULL, .003f, -1, .01f));
    Tensor *graph = tensor_mul(x, x);
    assert(graph);
    assert(!tensor_gradcheck(unary, one, 1, U(tensor_expn), NULL, .003f, .001f, .01f));
    tensor_release(graph);
    Tensor *frozen = tensor_create_ex(1.0f, 0);
    assert(frozen);
    Tensor *frozen_inputs[] = {frozen};
    assert(!tensor_gradcheck(unary, frozen_inputs, 1, U(tensor_expn), NULL, .003f, .001f, .01f));
    tensor_release(frozen);
    assert(grad_is_enabled() == 1 && x->data[0] == 1.6f && x->grad[0] == 9.25f);
    x->data[0] = 1e20f;
    assert(!tensor_gradcheck(unary, one, 1, U(tensor_Tanh), NULL, 1e-6f, .001f, .01f));
    assert(x->data[0] == 1e20f && x->grad[0] == 9.25f);
    x->data[0] = 0.001f;
    assert(!tensor_gradcheck(unary, one, 1, U(tensor_log), NULL, .003f, .001f, .01f));
    assert(x->data[0] == 0.001f && x->grad[0] == 9.25f);
    x->data[0] = 1.6f;
    grad_set_enabled(old_mode);
    tensor_release(x);
}

static void zero_base_integer_power(void) {
    Tensor *x = tensor_create(0.0f);
    Tensor *exponent = tensor_create(1.0f);
    Tensor *out = tensor_pow(x, exponent);
    assert(out && tensor_backward_with_grad(out, NULL));
    assert(isfinite(x->grad[0]) && fabsf(x->grad[0] - 1.0f) < 1e-6f);
    assert(exponent->grad[0] == 0.0f);
    tensor_release(out);
    tensor_release(exponent);
    tensor_release(x);
}

int main(void) {
    old_ops();
    broadcast_matmul_and_dags();
    new_ops_and_losses();
    errors_and_restoration();
    zero_base_integer_power();
    puts("gradcheck tests passed");
    return 0;
}
