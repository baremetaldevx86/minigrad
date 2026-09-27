#include "kernels.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

static float sample(size_t i, unsigned salt)
{
    return (float)((i * 37u + salt * 13u) % 97u) / 97.0f - 0.5f;
}

static void near(float actual, double expected)
{
    assert(isfinite(actual));
    assert(fabs((double)actual - expected) <= 2e-4 + 2e-4 * fabs(expected));
}

static void check_shape(int m, int k, int n)
{
    size_t na = (size_t)m * k, nb = (size_t)k * n, no = (size_t)m * n;
    float *a = malloc(na * sizeof(*a)), *b = malloc(nb * sizeof(*b));
    float *u = malloc(no * sizeof(*u)), *out = malloc(no * sizeof(*out));
    float *da = malloc(na * sizeof(*da)), *db = malloc(nb * sizeof(*db));
    assert(a && b && u && out && da && db);
    for (size_t x = 0; x < na; ++x) {
        a[x] = sample(x, 1);
        da[x] = sample(x, 4); /* Verify accumulation, not replacement. */
    }
    for (size_t x = 0; x < nb; ++x) {
        b[x] = sample(x, 2);
        db[x] = sample(x, 5);
    }
    for (size_t x = 0; x < no; ++x) {
        u[x] = sample(x, 3);
        out[x] = NAN; /* Forward must initialize every element. */
    }

    mg_matmul_forward(a, b, out, m, k, n);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int p = 0; p < k; ++p)
                sum += (double)a[(size_t)i * k + p] * b[(size_t)p * n + j];
            near(out[(size_t)i * n + j], sum);
        }

    mg_matmul_backward(a, b, u, da, db, m, k, n);
    for (int i = 0; i < m; ++i)
        for (int p = 0; p < k; ++p) {
            size_t x = (size_t)i * k + p;
            double sum = sample(x, 4);
            for (int j = 0; j < n; ++j)
                sum += (double)u[(size_t)i * n + j] * b[(size_t)p * n + j];
            near(da[x], sum);
        }
    for (int p = 0; p < k; ++p)
        for (int j = 0; j < n; ++j) {
            size_t x = (size_t)p * n + j;
            double sum = sample(x, 5);
            for (int i = 0; i < m; ++i)
                sum += (double)a[(size_t)i * k + p] * u[(size_t)i * n + j];
            near(db[x], sum);
        }

    /* Either output can be skipped, including when the unused input is NULL. */
    for (size_t x = 0; x < na; ++x) da[x] = 0.0f;
    for (size_t x = 0; x < nb; ++x) db[x] = 0.0f;
    mg_matmul_backward(NULL, b, u, da, NULL, m, k, n);
    mg_matmul_backward(a, NULL, u, NULL, db, m, k, n);
    for (int i = 0; i < m; ++i)
        for (int p = 0; p < k; ++p) {
            double sum = 0.0;
            for (int j = 0; j < n; ++j)
                sum += (double)u[(size_t)i * n + j] * b[(size_t)p * n + j];
            near(da[(size_t)i * k + p], sum);
        }
    for (int p = 0; p < k; ++p)
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int i = 0; i < m; ++i)
                sum += (double)a[(size_t)i * k + p] * u[(size_t)i * n + j];
            near(db[(size_t)p * n + j], sum);
        }
    mg_matmul_backward(NULL, NULL, u, NULL, NULL, m, k, n);

    free(a); free(b); free(u); free(out); free(da); free(db);
}

int main(void)
{
    /* Non-square, singletons, partial tiles, and the largest MNIST layer. */
    check_shape(2, 3, 4);
    check_shape(1, 1, 1);
    check_shape(3, 7, 1);
    check_shape(1, 5, 9);
    check_shape(17, 31, 19);
    check_shape(32, 784, 256);
    puts("matmul kernel tests passed");
    return 0;
}
