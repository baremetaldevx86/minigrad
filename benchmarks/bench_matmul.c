/* Standalone C11 benchmark; for example:
 * cc -std=c11 -O2 -Wall -Wextra -Wpedantic -Iinclude \
 *    benchmarks/bench_matmul.c src/kernels.c -lm -o /tmp/bench_matmul
 * /tmp/bench_matmul
 * C11 timespec_get measures wall time, so results depend on host/load/compiler.
 */
#include "kernels.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef void (*Forward)(const float *, const float *, float *, int, int, int);
typedef void (*Backward)(const float *, const float *, const float *,
                         float *, float *, int, int, int);

static volatile float sink;

/* Intentionally simple reference loops, independent of the tiled kernels. */
static void reference_forward(const float *a, const float *b, float *out,
                              int m, int k, int n)
{
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int p = 0; p < k; ++p)
                sum += a[(size_t)i * k + p] * b[(size_t)p * n + j];
            out[(size_t)i * n + j] = sum;
        }
}

static void reference_backward(const float *a, const float *b, const float *u,
                               float *da, float *db, int m, int k, int n)
{
    if (da)
        for (int i = 0; i < m; ++i)
            for (int p = 0; p < k; ++p)
                for (int j = 0; j < n; ++j)
                    da[(size_t)i * k + p] +=
                        u[(size_t)i * n + j] * b[(size_t)p * n + j];
    if (db)
        for (int p = 0; p < k; ++p)
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < m; ++i)
                    db[(size_t)p * n + j] +=
                        a[(size_t)i * k + p] * u[(size_t)i * n + j];
}

static double seconds(void)
{
    struct timespec t;
    if (timespec_get(&t, TIME_UTC) != TIME_UTC) {
        fputs("clock unavailable\n", stderr);
        exit(EXIT_FAILURE);
    }
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static float *allocate(size_t count)
{
    float *p = malloc(count * sizeof(*p));
    if (!p) { fputs("out of memory\n", stderr); exit(EXIT_FAILURE); }
    return p;
}

static float sample(size_t i, unsigned salt)
{
    return (float)((i * 37u + salt * 13u) % 97u) / 97.0f - 0.5f;
}

static double checksum(const float *a, size_t count)
{
    double sum = 0.0;
    for (size_t x = 0; x < count; ++x) sum += a[x];
    return sum;
}

static double max_difference(const float *a, const float *b, size_t count,
                             int *valid)
{
    double worst = 0.0;
    for (size_t x = 0; x < count; ++x) {
        double diff = fabs((double)a[x] - b[x]);
        if (!isfinite(a[x]) || !isfinite(b[x]) ||
            diff > 2e-4 + 2e-4 * fabs((double)b[x])) *valid = 0;
        if (diff > worst) worst = diff;
    }
    return worst;
}

static double time_forward(Forward fn, const float *a, const float *b, float *out,
                           int m, int k, int n, int repetitions)
{
    double best = HUGE_VAL;
    for (int trial = 0; trial < 3; ++trial) {
        double start = seconds();
        for (int r = 0; r < repetitions; ++r) fn(a, b, out, m, k, n);
        double elapsed = seconds() - start;
        if (elapsed < best) best = elapsed;
        sink = out[(size_t)m * n - 1];
    }
    return best * 1000.0 / repetitions;
}

static double time_backward(Backward fn, const float *a, const float *b,
                            const float *u, float *da, float *db,
                            int m, int k, int n, int repetitions)
{
    double best = HUGE_VAL;
    for (int trial = 0; trial < 3; ++trial) {
        memset(da, 0, (size_t)m * k * sizeof(*da));
        memset(db, 0, (size_t)k * n * sizeof(*db));
        double start = seconds();
        for (int r = 0; r < repetitions; ++r)
            fn(a, b, u, da, db, m, k, n);
        double elapsed = seconds() - start;
        if (elapsed < best) best = elapsed;
        sink = da[(size_t)m * k - 1] + db[(size_t)k * n - 1];
    }
    return best * 1000.0 / repetitions;
}

static int bench(int m, int k, int n)
{
    size_t na = (size_t)m * k, nb = (size_t)k * n, no = (size_t)m * n;
    float *a = allocate(na), *b = allocate(nb), *u = allocate(no);
    float *out = allocate(no), *out_ref = allocate(no);
    float *da = allocate(na), *da_ref = allocate(na);
    float *db = allocate(nb), *db_ref = allocate(nb);
    for (size_t x = 0; x < na; ++x) a[x] = sample(x, 1);
    for (size_t x = 0; x < nb; ++x) b[x] = sample(x, 2);
    for (size_t x = 0; x < no; ++x) u[x] = sample(x, 3);

    reference_forward(a, b, out_ref, m, k, n);
    mg_matmul_forward(a, b, out, m, k, n);
    memset(da, 0, na * sizeof(*da));
    memset(db, 0, nb * sizeof(*db));
    memset(da_ref, 0, na * sizeof(*da_ref));
    memset(db_ref, 0, nb * sizeof(*db_ref));
    reference_backward(a, b, u, da_ref, db_ref, m, k, n);
    mg_matmul_backward(a, b, u, da, db, m, k, n);
    int valid = 1;
    double out_diff = max_difference(out, out_ref, no, &valid);
    double da_diff = max_difference(da, da_ref, na, &valid);
    double db_diff = max_difference(db, db_ref, nb, &valid);
    printf("%dx%d @ %dx%d: %s, max_abs(out/da/db)=%.3g/%.3g/%.3g\n",
           m, k, k, n, valid ? "PASS" : "FAIL", out_diff, da_diff, db_diff);
    printf("  checksum optimized(out/da/db)=%.9g/%.9g/%.9g",
           checksum(out, no), checksum(da, na), checksum(db, nb));
    printf(" reference=%.9g/%.9g/%.9g\n",
           checksum(out_ref, no), checksum(da_ref, na), checksum(db_ref, nb));

    /* Same repetitions for both implementations. Warm them up first. Each
     * backward invocation accumulates into its output (reset between trials,
     * outside the timed region); forward overwrites its output each time. */
    reference_forward(a, b, out_ref, m, k, n);
    mg_matmul_forward(a, b, out, m, k, n);
    mg_matmul_backward(a, b, u, da, db, m, k, n);
    reference_backward(a, b, u, da_ref, db_ref, m, k, n);
    int repetitions = 1;
    double start = seconds();
    mg_matmul_forward(a, b, out, m, k, n);
    mg_matmul_backward(a, b, u, da, db, m, k, n);
    double sample_time = seconds() - start;
    if (sample_time > 0.0 && sample_time < 0.1) {
        double target = 0.1 / sample_time;
        repetitions = target < 128.0 ? (int)target + 1 : 128;
    }
    printf("  best of 3 (C11 wall clock, %d repetitions/trial):\n", repetitions);
    printf("    forward  tiled %.4f ms/call, reference %.4f ms/call\n",
           time_forward(mg_matmul_forward, a, b, out, m, k, n, repetitions),
           time_forward(reference_forward, a, b, out_ref, m, k, n, repetitions));
    printf("    backward tiled %.4f ms/call, reference %.4f ms/call\n",
           time_backward(mg_matmul_backward, a, b, u, da, db, m, k, n, repetitions),
           time_backward(reference_backward, a, b, u, da_ref, db_ref,
                         m, k, n, repetitions));
    free(a); free(b); free(u); free(out); free(out_ref);
    free(da); free(da_ref); free(db); free(db_ref);
    return valid;
}

int main(void)
{
    int ok = 1;
    ok &= bench(32, 784, 256);
    ok &= bench(32, 256, 128);
    ok &= bench(32, 128, 10);
    ok &= bench(17, 31, 19);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
