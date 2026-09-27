#include "kernels.h"

#include <stddef.h>

/* A is rows x inner, B is inner x cols, output is rows x cols.
 * Tile independent output axes; inner loops traverse contiguous B/output data.
 * No packing, allocation, intrinsics, or padded dimensions are required. */
enum { ROW_BLOCK = 16, COL_BLOCK = 64, REDUCE_BLOCK = 32 };

static int block_end(int start, int block, int extent)
{
    return extent - start < block ? extent : start + block;
}

void mg_matmul_forward(const float *a, const float *b, float *out,
                       int rows, int inner, int cols)
{
    for (int ii = 0; ii < rows; ) {
        int iend = block_end(ii, ROW_BLOCK, rows);
        for (int jj = 0; jj < cols; ) {
            int jend = block_end(jj, COL_BLOCK, cols);
            for (int i = ii; i < iend; ++i) {
                for (int j = jj; j < jend; ++j)
                    out[(size_t)i * cols + j] = 0.0f;
                for (int p = 0; p < inner; ++p) {
                    float x = a[(size_t)i * inner + p];
                    for (int j = jj; j < jend; ++j)
                        out[(size_t)i * cols + j] += x * b[(size_t)p * cols + j];
                }
            }
            jj = jend;
        }
        ii = iend;
    }
}

void mg_matmul_backward(const float *a, const float *b, const float *upstream,
                        float *da, float *db, int rows, int inner, int cols)
{
    /* dA = upstream @ B^T: contiguous dot products. */
    if (da) {
        for (int i = 0; i < rows; ++i) {
            for (int p = 0; p < inner; ++p) {
                float sum = 0.0f;
                for (int j = 0; j < cols; ++j)
                    sum += upstream[(size_t)i * cols + j] * b[(size_t)p * cols + j];
                da[(size_t)i * inner + p] += sum;
            }
        }
    }

    /* dB = A^T @ upstream: a small dB panel stays live in cache. */
    if (db) {
        for (int pp = 0; pp < inner; ) {
            int pend = block_end(pp, REDUCE_BLOCK, inner);
            for (int jj = 0; jj < cols; ) {
                int jend = block_end(jj, COL_BLOCK, cols);
                for (int i = 0; i < rows; ++i) {
                    for (int p = pp; p < pend; ++p) {
                        float x = a[(size_t)i * inner + p];
                        for (int j = jj; j < jend; ++j)
                            db[(size_t)p * cols + j] += x * upstream[(size_t)i * cols + j];
                    }
                }
                jj = jend;
            }
            pp = pend;
        }
    }
}
