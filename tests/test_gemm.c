#include "cob_gemm.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t rng_next(uint32_t* state)
{
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

static float rng_f32(uint32_t* state)
{
    const uint32_t bits = rng_next(state);
    const int value = (int)((bits >> 8) % 2001u) - 1000;
    return (float)value / 997.0f;
}

static void fill_random(float* x, int count, uint32_t* state)
{
    for (int i = 0; i < count; ++i) {
        x[i] = rng_f32(state);
    }
}

static float max_abs_diff(const float* a, const float* b, int count)
{
    float max_diff = 0.0f;
    for (int i = 0; i < count; ++i) {
        const float diff = fabsf(a[i] - b[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    return max_diff;
}

static int test_shape(int m, int n, int k)
{
    const int lda = k + 3;
    const int ldb = n + 5;
    const int ldc = n + 7;
    const int a_count = m * lda;
    const int b_count = k * ldb;
    const int c_count = m * ldc;

    float* a = (float*)malloc((size_t)a_count * sizeof(float));
    float* b = (float*)malloc((size_t)b_count * sizeof(float));
    float* cref = (float*)malloc((size_t)c_count * sizeof(float));
    float* cgot = (float*)malloc((size_t)c_count * sizeof(float));
    float* cpacked = (float*)malloc((size_t)c_count * sizeof(float));

    if (a == NULL || b == NULL || cref == NULL || cgot == NULL || cpacked == NULL) {
        fprintf(stderr, "allocation failed for %dx%dx%d\n", m, n, k);
        free(a);
        free(b);
        free(cref);
        free(cgot);
        free(cpacked);
        return 1;
    }

    uint32_t state = 0x12345678u ^ (uint32_t)(m * 131 + n * 17 + k);
    fill_random(a, a_count, &state);
    fill_random(b, b_count, &state);
    memset(cref, 0, (size_t)c_count * sizeof(float));
    memset(cgot, 0, (size_t)c_count * sizeof(float));
    memset(cpacked, 0, (size_t)c_count * sizeof(float));

    cob_sgemm_ref_rowmajor(m, n, k, a, lda, b, ldb, cref, ldc);
    cob_sgemm_rowmajor(m, n, k, a, lda, b, ldb, cgot, ldc);

    cob_packed_b_f32 packed_b;
    if (cob_sgemm_pack_b(&packed_b, k, n, b, ldb) != 0) {
        fprintf(stderr, "pack failed for %dx%dx%d\n", m, n, k);
        free(a);
        free(b);
        free(cref);
        free(cgot);
        free(cpacked);
        return 1;
    }
    cob_sgemm_rowmajor_packed_b(m, n, k, a, lda, &packed_b, cpacked, ldc);
    cob_sgemm_free_packed_b(&packed_b);

    const float direct_diff = max_abs_diff(cref, cgot, c_count);
    const float packed_diff = max_abs_diff(cref, cpacked, c_count);
    const float tol = 2.0e-3f;
    const int failed = direct_diff > tol || packed_diff > tol;

    if (failed) {
        fprintf(stderr,
            "shape %dx%dx%d failed: direct_diff=%g packed_diff=%g\n",
            m,
            n,
            k,
            (double)direct_diff,
            (double)packed_diff);
    }

    free(a);
    free(b);
    free(cref);
    free(cgot);
    free(cpacked);
    return failed;
}

int main(void)
{
    static const int shapes[][3] = {
        {1, 1, 1},
        {2, 3, 5},
        {7, 8, 9},
        {8, 8, 8},
        {8, 8, 31},
        {9, 8, 17},
        {8, 9, 17},
        {15, 15, 15},
        {16, 16, 16},
        {17, 19, 23},
        {31, 33, 37},
        {64, 64, 64},
        {96, 64, 128},
        {64, 96, 128},
        {127, 65, 257},
    };

    int failures = 0;
    const int shape_count = (int)(sizeof(shapes) / sizeof(shapes[0]));
    for (int i = 0; i < shape_count; ++i) {
        failures += test_shape(shapes[i][0], shapes[i][1], shapes[i][2]);
    }

    if (failures != 0) {
        fprintf(stderr, "%d GEMM test shape(s) failed\n", failures);
        return 1;
    }

    printf("all GEMM tests passed (%d shapes)\n", shape_count);
    return 0;
}

