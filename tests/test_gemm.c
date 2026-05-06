#if !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200112L
#endif

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

static float* alloc_f32(size_t count, int aligned)
{
    void* ptr = NULL;
    if (!aligned) {
        return (float*)malloc(count * sizeof(float));
    }
    if (posix_memalign(&ptr, 128, count * sizeof(float)) != 0) {
        return NULL;
    }
    return (float*)ptr;
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

static int test_shape_ex(int m, int n, int k, int lda, int ldb, int ldc, int aligned_c)
{
    const int a_count = m * lda;
    const int b_count = k * ldb;
    const int c_count = m * ldc;

    float* a = (float*)malloc((size_t)a_count * sizeof(float));
    float* b = (float*)malloc((size_t)b_count * sizeof(float));
    float* cref = alloc_f32((size_t)c_count, aligned_c);
    float* cgot = alloc_f32((size_t)c_count, aligned_c);
    float* cpacked = alloc_f32((size_t)c_count, aligned_c);

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

static int test_shape(int m, int n, int k)
{
    return test_shape_ex(m, n, k, k + 3, n + 5, n + 7, 0);
}

static int test_aligned_shape(int m, int n, int k)
{
    return test_shape_ex(m, n, k, k, n, n, 1);
}

static int test_packed_matches_direct_aligned_shape(int m, int n, int k)
{
    const int count = m * n;
    float* a = alloc_f32((size_t)m * (size_t)k, 1);
    float* b = alloc_f32((size_t)k * (size_t)n, 1);
    float* cdirect = alloc_f32((size_t)count, 1);
    float* cpacked = alloc_f32((size_t)count, 1);

    if (a == NULL || b == NULL || cdirect == NULL || cpacked == NULL) {
        fprintf(stderr, "allocation failed for packed/direct %dx%dx%d\n", m, n, k);
        free(a);
        free(b);
        free(cdirect);
        free(cpacked);
        return 1;
    }

    uint32_t state = 0xfeed5eedu ^ (uint32_t)(m * 131 + n * 17 + k);
    fill_random(a, m * k, &state);
    fill_random(b, k * n, &state);
    memset(cdirect, 0, (size_t)count * sizeof(float));
    memset(cpacked, 0, (size_t)count * sizeof(float));

    cob_sgemm_rowmajor(m, n, k, a, k, b, n, cdirect, n);

    cob_packed_b_f32 packed_b;
    if (cob_sgemm_pack_b(&packed_b, k, n, b, n) != 0) {
        fprintf(stderr, "pack failed for packed/direct %dx%dx%d\n", m, n, k);
        free(a);
        free(b);
        free(cdirect);
        free(cpacked);
        return 1;
    }
    cob_sgemm_rowmajor_packed_b(m, n, k, a, k, &packed_b, cpacked, n);
    cob_sgemm_free_packed_b(&packed_b);

    const float diff = max_abs_diff(cdirect, cpacked, count);
    const int failed = diff > 2.0e-3f;
    if (failed) {
        fprintf(stderr, "packed/direct %dx%dx%d failed: diff=%g\n", m, n, k, (double)diff);
    }

    free(a);
    free(b);
    free(cdirect);
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
        {32, 32, 1},
        {32, 32, 32},
        {32, 64, 31},
        {64, 32, 33},
        {65, 65, 67},
        {31, 33, 37},
        {64, 64, 64},
        {96, 64, 128},
        {64, 96, 128},
        {127, 65, 257},
    };

    int failures = 0;
    int total_shapes = 0;
    const int shape_count = (int)(sizeof(shapes) / sizeof(shapes[0]));
    for (int i = 0; i < shape_count; ++i) {
        failures += test_shape(shapes[i][0], shapes[i][1], shapes[i][2]);
    }
    total_shapes += shape_count;

    failures += test_aligned_shape(32, 32, 32);
    failures += test_aligned_shape(64, 64, 64);
    failures += test_aligned_shape(128, 128, 129);
    failures += test_packed_matches_direct_aligned_shape(512, 512, 512);
    total_shapes += 4;
#if defined(__APPLE__) && defined(__aarch64__)
    failures += test_packed_matches_direct_aligned_shape(832, 832, 832);
    failures += test_packed_matches_direct_aligned_shape(960, 960, 960);
    failures += test_packed_matches_direct_aligned_shape(1088, 1088, 1088);
    failures += test_packed_matches_direct_aligned_shape(1152, 1152, 1152);
    failures += test_packed_matches_direct_aligned_shape(1216, 1216, 1216);
    failures += test_packed_matches_direct_aligned_shape(832, 960, 896);
    failures += test_packed_matches_direct_aligned_shape(832, 1280, 832);
    failures += test_packed_matches_direct_aligned_shape(960, 1280, 960);
    failures += test_packed_matches_direct_aligned_shape(1280, 1280, 832);
    failures += test_packed_matches_direct_aligned_shape(832, 1344, 832);
    failures += test_packed_matches_direct_aligned_shape(1344, 1344, 960);
    failures += test_packed_matches_direct_aligned_shape(832, 1408, 832);
    failures += test_packed_matches_direct_aligned_shape(832, 1408, 1152);
    failures += test_packed_matches_direct_aligned_shape(1408, 1408, 960);
    failures += test_packed_matches_direct_aligned_shape(832, 1472, 1152);
    failures += test_packed_matches_direct_aligned_shape(960, 1472, 1152);
    failures += test_packed_matches_direct_aligned_shape(64, 2560, 1536);
    failures += test_packed_matches_direct_aligned_shape(64, 2048, 2048);
    failures += test_packed_matches_direct_aligned_shape(64, 1088, 2048);
    failures += test_packed_matches_direct_aligned_shape(64, 1152, 2048);
    failures += test_packed_matches_direct_aligned_shape(64, 1280, 2048);
    failures += test_packed_matches_direct_aligned_shape(64, 1408, 1536);
    failures += test_packed_matches_direct_aligned_shape(64, 1472, 1536);
    failures += test_packed_matches_direct_aligned_shape(64, 1600, 2048);
    failures += test_packed_matches_direct_aligned_shape(512, 1152, 2048);
    failures += test_packed_matches_direct_aligned_shape(512, 1024, 4096);
    failures += test_packed_matches_direct_aligned_shape(512, 1024, 2048);
    failures += test_packed_matches_direct_aligned_shape(768, 1024, 2048);
    failures += test_packed_matches_direct_aligned_shape(1024, 1024, 2048);
    failures += test_packed_matches_direct_aligned_shape(1024, 1152, 4096);
    failures += test_packed_matches_direct_aligned_shape(1024, 1152, 2048);
    failures += test_packed_matches_direct_aligned_shape(512, 1152, 3072);
    failures += test_packed_matches_direct_aligned_shape(512, 1152, 1536);
    failures += test_packed_matches_direct_aligned_shape(768, 1152, 1536);
    failures += test_packed_matches_direct_aligned_shape(1024, 1152, 1536);
    failures += test_packed_matches_direct_aligned_shape(512, 960, 4096);
    failures += test_packed_matches_direct_aligned_shape(512, 512, 2048);
    failures += test_packed_matches_direct_aligned_shape(768, 512, 3072);
    failures += test_packed_matches_direct_aligned_shape(1024, 512, 3072);
    failures += test_packed_matches_direct_aligned_shape(1280, 512, 3072);
    failures += test_packed_matches_direct_aligned_shape(1024, 512, 4096);
    failures += test_packed_matches_direct_aligned_shape(1280, 512, 4096);
    failures += test_packed_matches_direct_aligned_shape(384, 2048, 2048);
    failures += test_packed_matches_direct_aligned_shape(384, 4096, 1024);
    failures += test_packed_matches_direct_aligned_shape(384, 1152, 3072);
    failures += test_packed_matches_direct_aligned_shape(384, 1216, 4096);
    failures += test_packed_matches_direct_aligned_shape(384, 1216, 3072);
    failures += test_packed_matches_direct_aligned_shape(768, 1216, 2048);
    failures += test_packed_matches_direct_aligned_shape(1024, 1216, 2048);
    failures += test_packed_matches_direct_aligned_shape(1024, 1216, 3072);
    failures += test_packed_matches_direct_aligned_shape(512, 1024, 3072);
    failures += test_packed_matches_direct_aligned_shape(1024, 1024, 3072);
    failures += test_packed_matches_direct_aligned_shape(512, 768, 2048);
    failures += test_packed_matches_direct_aligned_shape(768, 768, 2048);
    failures += test_packed_matches_direct_aligned_shape(1024, 768, 2048);
    failures += test_packed_matches_direct_aligned_shape(512, 768, 3072);
    failures += test_packed_matches_direct_aligned_shape(1024, 768, 3072);
    failures += test_packed_matches_direct_aligned_shape(512, 768, 4096);
    failures += test_packed_matches_direct_aligned_shape(768, 768, 4096);
    failures += test_packed_matches_direct_aligned_shape(1024, 1024, 4096);
    failures += test_packed_matches_direct_aligned_shape(64, 1024, 7168);
    failures += test_packed_matches_direct_aligned_shape(64, 1088, 7168);
    failures += test_packed_matches_direct_aligned_shape(64, 2112, 512);
    failures += test_packed_matches_direct_aligned_shape(64, 4096, 512);
    failures += test_packed_matches_direct_aligned_shape(64, 8192, 512);
    failures += test_packed_matches_direct_aligned_shape(64, 2112, 7168);
    failures += test_packed_matches_direct_aligned_shape(64, 4096, 7168);
    failures += test_packed_matches_direct_aligned_shape(64, 32768, 512);
    failures += test_packed_matches_direct_aligned_shape(96, 4096, 512);
    failures += test_packed_matches_direct_aligned_shape(96, 8192, 512);
    failures += test_packed_matches_direct_aligned_shape(128, 4096, 512);
    failures += test_packed_matches_direct_aligned_shape(128, 8192, 512);
    failures += test_packed_matches_direct_aligned_shape(96, 4096, 1024);
    failures += test_packed_matches_direct_aligned_shape(96, 8192, 1024);
    failures += test_packed_matches_direct_aligned_shape(96, 4096, 2048);
    failures += test_packed_matches_direct_aligned_shape(128, 4096, 1024);
    failures += test_packed_matches_direct_aligned_shape(128, 8192, 1024);
    failures += test_packed_matches_direct_aligned_shape(128, 8192, 4096);
    failures += test_packed_matches_direct_aligned_shape(128, 2048, 2048);
    failures += test_packed_matches_direct_aligned_shape(1280, 1280, 1280);
    failures += test_packed_matches_direct_aligned_shape(2048, 2048, 2048);
    total_shapes += 81;
#endif

    if (failures != 0) {
        fprintf(stderr, "%d GEMM test shape(s) failed\n", failures);
        return 1;
    }

    printf("all GEMM tests passed (%d shapes)\n", total_shapes);
    return 0;
}
