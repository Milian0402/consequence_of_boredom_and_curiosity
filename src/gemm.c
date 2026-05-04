#include "cob_gemm.h"

#include <stdlib.h>
#include <string.h>

#if defined(COB_USE_NEON)
#include <arm_neon.h>
#endif

static int cob_min_i32(int a, int b)
{
    return a < b ? a : b;
}

static void* cob_aligned_alloc(size_t alignment, size_t size)
{
    void* ptr = NULL;
    if (size == 0) {
        return NULL;
    }
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return NULL;
    }
    return ptr;
}

void cob_sgemm_ref_rowmajor(
    int m,
    int n,
    int k,
    const float* a,
    int lda,
    const float* b,
    int ldb,
    float* c,
    int ldc)
{
    if (m <= 0 || n <= 0 || k <= 0 || a == NULL || b == NULL || c == NULL) {
        return;
    }

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int p = 0; p < k; ++p) {
                sum += a[i * lda + p] * b[p * ldb + j];
            }
            c[i * ldc + j] = sum;
        }
    }
}

int cob_sgemm_pack_b(
    cob_packed_b_f32* packed,
    int k,
    int n,
    const float* b,
    int ldb)
{
    if (packed == NULL) {
        return -1;
    }

    packed->k = 0;
    packed->n = 0;
    packed->nr = COB_SGEMM_NR;
    packed->bytes = 0;
    packed->data = NULL;

    if (k <= 0 || n <= 0 || b == NULL || ldb < n) {
        return -1;
    }

    const int panels = (n + COB_SGEMM_NR - 1) / COB_SGEMM_NR;
    const size_t count = (size_t)panels * (size_t)k * (size_t)COB_SGEMM_NR;
    const size_t bytes = count * sizeof(float);
    float* data = (float*)cob_aligned_alloc(64, bytes);
    if (data == NULL) {
        return -1;
    }

    for (int panel = 0; panel < panels; ++panel) {
        const int col0 = panel * COB_SGEMM_NR;
        float* dst_panel = data + (size_t)panel * (size_t)k * (size_t)COB_SGEMM_NR;

        for (int p = 0; p < k; ++p) {
            float* dst = dst_panel + (size_t)p * (size_t)COB_SGEMM_NR;
            for (int j = 0; j < COB_SGEMM_NR; ++j) {
                const int col = col0 + j;
                dst[j] = col < n ? b[p * ldb + col] : 0.0f;
            }
        }
    }

    packed->k = k;
    packed->n = n;
    packed->nr = COB_SGEMM_NR;
    packed->bytes = bytes;
    packed->data = data;
    return 0;
}

void cob_sgemm_free_packed_b(cob_packed_b_f32* packed)
{
    if (packed == NULL) {
        return;
    }

    free(packed->data);
    packed->k = 0;
    packed->n = 0;
    packed->nr = COB_SGEMM_NR;
    packed->bytes = 0;
    packed->data = NULL;
}

static void cob_sgemm_tile_scalar_packed_b(
    int mr,
    int nr,
    int k,
    const float* a,
    int lda,
    const float* packed_b,
    float* c,
    int ldc)
{
    for (int i = 0; i < mr; ++i) {
        for (int j = 0; j < nr; ++j) {
            float sum = 0.0f;
            for (int p = 0; p < k; ++p) {
                sum += a[i * lda + p] * packed_b[(size_t)p * (size_t)COB_SGEMM_NR + j];
            }
            c[i * ldc + j] = sum;
        }
    }
}

#if defined(COB_USE_NEON)
static void cob_sgemm_8x8_neon_packed_b(
    int k,
    const float* a,
    int lda,
    const float* packed_b,
    float* c,
    int ldc)
{
    float32x4_t c00 = vdupq_n_f32(0.0f);
    float32x4_t c01 = vdupq_n_f32(0.0f);
    float32x4_t c10 = vdupq_n_f32(0.0f);
    float32x4_t c11 = vdupq_n_f32(0.0f);
    float32x4_t c20 = vdupq_n_f32(0.0f);
    float32x4_t c21 = vdupq_n_f32(0.0f);
    float32x4_t c30 = vdupq_n_f32(0.0f);
    float32x4_t c31 = vdupq_n_f32(0.0f);
    float32x4_t c40 = vdupq_n_f32(0.0f);
    float32x4_t c41 = vdupq_n_f32(0.0f);
    float32x4_t c50 = vdupq_n_f32(0.0f);
    float32x4_t c51 = vdupq_n_f32(0.0f);
    float32x4_t c60 = vdupq_n_f32(0.0f);
    float32x4_t c61 = vdupq_n_f32(0.0f);
    float32x4_t c70 = vdupq_n_f32(0.0f);
    float32x4_t c71 = vdupq_n_f32(0.0f);

    for (int p = 0; p < k; ++p) {
        const float* bp = packed_b + (size_t)p * (size_t)COB_SGEMM_NR;
        const float32x4_t b0 = vld1q_f32(bp);
        const float32x4_t b1 = vld1q_f32(bp + 4);

        const float a0 = a[0 * lda + p];
        const float a1 = a[1 * lda + p];
        const float a2 = a[2 * lda + p];
        const float a3 = a[3 * lda + p];
        const float a4 = a[4 * lda + p];
        const float a5 = a[5 * lda + p];
        const float a6 = a[6 * lda + p];
        const float a7 = a[7 * lda + p];

        c00 = vfmaq_n_f32(c00, b0, a0);
        c01 = vfmaq_n_f32(c01, b1, a0);
        c10 = vfmaq_n_f32(c10, b0, a1);
        c11 = vfmaq_n_f32(c11, b1, a1);
        c20 = vfmaq_n_f32(c20, b0, a2);
        c21 = vfmaq_n_f32(c21, b1, a2);
        c30 = vfmaq_n_f32(c30, b0, a3);
        c31 = vfmaq_n_f32(c31, b1, a3);
        c40 = vfmaq_n_f32(c40, b0, a4);
        c41 = vfmaq_n_f32(c41, b1, a4);
        c50 = vfmaq_n_f32(c50, b0, a5);
        c51 = vfmaq_n_f32(c51, b1, a5);
        c60 = vfmaq_n_f32(c60, b0, a6);
        c61 = vfmaq_n_f32(c61, b1, a6);
        c70 = vfmaq_n_f32(c70, b0, a7);
        c71 = vfmaq_n_f32(c71, b1, a7);
    }

    vst1q_f32(c + 0 * ldc, c00);
    vst1q_f32(c + 0 * ldc + 4, c01);
    vst1q_f32(c + 1 * ldc, c10);
    vst1q_f32(c + 1 * ldc + 4, c11);
    vst1q_f32(c + 2 * ldc, c20);
    vst1q_f32(c + 2 * ldc + 4, c21);
    vst1q_f32(c + 3 * ldc, c30);
    vst1q_f32(c + 3 * ldc + 4, c31);
    vst1q_f32(c + 4 * ldc, c40);
    vst1q_f32(c + 4 * ldc + 4, c41);
    vst1q_f32(c + 5 * ldc, c50);
    vst1q_f32(c + 5 * ldc + 4, c51);
    vst1q_f32(c + 6 * ldc, c60);
    vst1q_f32(c + 6 * ldc + 4, c61);
    vst1q_f32(c + 7 * ldc, c70);
    vst1q_f32(c + 7 * ldc + 4, c71);
}
#endif

void cob_sgemm_rowmajor_packed_b(
    int m,
    int n,
    int k,
    const float* a,
    int lda,
    const cob_packed_b_f32* packed_b,
    float* c,
    int ldc)
{
    if (m <= 0 || n <= 0 || k <= 0 || a == NULL || packed_b == NULL || c == NULL ||
        packed_b->data == NULL || packed_b->k != k || packed_b->n < n ||
        lda < k || ldc < n) {
        return;
    }

    for (int jc = 0; jc < n; jc += COB_SGEMM_NR) {
        const int nr = cob_min_i32(COB_SGEMM_NR, n - jc);
        const int panel = jc / COB_SGEMM_NR;
        const float* bp = packed_b->data +
            (size_t)panel * (size_t)k * (size_t)COB_SGEMM_NR;

        for (int ic = 0; ic < m; ic += COB_SGEMM_MR) {
            const int mr = cob_min_i32(COB_SGEMM_MR, m - ic);
            const float* at = a + (size_t)ic * (size_t)lda;
            float* ct = c + (size_t)ic * (size_t)ldc + jc;

#if defined(COB_USE_NEON)
            if (mr == COB_SGEMM_MR && nr == COB_SGEMM_NR) {
                cob_sgemm_8x8_neon_packed_b(k, at, lda, bp, ct, ldc);
            } else
#endif
            {
                cob_sgemm_tile_scalar_packed_b(mr, nr, k, at, lda, bp, ct, ldc);
            }
        }
    }
}

void cob_sgemm_rowmajor(
    int m,
    int n,
    int k,
    const float* a,
    int lda,
    const float* b,
    int ldb,
    float* c,
    int ldc)
{
    if (m <= 0 || n <= 0 || k <= 0 || a == NULL || b == NULL || c == NULL ||
        lda < k || ldb < n || ldc < n) {
        return;
    }

    cob_packed_b_f32 packed_b;
    if (cob_sgemm_pack_b(&packed_b, k, n, b, ldb) != 0) {
        cob_sgemm_ref_rowmajor(m, n, k, a, lda, b, ldb, c, ldc);
        return;
    }

    cob_sgemm_rowmajor_packed_b(m, n, k, a, lda, &packed_b, c, ldc);
    cob_sgemm_free_packed_b(&packed_b);
}

