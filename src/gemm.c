#if !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200112L
#endif

#include "cob_gemm.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(COB_USE_NEON)
#include <arm_neon.h>
#endif

#if defined(__APPLE__) && defined(__aarch64__) && !defined(COB_DISABLE_APPLE_AMX)
#define COB_USE_APPLE_AMX 1
#endif

enum {
    COB_SGEMM_AMX_MR = 32,
    COB_SGEMM_AMX_NR = 32
};

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

static int cob_sgemm_pack_nr(void)
{
#if defined(COB_USE_APPLE_AMX)
    return COB_SGEMM_AMX_NR;
#else
    return COB_SGEMM_NR;
#endif
}

#if defined(COB_USE_APPLE_AMX)
static void cob_amx_set(void)
{
    __asm__ volatile(".word 0x00201220" ::: "memory");
}

static void cob_amx_clr(void)
{
    __asm__ volatile(".word 0x00201221" ::: "memory");
}

#define COB_AMX_OP(op, operand)                                                         \
    do {                                                                                \
        register uint64_t cob_amx_operand __asm__("x0") = (uint64_t)(operand);          \
        __asm__ volatile(".word %c0"                                                    \
                         :                                                              \
                         : "i"(0x00201000u | ((unsigned)(op) << 5)),                   \
                           "r"(cob_amx_operand)                                        \
                         : "memory");                                                  \
    } while (0)

static void cob_amx_ldx_pair(const float* ptr, uint64_t reg)
{
    const uint64_t operand =
        (1ull << 62) | ((reg & 7ull) << 56) | ((uintptr_t)ptr & ((1ull << 56) - 1ull));
    COB_AMX_OP(0, operand);
}

static void cob_amx_ldx_64(const float* ptr, uint64_t reg)
{
    const uint64_t operand =
        ((reg & 7ull) << 56) | ((uintptr_t)ptr & ((1ull << 56) - 1ull));
    COB_AMX_OP(0, operand);
}

static void cob_amx_ldy_pair(const float* ptr, uint64_t reg)
{
    const uint64_t operand =
        (1ull << 62) | ((reg & 7ull) << 56) | ((uintptr_t)ptr & ((1ull << 56) - 1ull));
    COB_AMX_OP(1, operand);
}

static void cob_amx_ldy_64(const float* ptr, uint64_t reg)
{
    const uint64_t operand =
        ((reg & 7ull) << 56) | ((uintptr_t)ptr & ((1ull << 56) - 1ull));
    COB_AMX_OP(1, operand);
}

static void cob_amx_stz_pair(float* ptr, uint64_t row)
{
    const uint64_t operand =
        (1ull << 62) | ((row & 63ull) << 56) | ((uintptr_t)ptr & ((1ull << 56) - 1ull));
    COB_AMX_OP(5, operand);
}

static void cob_amx_stz_64(float* ptr, uint64_t row)
{
    const uint64_t operand =
        ((row & 63ull) << 56) | ((uintptr_t)ptr & ((1ull << 56) - 1ull));
    COB_AMX_OP(5, operand);
}

static void cob_amx_fma32_16x16(
    uint64_t y_byte_offset,
    uint64_t x_byte_offset,
    uint64_t zrow,
    int zero_z)
{
    const uint64_t operand =
        (y_byte_offset & 0x1ffull) |
        ((x_byte_offset & 0x1ffull) << 10) |
        ((zrow & 63ull) << 20) |
        (zero_z ? (1ull << 27) : 0ull);
    COB_AMX_OP(12, operand);
}

static void cob_amx_fmul32_selrow(uint64_t row, uint64_t xreg, uint64_t yreg, uint64_t zrow)
{
    const uint64_t operand =
        (((yreg & 7ull) << 6) & 0x1ffull) |
        ((((xreg & 7ull) << 6) & 0x1ffull) << 10) |
        ((zrow & 63ull) << 20) |
        (1ull << 27) |
        (((0x20ull | row) & 0x3full) << 41);
    COB_AMX_OP(12, operand);
}

static const float cob_amx_ones_f32[16] __attribute__((aligned(64))) = {
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f
};

static void cob_sgemm_pack_a32_partial(
    float* packed,
    int k,
    const float* a,
    int lda,
    int mr)
{
    for (int p = 0; p < k; ++p) {
        float* dst = packed + (size_t)p * (size_t)COB_SGEMM_AMX_MR;
        for (int i = 0; i < COB_SGEMM_AMX_MR; ++i) {
            dst[i] = i < mr ? a[(size_t)i * (size_t)lda + p] : 0.0f;
        }
    }
}

static void cob_sgemm_pack_a32_full_scalar(float* packed, int k, const float* a, int lda)
{
    for (int p = 0; p < k; ++p) {
        float* dst = packed + (size_t)p * (size_t)COB_SGEMM_AMX_MR;
        for (int i = 0; i < COB_SGEMM_AMX_MR; ++i) {
            dst[i] = a[(size_t)i * (size_t)lda + p];
        }
    }
}

static void cob_sgemm_pack_a32_full(float* packed, int k, const float* a, int lda)
{
    if ((((uintptr_t)a & 63ull) != 0) ||
        (((uintptr_t)lda * sizeof(float) & 63ull) != 0)) {
        cob_sgemm_pack_a32_full_scalar(packed, k, a, lda);
        return;
    }

    cob_amx_ldx_64(cob_amx_ones_f32, 0);
    int p = 0;
    for (; p + 15 < k; p += 16) {
        const float* ap = a + p;
        float* pp = packed + (size_t)p * (size_t)COB_SGEMM_AMX_MR;

        for (int i = 0; i < 8; ++i) {
            cob_amx_ldy_64(ap + (size_t)i * (size_t)lda, (uint64_t)i);
        }
        for (int i = 0; i < 8; ++i) {
            cob_amx_fmul32_selrow((uint64_t)i, 0, (uint64_t)i, 0);
        }

        for (int i = 0; i < 8; ++i) {
            cob_amx_ldy_64(ap + (size_t)(i + 8) * (size_t)lda, (uint64_t)i);
        }
        for (int i = 0; i < 8; ++i) {
            cob_amx_fmul32_selrow((uint64_t)(i + 8), 0, (uint64_t)i, 0);
        }

        for (int i = 0; i < 8; ++i) {
            cob_amx_ldy_64(ap + (size_t)(i + 16) * (size_t)lda, (uint64_t)i);
        }
        for (int i = 0; i < 8; ++i) {
            cob_amx_fmul32_selrow((uint64_t)i, 0, (uint64_t)i, 1);
        }

        for (int i = 0; i < 8; ++i) {
            cob_amx_ldy_64(ap + (size_t)(i + 24) * (size_t)lda, (uint64_t)i);
        }
        for (int i = 0; i < 8; ++i) {
            cob_amx_fmul32_selrow((uint64_t)(i + 8), 0, (uint64_t)i, 1);
        }

        for (int i = 0; i < 16; ++i) {
            cob_amx_stz_64(pp + (size_t)i * (size_t)COB_SGEMM_AMX_MR, (uint64_t)i * 4ull);
            cob_amx_stz_64(
                pp + (size_t)i * (size_t)COB_SGEMM_AMX_MR + 16,
                (uint64_t)i * 4ull + 1ull);
        }
    }

    for (; p < k; ++p) {
        float* dst = packed + (size_t)p * (size_t)COB_SGEMM_AMX_MR;
        for (int i = 0; i < COB_SGEMM_AMX_MR; ++i) {
            dst[i] = a[(size_t)i * (size_t)lda + p];
        }
    }
}

static void cob_sgemm_pack_b32_panel_partial(
    float* packed,
    int k,
    const float* b,
    int ldb,
    int nr)
{
    for (int p = 0; p < k; ++p) {
        float* dst = packed + (size_t)p * (size_t)COB_SGEMM_AMX_NR;
        if (nr == COB_SGEMM_AMX_NR) {
            memcpy(dst, b + (size_t)p * (size_t)ldb, COB_SGEMM_AMX_NR * sizeof(float));
        } else {
            for (int j = 0; j < COB_SGEMM_AMX_NR; ++j) {
                dst[j] = j < nr ? b[(size_t)p * (size_t)ldb + j] : 0.0f;
            }
        }
    }
}

static void cob_sgemm_pack_b32_panel_full(float* packed, int k, const float* b, int ldb)
{
    for (int p = 0; p < k; ++p) {
        float* dst = packed + (size_t)p * (size_t)COB_SGEMM_AMX_NR;
        memcpy(dst, b + (size_t)p * (size_t)ldb, COB_SGEMM_AMX_NR * sizeof(float));
    }
}

static void cob_sgemm_32x32_amx_compute(int k, const float* packed_a, const float* packed_b)
{
    for (int p = 0; p < k; ++p) {
        const int zero_z = p == 0;
        cob_amx_ldy_pair(packed_a + (size_t)p * (size_t)COB_SGEMM_AMX_MR, 0);
        cob_amx_ldx_pair(packed_b + (size_t)p * (size_t)COB_SGEMM_AMX_NR, 0);

        cob_amx_fma32_16x16(0, 0, 0, zero_z);
        cob_amx_fma32_16x16(0, 64, 1, zero_z);
        cob_amx_fma32_16x16(64, 0, 2, zero_z);
        cob_amx_fma32_16x16(64, 64, 3, zero_z);
    }
}

static void cob_sgemm_32x32_amx_packed_full(
    int k,
    const float* packed_a,
    const float* packed_b,
    float* c,
    int ldc)
{
    float row[COB_SGEMM_AMX_NR] __attribute__((aligned(128)));
    const int direct_store_128 =
        (((uintptr_t)c & 127ull) == 0) &&
        (((uintptr_t)ldc * sizeof(float) & 127ull) == 0);
    const int direct_store_64 =
        !direct_store_128 &&
        (((uintptr_t)c & 63ull) == 0) &&
        (((uintptr_t)ldc * sizeof(float) & 63ull) == 0);

    cob_sgemm_32x32_amx_compute(k, packed_a, packed_b);

    if (direct_store_128) {
        for (int i = 0; i < 16; ++i) {
            cob_amx_stz_pair(c + (size_t)i * (size_t)ldc, (uint64_t)i * 4ull);
        }
        for (int i = 0; i < 16; ++i) {
            cob_amx_stz_pair(
                c + (size_t)(i + 16) * (size_t)ldc,
                (uint64_t)i * 4ull + 2ull);
        }
        return;
    }

    if (direct_store_64) {
        for (int i = 0; i < 16; ++i) {
            float* rowp = c + (size_t)i * (size_t)ldc;
            cob_amx_stz_64(rowp, (uint64_t)i * 4ull);
            cob_amx_stz_64(rowp + 16, (uint64_t)i * 4ull + 1ull);
        }
        for (int i = 0; i < 16; ++i) {
            float* rowp = c + (size_t)(i + 16) * (size_t)ldc;
            cob_amx_stz_64(rowp, (uint64_t)i * 4ull + 2ull);
            cob_amx_stz_64(rowp + 16, (uint64_t)i * 4ull + 3ull);
        }
        return;
    }

    for (int i = 0; i < 16; ++i) {
        cob_amx_stz_pair(row, (uint64_t)i * 4ull);
        memcpy(c + (size_t)i * (size_t)ldc, row, sizeof(row));
    }
    for (int i = 0; i < 16; ++i) {
        cob_amx_stz_pair(row, (uint64_t)i * 4ull + 2ull);
        memcpy(c + (size_t)(i + 16) * (size_t)ldc, row, sizeof(row));
    }
}

static void cob_sgemm_32x32_amx_packed_partial(
    int k,
    const float* packed_a,
    const float* packed_b,
    float* c,
    int ldc,
    int mr,
    int nr)
{
    if (mr == COB_SGEMM_AMX_MR && nr == COB_SGEMM_AMX_NR) {
        cob_sgemm_32x32_amx_packed_full(k, packed_a, packed_b, c, ldc);
        return;
    }

    float row[COB_SGEMM_AMX_NR] __attribute__((aligned(128)));

    cob_sgemm_32x32_amx_compute(k, packed_a, packed_b);

    const int top_rows = cob_min_i32(mr, 16);
    const int direct_store_64 =
        (nr == 16 || nr == COB_SGEMM_AMX_NR) &&
        (((uintptr_t)c & 63ull) == 0) &&
        (((uintptr_t)ldc * sizeof(float) & 63ull) == 0);

    if (direct_store_64) {
        for (int i = 0; i < top_rows; ++i) {
            float* rowp = c + (size_t)i * (size_t)ldc;
            cob_amx_stz_64(rowp, (uint64_t)i * 4ull);
            if (nr == COB_SGEMM_AMX_NR) {
                cob_amx_stz_64(rowp + 16, (uint64_t)i * 4ull + 1ull);
            }
        }

        const int bottom_rows = mr > 16 ? mr - 16 : 0;
        for (int i = 0; i < bottom_rows; ++i) {
            float* rowp = c + (size_t)(i + 16) * (size_t)ldc;
            cob_amx_stz_64(rowp, (uint64_t)i * 4ull + 2ull);
            if (nr == COB_SGEMM_AMX_NR) {
                cob_amx_stz_64(rowp + 16, (uint64_t)i * 4ull + 3ull);
            }
        }
        return;
    }

    for (int i = 0; i < top_rows; ++i) {
        cob_amx_stz_pair(row, (uint64_t)i * 4ull);
        memcpy(c + (size_t)i * (size_t)ldc, row, (size_t)nr * sizeof(float));
    }
    const int bottom_rows = mr > 16 ? mr - 16 : 0;
    for (int i = 0; i < bottom_rows; ++i) {
        cob_amx_stz_pair(row, (uint64_t)i * 4ull + 2ull);
        memcpy(c + (size_t)(i + 16) * (size_t)ldc, row, (size_t)nr * sizeof(float));
    }
}
#endif

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
    packed->nr = cob_sgemm_pack_nr();
    packed->bytes = 0;
    packed->data = NULL;

    if (k <= 0 || n <= 0 || b == NULL || ldb < n) {
        return -1;
    }

    const int nr_pack = cob_sgemm_pack_nr();
    const int panels = (n + nr_pack - 1) / nr_pack;
    const size_t count = (size_t)panels * (size_t)k * (size_t)nr_pack;
    const size_t bytes = count * sizeof(float);
    const size_t alignment = nr_pack == COB_SGEMM_AMX_NR ? 128 : 64;
    float* data = (float*)cob_aligned_alloc(alignment, bytes);
    if (data == NULL) {
        return -1;
    }

    for (int panel = 0; panel < panels; ++panel) {
        const int col0 = panel * nr_pack;
        float* dst_panel = data + (size_t)panel * (size_t)k * (size_t)nr_pack;

        for (int p = 0; p < k; ++p) {
            float* dst = dst_panel + (size_t)p * (size_t)nr_pack;
            for (int j = 0; j < nr_pack; ++j) {
                const int col = col0 + j;
                dst[j] = col < n ? b[p * ldb + col] : 0.0f;
            }
        }
    }

    packed->k = k;
    packed->n = n;
    packed->nr = nr_pack;
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
    packed->nr = cob_sgemm_pack_nr();
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
    int packed_nr,
    float* c,
    int ldc)
{
    for (int i = 0; i < mr; ++i) {
        for (int j = 0; j < nr; ++j) {
            float sum = 0.0f;
            for (int p = 0; p < k; ++p) {
                sum += a[i * lda + p] * packed_b[(size_t)p * (size_t)packed_nr + j];
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

static void cob_sgemm_rowmajor_packed_b_fallback(
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
        packed_b->nr <= 0 || lda < k || ldc < n) {
        return;
    }

    const int packed_nr = packed_b->nr;
    for (int jc = 0; jc < n; jc += packed_nr) {
        const int nr = cob_min_i32(packed_nr, n - jc);
        const int panel = jc / packed_nr;
        const float* bp = packed_b->data +
            (size_t)panel * (size_t)k * (size_t)packed_nr;

        for (int ic = 0; ic < m; ic += COB_SGEMM_MR) {
            const int mr = cob_min_i32(COB_SGEMM_MR, m - ic);
            const float* at = a + (size_t)ic * (size_t)lda;
            float* ct = c + (size_t)ic * (size_t)ldc + jc;

#if defined(COB_USE_NEON)
            if (packed_nr == COB_SGEMM_NR && mr == COB_SGEMM_MR && nr == COB_SGEMM_NR) {
                cob_sgemm_8x8_neon_packed_b(k, at, lda, bp, ct, ldc);
            } else
#endif
            {
                cob_sgemm_tile_scalar_packed_b(mr, nr, k, at, lda, bp, packed_nr, ct, ldc);
            }
        }
    }
}

#if defined(COB_USE_APPLE_AMX)
static int cob_sgemm_rowmajor_amx_from_packed_b32(
    int m,
    int n,
    int k,
    const float* a,
    int lda,
    const cob_packed_b_f32* packed_b,
    float* c,
    int ldc)
{
    if (packed_b->nr != COB_SGEMM_AMX_NR) {
        return 0;
    }

    const size_t a_panel_floats = (size_t)k * (size_t)COB_SGEMM_AMX_MR;
    const int b_panels = (n + COB_SGEMM_AMX_NR - 1) / COB_SGEMM_AMX_NR;
    const size_t b_panel_floats = (size_t)k * (size_t)COB_SGEMM_AMX_NR;
    float* packed_a = (float*)cob_aligned_alloc(128, a_panel_floats * sizeof(float));
    if (packed_a == NULL) {
        return 0;
    }

    if (m % COB_SGEMM_AMX_MR == 0 && n % COB_SGEMM_AMX_NR == 0) {
        cob_amx_set();
        for (int ic = 0; ic < m; ic += COB_SGEMM_AMX_MR) {
            cob_sgemm_pack_a32_full(packed_a, k, a + (size_t)ic * (size_t)lda, lda);
            for (int panel = 0; panel < b_panels; ++panel) {
                cob_sgemm_32x32_amx_packed_full(
                    k,
                    packed_a,
                    packed_b->data + (size_t)panel * b_panel_floats,
                    c + (size_t)ic * (size_t)ldc +
                        (size_t)panel * (size_t)COB_SGEMM_AMX_NR,
                    ldc);
            }
        }
        cob_amx_clr();

        free(packed_a);
        return 1;
    }

    cob_amx_set();
    for (int ic = 0; ic < m; ic += COB_SGEMM_AMX_MR) {
        const int mr = cob_min_i32(COB_SGEMM_AMX_MR, m - ic);
        if (mr == COB_SGEMM_AMX_MR) {
            cob_sgemm_pack_a32_full(packed_a, k, a + (size_t)ic * (size_t)lda, lda);
        } else {
            cob_sgemm_pack_a32_partial(packed_a, k, a + (size_t)ic * (size_t)lda, lda, mr);
        }

        for (int panel = 0; panel < b_panels; ++panel) {
            const int jc = panel * COB_SGEMM_AMX_NR;
            const int nr = cob_min_i32(COB_SGEMM_AMX_NR, n - jc);
            const float* bp = packed_b->data + (size_t)panel * b_panel_floats;
            float* ct = c + (size_t)ic * (size_t)ldc + jc;
            if (mr == COB_SGEMM_AMX_MR && nr == COB_SGEMM_AMX_NR) {
                cob_sgemm_32x32_amx_packed_full(k, packed_a, bp, ct, ldc);
            } else {
                cob_sgemm_32x32_amx_packed_partial(k, packed_a, bp, ct, ldc, mr, nr);
            }
        }
    }
    cob_amx_clr();

    free(packed_a);
    return 1;
}

static int cob_sgemm_rowmajor_amx_from_packed_b8(
    int m,
    int n,
    int k,
    const float* a,
    int lda,
    const cob_packed_b_f32* packed_b,
    float* c,
    int ldc)
{
    const int m32 = (m / COB_SGEMM_AMX_MR) * COB_SGEMM_AMX_MR;
    const int n32 = (n / COB_SGEMM_AMX_NR) * COB_SGEMM_AMX_NR;
    if (m32 == 0 || n32 == 0 || packed_b->nr != COB_SGEMM_NR) {
        return 0;
    }

    const int b32_panels = n32 / COB_SGEMM_AMX_NR;
    const size_t a_bytes =
        (size_t)k * (size_t)COB_SGEMM_AMX_MR * sizeof(float);
    const size_t b32_panel_floats = (size_t)k * (size_t)COB_SGEMM_AMX_NR;
    const size_t b32_bytes = (size_t)b32_panels * b32_panel_floats * sizeof(float);

    float* packed_a = (float*)cob_aligned_alloc(128, a_bytes);
    float* packed_b32 = (float*)cob_aligned_alloc(128, b32_bytes);
    if (packed_a == NULL || packed_b32 == NULL) {
        free(packed_a);
        free(packed_b32);
        return 0;
    }

    for (int panel32 = 0; panel32 < b32_panels; ++panel32) {
        float* dst_panel = packed_b32 + (size_t)panel32 * b32_panel_floats;
        for (int p = 0; p < k; ++p) {
            float* dst = dst_panel + (size_t)p * (size_t)COB_SGEMM_AMX_NR;
            for (int chunk = 0; chunk < 4; ++chunk) {
                const int panel8 = panel32 * 4 + chunk;
                const float* src = packed_b->data +
                    ((size_t)panel8 * (size_t)k + (size_t)p) *
                        (size_t)COB_SGEMM_NR;
                memcpy(dst + chunk * COB_SGEMM_NR, src, COB_SGEMM_NR * sizeof(float));
            }
        }
    }

    cob_amx_set();
    for (int ic = 0; ic < m32; ic += COB_SGEMM_AMX_MR) {
        cob_sgemm_pack_a32_full(packed_a, k, a + (size_t)ic * (size_t)lda, lda);
        for (int panel = 0; panel < b32_panels; ++panel) {
            cob_sgemm_32x32_amx_packed_full(
                k,
                packed_a,
                packed_b32 + (size_t)panel * b32_panel_floats,
                c + (size_t)ic * (size_t)ldc +
                    (size_t)panel * (size_t)COB_SGEMM_AMX_NR,
                ldc);
        }
    }
    cob_amx_clr();

    free(packed_a);
    free(packed_b32);

    if (n32 < n) {
        cob_packed_b_f32 tail_b = *packed_b;
        tail_b.n = packed_b->n - n32;
        tail_b.data = packed_b->data +
            (size_t)(n32 / COB_SGEMM_NR) * (size_t)k * (size_t)COB_SGEMM_NR;
        tail_b.bytes = packed_b->bytes -
            (size_t)(n32 / COB_SGEMM_NR) * (size_t)k *
                (size_t)COB_SGEMM_NR * sizeof(float);

        cob_sgemm_rowmajor_packed_b_fallback(
            m32,
            n - n32,
            k,
            a,
            lda,
            &tail_b,
            c + n32,
            ldc);
    }
    if (m32 < m) {
        cob_sgemm_rowmajor_packed_b_fallback(
            m - m32,
            n,
            k,
            a + (size_t)m32 * (size_t)lda,
            lda,
            packed_b,
            c + (size_t)m32 * (size_t)ldc,
            ldc);
    }

    return 1;
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
#if defined(COB_USE_APPLE_AMX)
    if (m > 0 && n > 0 && k > 0 && a != NULL && packed_b != NULL && c != NULL &&
        packed_b->data != NULL && packed_b->k == k && packed_b->n >= n &&
        lda >= k && ldc >= n &&
        (cob_sgemm_rowmajor_amx_from_packed_b32(m, n, k, a, lda, packed_b, c, ldc) ||
            cob_sgemm_rowmajor_amx_from_packed_b8(m, n, k, a, lda, packed_b, c, ldc))) {
        return;
    }
#endif

    cob_sgemm_rowmajor_packed_b_fallback(m, n, k, a, lda, packed_b, c, ldc);
}

static void cob_sgemm_rowmajor_fallback(
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

    cob_sgemm_rowmajor_packed_b_fallback(m, n, k, a, lda, &packed_b, c, ldc);
    cob_sgemm_free_packed_b(&packed_b);
}

#if defined(COB_USE_APPLE_AMX)
static int cob_sgemm_rowmajor_amx(
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
    const int b_panels = (n + COB_SGEMM_AMX_NR - 1) / COB_SGEMM_AMX_NR;
    const size_t a_panel_floats = (size_t)k * (size_t)COB_SGEMM_AMX_MR;
    const size_t b_panel_floats = (size_t)k * (size_t)COB_SGEMM_AMX_NR;
    const size_t b_bytes = (size_t)b_panels * b_panel_floats * sizeof(float);

    float* packed_a = (float*)cob_aligned_alloc(128, a_panel_floats * sizeof(float));
    float* packed_b = (float*)cob_aligned_alloc(128, b_bytes);
    if (packed_a == NULL || packed_b == NULL) {
        free(packed_a);
        free(packed_b);
        return 0;
    }

    for (int panel = 0; panel < b_panels; ++panel) {
        const int jc = panel * COB_SGEMM_AMX_NR;
        const int nr = cob_min_i32(COB_SGEMM_AMX_NR, n - jc);
        if (nr == COB_SGEMM_AMX_NR) {
            cob_sgemm_pack_b32_panel_full(
                packed_b + (size_t)panel * b_panel_floats,
                k,
                b + jc,
                ldb);
        } else {
            cob_sgemm_pack_b32_panel_partial(
                packed_b + (size_t)panel * b_panel_floats,
                k,
                b + jc,
                ldb,
                nr);
        }
    }

    if (m % COB_SGEMM_AMX_MR == 0 && n % COB_SGEMM_AMX_NR == 0) {
        cob_amx_set();
        for (int ic = 0; ic < m; ic += COB_SGEMM_AMX_MR) {
            cob_sgemm_pack_a32_full(packed_a, k, a + (size_t)ic * (size_t)lda, lda);
            for (int panel = 0; panel < b_panels; ++panel) {
                cob_sgemm_32x32_amx_packed_full(
                    k,
                    packed_a,
                    packed_b + (size_t)panel * b_panel_floats,
                    c + (size_t)ic * (size_t)ldc +
                        (size_t)panel * (size_t)COB_SGEMM_AMX_NR,
                    ldc);
            }
        }
        cob_amx_clr();

        free(packed_a);
        free(packed_b);
        return 1;
    }

    cob_amx_set();
    for (int ic = 0; ic < m; ic += COB_SGEMM_AMX_MR) {
        const int mr = cob_min_i32(COB_SGEMM_AMX_MR, m - ic);
        if (mr == COB_SGEMM_AMX_MR) {
            cob_sgemm_pack_a32_full(packed_a, k, a + (size_t)ic * (size_t)lda, lda);
        } else {
            cob_sgemm_pack_a32_partial(packed_a, k, a + (size_t)ic * (size_t)lda, lda, mr);
        }

        for (int panel = 0; panel < b_panels; ++panel) {
            const int jc = panel * COB_SGEMM_AMX_NR;
            const int nr = cob_min_i32(COB_SGEMM_AMX_NR, n - jc);
            const float* bp = packed_b + (size_t)panel * b_panel_floats;
            float* ct = c + (size_t)ic * (size_t)ldc + jc;
            if (mr == COB_SGEMM_AMX_MR && nr == COB_SGEMM_AMX_NR) {
                cob_sgemm_32x32_amx_packed_full(k, packed_a, bp, ct, ldc);
            } else {
                cob_sgemm_32x32_amx_packed_partial(k, packed_a, bp, ct, ldc, mr, nr);
            }
        }
    }
    cob_amx_clr();

    free(packed_a);
    free(packed_b);

    return 1;
}
#endif

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
#if defined(COB_USE_APPLE_AMX)
    if (m > 0 && n > 0 && k > 0 && a != NULL && b != NULL && c != NULL &&
        lda >= k && ldb >= n && ldc >= n &&
        cob_sgemm_rowmajor_amx(m, n, k, a, lda, b, ldb, c, ldc)) {
        return;
    }
#endif

    cob_sgemm_rowmajor_fallback(m, n, k, a, lda, b, ldb, c, ldc);
}
