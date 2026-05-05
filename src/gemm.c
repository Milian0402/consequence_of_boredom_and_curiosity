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

#if defined(__APPLE__) && defined(__aarch64__) && !defined(COB_DISABLE_APPLE_SME)
#if defined(__has_include)
#if __has_include(<arm_sme.h>)
#define COB_USE_APPLE_SME 1
#include <arm_sme.h>
#include <sys/sysctl.h>
#endif
#endif
#endif

enum {
    COB_SGEMM_AMX_MR = 32,
    COB_SGEMM_AMX_NR = 32
};

#ifndef COB_SGEMM_AMX_MC
#define COB_SGEMM_AMX_MC 384
#endif

#ifndef COB_SGEMM_AMX_PACKED_LARGE_MC
#define COB_SGEMM_AMX_PACKED_LARGE_MC 512
#endif

#ifndef COB_SGEMM_SME_DIRECT_MC
#define COB_SGEMM_SME_DIRECT_MC 256
#endif

#ifndef COB_SGEMM_AMX_STRIDED_B_MAX_N
#define COB_SGEMM_AMX_STRIDED_B_MAX_N 832
#endif

#ifndef COB_SGEMM_AMX_STRIDED_B_EXTRA_N
#define COB_SGEMM_AMX_STRIDED_B_EXTRA_N 960
#endif

#ifndef COB_SGEMM_AMX_STRIDED_B_EXTRA_N2
#define COB_SGEMM_AMX_STRIDED_B_EXTRA_N2 1088
#endif

#ifndef COB_SGEMM_AMX_STRIDED_B_EXTRA_N3
#define COB_SGEMM_AMX_STRIDED_B_EXTRA_N3 1152
#endif

#ifndef COB_SGEMM_AMX_STRIDED_B_EXTRA_N4
#define COB_SGEMM_AMX_STRIDED_B_EXTRA_N4 1216
#endif

/* Direct source-B loads are slower for this power-of-two row stride; packed B avoids the conflict. */
#ifndef COB_SGEMM_AMX_STRIDED_B_CONFLICT_LDB
#define COB_SGEMM_AMX_STRIDED_B_CONFLICT_LDB 512
#endif

#ifndef COB_SGEMM_SKINNY_SME_KC
#define COB_SGEMM_SKINNY_SME_KC 512
#endif

#ifndef COB_SGEMM_SKINNY_SME_LARGE_KC
#define COB_SGEMM_SKINNY_SME_LARGE_KC 1024
#endif

#ifndef COB_SGEMM_M64_SME_REUSE_NC
#define COB_SGEMM_M64_SME_REUSE_NC 512
#endif

#ifndef COB_SGEMM_M64_SME_LONG_WIDE_NC
#define COB_SGEMM_M64_SME_LONG_WIDE_NC 256
#endif

#ifndef COB_SGEMM_M64_SME_LONG_N_K512_MIN_N
#define COB_SGEMM_M64_SME_LONG_N_K512_MIN_N 4096
#endif

#ifndef COB_SGEMM_M96_128_SME_REUSE_K512_MIN_N
#define COB_SGEMM_M96_128_SME_REUSE_K512_MIN_N 4096
#endif

#ifndef COB_SGEMM_M96_128_SME_REUSE_K1024_MIN_N
#define COB_SGEMM_M96_128_SME_REUSE_K1024_MIN_N 4096
#endif

#ifndef COB_SGEMM_M64_SME_WIDE_KC
#define COB_SGEMM_M64_SME_WIDE_KC 1024
#endif

#ifndef COB_SGEMM_M64_SME_WIDE_K1536_KC
#define COB_SGEMM_M64_SME_WIDE_K1536_KC 1536
#endif

#ifndef COB_SGEMM_M64_SME_WIDE_LARGE_KC
#define COB_SGEMM_M64_SME_WIDE_LARGE_KC 1280
#endif

#ifndef COB_SGEMM_SME_DIRECT_MAX_N
#define COB_SGEMM_SME_DIRECT_MAX_N 1216
#endif

#ifndef COB_SGEMM_SME_DIRECT_EXTRA_N_MIN
#define COB_SGEMM_SME_DIRECT_EXTRA_N_MIN 1280
#endif

#ifndef COB_SGEMM_SME_DIRECT_EXTRA_N_MAX
#define COB_SGEMM_SME_DIRECT_EXTRA_N_MAX 1472
#endif

#ifndef COB_SGEMM_SME_PACKED_MAX_N
#define COB_SGEMM_SME_PACKED_MAX_N 1152
#endif

#ifndef COB_SGEMM_PACK_B_PREFETCH_DISTANCE
#define COB_SGEMM_PACK_B_PREFETCH_DISTANCE 64
#endif

#ifndef COB_SGEMM_M64_SME_B_PREFETCH_DISTANCE
#define COB_SGEMM_M64_SME_B_PREFETCH_DISTANCE 32
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

static int cob_sgemm_pack_nr(void)
{
#if defined(COB_USE_APPLE_AMX)
    return COB_SGEMM_AMX_NR;
#else
    return COB_SGEMM_NR;
#endif
}

static int cob_sgemm_sme_direct_extra_n_shape(int m, int n, int k)
{
    if (n < COB_SGEMM_SME_DIRECT_EXTRA_N_MIN ||
        n > COB_SGEMM_SME_DIRECT_EXTRA_N_MAX ||
        ((n - COB_SGEMM_SME_DIRECT_EXTRA_N_MIN) % 64) != 0) {
        return 0;
    }
    if (m >= 832 && m <= 960 && k >= 832 && k <= 1152) {
        return 1;
    }
    if (m == n && n <= 1408 && k >= 832 && k <= 960) {
        return 1;
    }
    return 0;
}

static int cob_sgemm_m64_sme_direct_prefetch_shape(int n, int k)
{
    if (n < 1088 || n > 4096 || (n % 512) == 0) {
        return 0;
    }
    if (k >= 2048) {
        return 1;
    }
    return k == 1536 && n >= 1408;
}

static int cob_sgemm_m64_sme_large_kc_shape(int n, int k)
{
    if (n == 1024) {
        return 1;
    }
    if (k == 2048) {
        return (n >= 1088 && n <= 1280) || n == 1600;
    }
    return 0;
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
#if COB_SGEMM_PACK_B_PREFETCH_DISTANCE > 0
        if (p + COB_SGEMM_PACK_B_PREFETCH_DISTANCE < k) {
            __builtin_prefetch(
                b + (size_t)(p + COB_SGEMM_PACK_B_PREFETCH_DISTANCE) * (size_t)ldb,
                0,
                1);
        }
#endif
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

static void cob_sgemm_32x32_amx_compute_strided_b(
    int k,
    const float* packed_a,
    const float* b,
    int ldb)
{
    for (int p = 0; p < k; ++p) {
        const int zero_z = p == 0;
        cob_amx_ldy_pair(packed_a + (size_t)p * (size_t)COB_SGEMM_AMX_MR, 0);
        cob_amx_ldx_pair(b + (size_t)p * (size_t)ldb, 0);

        cob_amx_fma32_16x16(0, 0, 0, zero_z);
        cob_amx_fma32_16x16(0, 64, 1, zero_z);
        cob_amx_fma32_16x16(64, 0, 2, zero_z);
        cob_amx_fma32_16x16(64, 64, 3, zero_z);
    }
}

static void cob_sgemm_32x32_amx_store_full(float* c, int ldc)
{
    float row[COB_SGEMM_AMX_NR] __attribute__((aligned(128)));
    const int direct_store_128 =
        (((uintptr_t)c & 127ull) == 0) &&
        (((uintptr_t)ldc * sizeof(float) & 127ull) == 0);
    const int direct_store_64 =
        !direct_store_128 &&
        (((uintptr_t)c & 63ull) == 0) &&
        (((uintptr_t)ldc * sizeof(float) & 63ull) == 0);

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

static void cob_sgemm_32x32_amx_packed_full(
    int k,
    const float* packed_a,
    const float* packed_b,
    float* c,
    int ldc)
{
    cob_sgemm_32x32_amx_compute(k, packed_a, packed_b);
    cob_sgemm_32x32_amx_store_full(c, ldc);
}

static void cob_sgemm_32x32_amx_strided_b_full(
    int k,
    const float* packed_a,
    const float* b,
    int ldb,
    float* c,
    int ldc)
{
    cob_sgemm_32x32_amx_compute_strided_b(k, packed_a, b, ldb);
    cob_sgemm_32x32_amx_store_full(c, ldc);
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

#if defined(COB_USE_APPLE_SME)
__attribute__((target("sme"))) static int cob_apple_sme_f32_lanes(void)
{
    return (int)svcntsw();
}

static int cob_apple_sme2p1_available(void)
{
    static int cached = -1;
    if (cached >= 0) {
        return cached;
    }

    int value = 0;
    size_t size = sizeof(value);
    cached =
        sysctlbyname("hw.optional.arm.FEAT_SME2p1", &value, &size, NULL, 0) == 0 &&
        value != 0 && __arm_has_sme() && cob_apple_sme_f32_lanes() == 16;
    return cached;
}

__arm_new("za") __arm_locally_streaming __attribute__((target("sme,sme2,sme2p1")))
static void cob_sgemm_16x64_sme_packed_b32(
    int a_panels16,
    int b_panels64,
    int k,
    const float* packed_a32,
    const float* packed_b32,
    float* c,
    int ldc)
{
    const svbool_t pg = svptrue_b32();
    const size_t b_panel_floats = (size_t)k * (size_t)COB_SGEMM_AMX_NR;

    for (int ap16 = 0; ap16 < a_panels16; ++ap16) {
        const int ap32 = ap16 / 2;
        const int a_offset = (ap16 & 1) * 16;
        const float* a_panel =
            packed_a32 + (size_t)ap32 * (size_t)k * (size_t)COB_SGEMM_AMX_MR +
            (size_t)a_offset;

        for (int bp64 = 0; bp64 < b_panels64; ++bp64) {
            const float* b_panel0 = packed_b32 + (size_t)(bp64 * 2) * b_panel_floats;
            const float* b_panel1 = b_panel0 + b_panel_floats;
            float* ct =
                c + (size_t)ap16 * 16u * (size_t)ldc + (size_t)bp64 * 64u;

            svzero_za();
            for (int p = 0; p < k; ++p) {
                const svfloat32_t av =
                    svld1(pg, a_panel + (size_t)p * (size_t)COB_SGEMM_AMX_MR);
                const float* b0 = b_panel0 + (size_t)p * (size_t)COB_SGEMM_AMX_NR;
                const float* b1 = b_panel1 + (size_t)p * (size_t)COB_SGEMM_AMX_NR;
                const svfloat32_t bv0 = svld1(pg, b0);
                const svfloat32_t bv1 = svld1(pg, b0 + 16);
                const svfloat32_t bv2 = svld1(pg, b1);
                const svfloat32_t bv3 = svld1(pg, b1 + 16);
                svmopa_za32_f32_m(0, pg, pg, av, bv0);
                svmopa_za32_f32_m(1, pg, pg, av, bv1);
                svmopa_za32_f32_m(2, pg, pg, av, bv2);
                svmopa_za32_f32_m(3, pg, pg, av, bv3);
            }

            for (int i = 0; i < 16; ++i) {
                float* row = ct + (size_t)i * (size_t)ldc;
                svst1(pg, row, svreadz_hor_za32_f32(0, (uint32_t)i));
                svst1(pg, row + 16, svreadz_hor_za32_f32(1, (uint32_t)i));
                svst1(pg, row + 32, svreadz_hor_za32_f32(2, (uint32_t)i));
                svst1(pg, row + 48, svreadz_hor_za32_f32(3, (uint32_t)i));
            }
        }
    }
}

__arm_new("za") __arm_locally_streaming __attribute__((target("sme,sme2,sme2p1")))
static void cob_sgemm_16x64_sme_strided_b32(
    int a_panels16,
    int b_panels64,
    int k,
    const float* packed_a32,
    const float* b,
    int ldb,
    float* c,
    int ldc,
    int accumulate)
{
    const svbool_t pg = svptrue_b32();
    const svcount_t pg4 = svptrue_c32();

    for (int ap16 = 0; ap16 < a_panels16; ++ap16) {
        const int ap32 = ap16 / 2;
        const int a_offset = (ap16 & 1) * 16;
        const float* a_panel =
            packed_a32 + (size_t)ap32 * (size_t)k * (size_t)COB_SGEMM_AMX_MR +
            (size_t)a_offset;

        for (int bp64 = 0; bp64 < b_panels64; ++bp64) {
            const int col0 = bp64 * 64;
            float* ct = c + (size_t)ap16 * 16u * (size_t)ldc + (size_t)col0;

            svzero_za();
            for (int p = 0; p < k; ++p) {
                const svfloat32_t av =
                    svld1(pg, a_panel + (size_t)p * (size_t)COB_SGEMM_AMX_MR);
                const float* brow = b + (size_t)p * (size_t)ldb + (size_t)col0;
                const svfloat32x4_t bv = svld1_x4(pg4, brow);
                const svfloat32_t bv0 = svget4(bv, 0);
                const svfloat32_t bv1 = svget4(bv, 1);
                const svfloat32_t bv2 = svget4(bv, 2);
                const svfloat32_t bv3 = svget4(bv, 3);
                svmopa_za32_f32_m(0, pg, pg, av, bv0);
                svmopa_za32_f32_m(1, pg, pg, av, bv1);
                svmopa_za32_f32_m(2, pg, pg, av, bv2);
                svmopa_za32_f32_m(3, pg, pg, av, bv3);
            }

            for (int i = 0; i < 16; ++i) {
                float* row = ct + (size_t)i * (size_t)ldc;
                svfloat32_t c0 = svreadz_hor_za32_f32(0, (uint32_t)i);
                svfloat32_t c1 = svreadz_hor_za32_f32(1, (uint32_t)i);
                svfloat32_t c2 = svreadz_hor_za32_f32(2, (uint32_t)i);
                svfloat32_t c3 = svreadz_hor_za32_f32(3, (uint32_t)i);
                if (accumulate) {
                    c0 = svadd_f32_x(pg, c0, svld1(pg, row));
                    c1 = svadd_f32_x(pg, c1, svld1(pg, row + 16));
                    c2 = svadd_f32_x(pg, c2, svld1(pg, row + 32));
                    c3 = svadd_f32_x(pg, c3, svld1(pg, row + 48));
                }
                svst1(pg, row, c0);
                svst1(pg, row + 16, c1);
                svst1(pg, row + 32, c2);
                svst1(pg, row + 48, c3);
            }
        }
    }
}

__arm_new("za") __arm_locally_streaming __attribute__((target("sme,sme2,sme2p1")))
static void cob_sgemm_16x64_sme_strided_b32_prefetch2(
    int a_panels16,
    int b_panels64,
    int k,
    const float* packed_a32,
    const float* b,
    int ldb,
    float* c,
    int ldc,
    int accumulate)
{
    const svbool_t pg = svptrue_b32();
    const svcount_t pg4 = svptrue_c32();

    for (int ap16 = 0; ap16 < a_panels16; ++ap16) {
        const int ap32 = ap16 / 2;
        const int a_offset = (ap16 & 1) * 16;
        const float* a_panel =
            packed_a32 + (size_t)ap32 * (size_t)k * (size_t)COB_SGEMM_AMX_MR +
            (size_t)a_offset;

        for (int bp64 = 0; bp64 < b_panels64; ++bp64) {
            const int col0 = bp64 * 64;
            float* ct = c + (size_t)ap16 * 16u * (size_t)ldc + (size_t)col0;

            svzero_za();
            for (int p = 0; p < k; ++p) {
                const svfloat32_t av =
                    svld1(pg, a_panel + (size_t)p * (size_t)COB_SGEMM_AMX_MR);
                const float* brow = b + (size_t)p * (size_t)ldb + (size_t)col0;
                if (p + COB_SGEMM_M64_SME_B_PREFETCH_DISTANCE < k) {
                    const float* pf =
                        b + (size_t)(p + COB_SGEMM_M64_SME_B_PREFETCH_DISTANCE) *
                            (size_t)ldb + (size_t)col0;
                    __builtin_prefetch(pf, 0, 1);
                    __builtin_prefetch(pf + 32, 0, 1);
                }
                const svfloat32x4_t bv = svld1_x4(pg4, brow);
                const svfloat32_t bv0 = svget4(bv, 0);
                const svfloat32_t bv1 = svget4(bv, 1);
                const svfloat32_t bv2 = svget4(bv, 2);
                const svfloat32_t bv3 = svget4(bv, 3);
                svmopa_za32_f32_m(0, pg, pg, av, bv0);
                svmopa_za32_f32_m(1, pg, pg, av, bv1);
                svmopa_za32_f32_m(2, pg, pg, av, bv2);
                svmopa_za32_f32_m(3, pg, pg, av, bv3);
            }

            for (int i = 0; i < 16; ++i) {
                float* row = ct + (size_t)i * (size_t)ldc;
                svfloat32_t c0 = svreadz_hor_za32_f32(0, (uint32_t)i);
                svfloat32_t c1 = svreadz_hor_za32_f32(1, (uint32_t)i);
                svfloat32_t c2 = svreadz_hor_za32_f32(2, (uint32_t)i);
                svfloat32_t c3 = svreadz_hor_za32_f32(3, (uint32_t)i);
                if (accumulate) {
                    c0 = svadd_f32_x(pg, c0, svld1(pg, row));
                    c1 = svadd_f32_x(pg, c1, svld1(pg, row + 16));
                    c2 = svadd_f32_x(pg, c2, svld1(pg, row + 32));
                    c3 = svadd_f32_x(pg, c3, svld1(pg, row + 48));
                }
                svst1(pg, row, c0);
                svst1(pg, row + 16, c1);
                svst1(pg, row + 32, c2);
                svst1(pg, row + 48, c3);
            }
        }
    }
}

__arm_new("za") __arm_locally_streaming __attribute__((target("sme,sme2,sme2p1")))
static void cob_sgemm_16x64_sme_strided_b_pack_b32(
    int b_panels64,
    int k,
    const float* a_panel,
    const float* b,
    int ldb,
    float* packed_b64,
    float* c,
    int ldc,
    int accumulate)
{
    const svbool_t pg = svptrue_b32();
    const size_t b_panel_floats = (size_t)k * 64u;

    for (int bp64 = 0; bp64 < b_panels64; ++bp64) {
        const int col0 = bp64 * 64;
        float* ct = c + (size_t)col0;
        float* packed_panel = packed_b64 + (size_t)bp64 * b_panel_floats;

        svzero_za();
        for (int p = 0; p < k; ++p) {
            const svfloat32_t av =
                svld1(pg, a_panel + (size_t)p * (size_t)COB_SGEMM_AMX_MR);
            const float* brow = b + (size_t)p * (size_t)ldb + (size_t)col0;
            float* pbrow = packed_panel + (size_t)p * 64u;
            const svfloat32_t bv0 = svld1(pg, brow);
            const svfloat32_t bv1 = svld1(pg, brow + 16);
            const svfloat32_t bv2 = svld1(pg, brow + 32);
            const svfloat32_t bv3 = svld1(pg, brow + 48);
            svmopa_za32_f32_m(0, pg, pg, av, bv0);
            svmopa_za32_f32_m(1, pg, pg, av, bv1);
            svmopa_za32_f32_m(2, pg, pg, av, bv2);
            svmopa_za32_f32_m(3, pg, pg, av, bv3);
            svst1(pg, pbrow, bv0);
            svst1(pg, pbrow + 16, bv1);
            svst1(pg, pbrow + 32, bv2);
            svst1(pg, pbrow + 48, bv3);
        }

        for (int i = 0; i < 16; ++i) {
            float* row = ct + (size_t)i * (size_t)ldc;
            svfloat32_t c0 = svreadz_hor_za32_f32(0, (uint32_t)i);
            svfloat32_t c1 = svreadz_hor_za32_f32(1, (uint32_t)i);
            svfloat32_t c2 = svreadz_hor_za32_f32(2, (uint32_t)i);
            svfloat32_t c3 = svreadz_hor_za32_f32(3, (uint32_t)i);
            if (accumulate) {
                c0 = svadd_f32_x(pg, c0, svld1(pg, row));
                c1 = svadd_f32_x(pg, c1, svld1(pg, row + 16));
                c2 = svadd_f32_x(pg, c2, svld1(pg, row + 32));
                c3 = svadd_f32_x(pg, c3, svld1(pg, row + 48));
            }
            svst1(pg, row, c0);
            svst1(pg, row + 16, c1);
            svst1(pg, row + 32, c2);
            svst1(pg, row + 48, c3);
        }
    }
}

__arm_new("za") __arm_locally_streaming __attribute__((target("sme,sme2,sme2p1")))
static void cob_sgemm_16x64_sme_from_packed_b64(
    int b_panels64,
    int k,
    const float* a_panel,
    const float* packed_b64,
    float* c,
    int ldc,
    int accumulate)
{
    const svbool_t pg = svptrue_b32();
    const size_t b_panel_floats = (size_t)k * 64u;

    for (int bp64 = 0; bp64 < b_panels64; ++bp64) {
        const int col0 = bp64 * 64;
        const float* packed_panel = packed_b64 + (size_t)bp64 * b_panel_floats;
        float* ct = c + (size_t)col0;

        svzero_za();
        for (int p = 0; p < k; ++p) {
            const svfloat32_t av =
                svld1(pg, a_panel + (size_t)p * (size_t)COB_SGEMM_AMX_MR);
            const float* pbrow = packed_panel + (size_t)p * 64u;
            const svfloat32_t bv0 = svld1(pg, pbrow);
            const svfloat32_t bv1 = svld1(pg, pbrow + 16);
            const svfloat32_t bv2 = svld1(pg, pbrow + 32);
            const svfloat32_t bv3 = svld1(pg, pbrow + 48);
            svmopa_za32_f32_m(0, pg, pg, av, bv0);
            svmopa_za32_f32_m(1, pg, pg, av, bv1);
            svmopa_za32_f32_m(2, pg, pg, av, bv2);
            svmopa_za32_f32_m(3, pg, pg, av, bv3);
        }

        for (int i = 0; i < 16; ++i) {
            float* row = ct + (size_t)i * (size_t)ldc;
            svfloat32_t c0 = svreadz_hor_za32_f32(0, (uint32_t)i);
            svfloat32_t c1 = svreadz_hor_za32_f32(1, (uint32_t)i);
            svfloat32_t c2 = svreadz_hor_za32_f32(2, (uint32_t)i);
            svfloat32_t c3 = svreadz_hor_za32_f32(3, (uint32_t)i);
            if (accumulate) {
                c0 = svadd_f32_x(pg, c0, svld1(pg, row));
                c1 = svadd_f32_x(pg, c1, svld1(pg, row + 16));
                c2 = svadd_f32_x(pg, c2, svld1(pg, row + 32));
                c3 = svadd_f32_x(pg, c3, svld1(pg, row + 48));
            }
            svst1(pg, row, c0);
            svst1(pg, row + 16, c1);
            svst1(pg, row + 32, c2);
            svst1(pg, row + 48, c3);
        }
    }
}

__arm_new("za") __arm_locally_streaming __attribute__((target("sme,sme2,sme2p1")))
static void cob_sgemm_16x64_sme_strided_b_pack_b32_tuple(
    int b_panels64,
    int k,
    const float* a_panel,
    const float* b,
    int ldb,
    float* packed_b64,
    float* c,
    int ldc,
    int accumulate)
{
    const svbool_t pg = svptrue_b32();
    const svcount_t pg4 = svptrue_c32();
    const size_t b_panel_floats = (size_t)k * 64u;

    for (int bp64 = 0; bp64 < b_panels64; ++bp64) {
        const int col0 = bp64 * 64;
        float* ct = c + (size_t)col0;
        float* packed_panel = packed_b64 + (size_t)bp64 * b_panel_floats;

        svzero_za();
        for (int p = 0; p < k; ++p) {
            const svfloat32_t av =
                svld1(pg, a_panel + (size_t)p * (size_t)COB_SGEMM_AMX_MR);
            const float* brow = b + (size_t)p * (size_t)ldb + (size_t)col0;
            float* pbrow = packed_panel + (size_t)p * 64u;
            const svfloat32x4_t bv = svld1_x4(pg4, brow);
            const svfloat32_t bv0 = svget4(bv, 0);
            const svfloat32_t bv1 = svget4(bv, 1);
            const svfloat32_t bv2 = svget4(bv, 2);
            const svfloat32_t bv3 = svget4(bv, 3);
            svmopa_za32_f32_m(0, pg, pg, av, bv0);
            svmopa_za32_f32_m(1, pg, pg, av, bv1);
            svmopa_za32_f32_m(2, pg, pg, av, bv2);
            svmopa_za32_f32_m(3, pg, pg, av, bv3);
            svst1(pg4, pbrow, bv);
        }

        for (int i = 0; i < 16; ++i) {
            float* row = ct + (size_t)i * (size_t)ldc;
            svfloat32_t c0 = svreadz_hor_za32_f32(0, (uint32_t)i);
            svfloat32_t c1 = svreadz_hor_za32_f32(1, (uint32_t)i);
            svfloat32_t c2 = svreadz_hor_za32_f32(2, (uint32_t)i);
            svfloat32_t c3 = svreadz_hor_za32_f32(3, (uint32_t)i);
            if (accumulate) {
                c0 = svadd_f32_x(pg, c0, svld1(pg, row));
                c1 = svadd_f32_x(pg, c1, svld1(pg, row + 16));
                c2 = svadd_f32_x(pg, c2, svld1(pg, row + 32));
                c3 = svadd_f32_x(pg, c3, svld1(pg, row + 48));
            }
            svst1(pg, row, c0);
            svst1(pg, row + 16, c1);
            svst1(pg, row + 32, c2);
            svst1(pg, row + 48, c3);
        }
    }
}

__arm_new("za") __arm_locally_streaming __attribute__((target("sme,sme2,sme2p1")))
static void cob_sgemm_16x64_sme_strided_b_pack_b32_tuple_prefetch2(
    int b_panels64,
    int k,
    const float* a_panel,
    const float* b,
    int ldb,
    float* packed_b64,
    float* c,
    int ldc,
    int accumulate)
{
    const svbool_t pg = svptrue_b32();
    const svcount_t pg4 = svptrue_c32();
    const size_t b_panel_floats = (size_t)k * 64u;

    for (int bp64 = 0; bp64 < b_panels64; ++bp64) {
        const int col0 = bp64 * 64;
        float* ct = c + (size_t)col0;
        float* packed_panel = packed_b64 + (size_t)bp64 * b_panel_floats;

        svzero_za();
        for (int p = 0; p < k; ++p) {
            const svfloat32_t av =
                svld1(pg, a_panel + (size_t)p * (size_t)COB_SGEMM_AMX_MR);
            const float* brow = b + (size_t)p * (size_t)ldb + (size_t)col0;
            float* pbrow = packed_panel + (size_t)p * 64u;
            if (p + COB_SGEMM_M64_SME_B_PREFETCH_DISTANCE < k) {
                const float* pf =
                    b + (size_t)(p + COB_SGEMM_M64_SME_B_PREFETCH_DISTANCE) *
                        (size_t)ldb + (size_t)col0;
                __builtin_prefetch(pf, 0, 1);
                __builtin_prefetch(pf + 32, 0, 1);
            }
            const svfloat32x4_t bv = svld1_x4(pg4, brow);
            const svfloat32_t bv0 = svget4(bv, 0);
            const svfloat32_t bv1 = svget4(bv, 1);
            const svfloat32_t bv2 = svget4(bv, 2);
            const svfloat32_t bv3 = svget4(bv, 3);
            svmopa_za32_f32_m(0, pg, pg, av, bv0);
            svmopa_za32_f32_m(1, pg, pg, av, bv1);
            svmopa_za32_f32_m(2, pg, pg, av, bv2);
            svmopa_za32_f32_m(3, pg, pg, av, bv3);
            svst1(pg4, pbrow, bv);
        }

        for (int i = 0; i < 16; ++i) {
            float* row = ct + (size_t)i * (size_t)ldc;
            svfloat32_t c0 = svreadz_hor_za32_f32(0, (uint32_t)i);
            svfloat32_t c1 = svreadz_hor_za32_f32(1, (uint32_t)i);
            svfloat32_t c2 = svreadz_hor_za32_f32(2, (uint32_t)i);
            svfloat32_t c3 = svreadz_hor_za32_f32(3, (uint32_t)i);
            if (accumulate) {
                c0 = svadd_f32_x(pg, c0, svld1(pg, row));
                c1 = svadd_f32_x(pg, c1, svld1(pg, row + 16));
                c2 = svadd_f32_x(pg, c2, svld1(pg, row + 32));
                c3 = svadd_f32_x(pg, c3, svld1(pg, row + 48));
            }
            svst1(pg, row, c0);
            svst1(pg, row + 16, c1);
            svst1(pg, row + 32, c2);
            svst1(pg, row + 48, c3);
        }
    }
}

__arm_new("za") __arm_locally_streaming __attribute__((target("sme,sme2,sme2p1")))
static void cob_sgemm_16x64_sme_from_packed_b64_tuple(
    int b_panels64,
    int k,
    const float* a_panel,
    const float* packed_b64,
    float* c,
    int ldc,
    int accumulate)
{
    const svbool_t pg = svptrue_b32();
    const svcount_t pg4 = svptrue_c32();
    const size_t b_panel_floats = (size_t)k * 64u;

    for (int bp64 = 0; bp64 < b_panels64; ++bp64) {
        const int col0 = bp64 * 64;
        const float* packed_panel = packed_b64 + (size_t)bp64 * b_panel_floats;
        float* ct = c + (size_t)col0;

        svzero_za();
        for (int p = 0; p < k; ++p) {
            const svfloat32_t av =
                svld1(pg, a_panel + (size_t)p * (size_t)COB_SGEMM_AMX_MR);
            const float* pbrow = packed_panel + (size_t)p * 64u;
            const svfloat32x4_t bv = svld1_x4(pg4, pbrow);
            const svfloat32_t bv0 = svget4(bv, 0);
            const svfloat32_t bv1 = svget4(bv, 1);
            const svfloat32_t bv2 = svget4(bv, 2);
            const svfloat32_t bv3 = svget4(bv, 3);
            svmopa_za32_f32_m(0, pg, pg, av, bv0);
            svmopa_za32_f32_m(1, pg, pg, av, bv1);
            svmopa_za32_f32_m(2, pg, pg, av, bv2);
            svmopa_za32_f32_m(3, pg, pg, av, bv3);
        }

        for (int i = 0; i < 16; ++i) {
            float* row = ct + (size_t)i * (size_t)ldc;
            svfloat32_t c0 = svreadz_hor_za32_f32(0, (uint32_t)i);
            svfloat32_t c1 = svreadz_hor_za32_f32(1, (uint32_t)i);
            svfloat32_t c2 = svreadz_hor_za32_f32(2, (uint32_t)i);
            svfloat32_t c3 = svreadz_hor_za32_f32(3, (uint32_t)i);
            if (accumulate) {
                c0 = svadd_f32_x(pg, c0, svld1(pg, row));
                c1 = svadd_f32_x(pg, c1, svld1(pg, row + 16));
                c2 = svadd_f32_x(pg, c2, svld1(pg, row + 32));
                c3 = svadd_f32_x(pg, c3, svld1(pg, row + 48));
            }
            svst1(pg, row, c0);
            svst1(pg, row + 16, c1);
            svst1(pg, row + 32, c2);
            svst1(pg, row + 48, c3);
        }
    }
}

static int cob_sgemm_rowmajor_sme_skinny_pack_b_reuse(
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
    const int use_m64 = m == 64;
    const int use_m96_128_k512 =
        (m == 96 || m == 128) && n >= COB_SGEMM_M96_128_SME_REUSE_K512_MIN_N && k == 512;
    const int use_m96_128_k1024 =
        (m == 96 || m == 128) && n >= COB_SGEMM_M96_128_SME_REUSE_K1024_MIN_N && k >= 1024;
    const int use_long_n_k512 =
        (use_m64 && n >= COB_SGEMM_M64_SME_LONG_N_K512_MIN_N && k == 512) ||
        use_m96_128_k512;
    const int use_n4096_large_k = use_m64 && n == 4096 && k >= 7168;
    const int use_wide = use_m64 && n >= 7168 && k >= 1024;
    if ((!use_m64 && !use_m96_128_k512 && !use_m96_128_k1024) ||
        (!use_long_n_k512 && !use_n4096_large_k && !use_wide && !use_m96_128_k1024) ||
        lda != k || ldb != n || (n % 64) != 0 || !cob_apple_sme2p1_available()) {
        return 0;
    }

    const int a32_panels = m / COB_SGEMM_AMX_MR;
    const int a16_panels = m / 16;
    const int nc_max =
        (use_wide && n >= 24576 && k == 1536) ?
            COB_SGEMM_M64_SME_LONG_WIDE_NC : COB_SGEMM_M64_SME_REUSE_NC;
    if (nc_max < 64 || (nc_max % 64) != 0) {
        return 0;
    }
    const int use_m96_128_large_kc = use_m96_128_k1024 && (k == 1024 || n >= 8192);
    const int use_large_kc = use_m96_128_large_kc || (use_m64 && use_wide && k == 1024);
    const int kc_max =
        use_long_n_k512 ? k :
        use_large_kc ? COB_SGEMM_SKINNY_SME_LARGE_KC :
        (use_wide && k == 1536) ? COB_SGEMM_M64_SME_WIDE_K1536_KC :
        (use_wide && k >= 8192) ? COB_SGEMM_M64_SME_WIDE_LARGE_KC :
        (use_wide && k >= 1536) ? COB_SGEMM_M64_SME_WIDE_KC : COB_SGEMM_SKINNY_SME_KC;
    const size_t a_panel_floats =
        (size_t)kc_max * (size_t)COB_SGEMM_AMX_MR;
    float* packed_a =
        (float*)cob_aligned_alloc(128, (size_t)a32_panels * a_panel_floats * sizeof(float));
    float* packed_b64 =
        (float*)cob_aligned_alloc(128, (size_t)nc_max * (size_t)kc_max * sizeof(float));
    if (packed_a == NULL || packed_b64 == NULL) {
        free(packed_a);
        free(packed_b64);
        return 0;
    }

    for (int pc = 0; pc < k; pc += kc_max) {
        const int kc = cob_min_i32(kc_max, k - pc);
        const size_t a_kc_panel_floats = (size_t)kc * (size_t)COB_SGEMM_AMX_MR;
        cob_amx_set();
        for (int ap = 0; ap < a32_panels; ++ap) {
            const int ic = ap * COB_SGEMM_AMX_MR;
            cob_sgemm_pack_a32_full(
                packed_a + (size_t)ap * a_kc_panel_floats,
                kc,
                a + (size_t)ic * (size_t)lda + pc,
                lda);
        }
        cob_amx_clr();

        for (int jc = 0; jc < n; jc += nc_max) {
            const int nc = cob_min_i32(nc_max, n - jc);
            const int b_panels64 = nc / 64;
            if (use_long_n_k512) {
                cob_sgemm_16x64_sme_strided_b_pack_b32(
                    b_panels64, kc, packed_a, b + (size_t)pc * (size_t)ldb + jc,
                    ldb, packed_b64, c + jc, ldc, pc != 0);
                for (int a16 = 1; a16 < a16_panels; ++a16) {
                    const int a32 = a16 / 2;
                    const int row = a16 * 16;
                    const float* ap =
                        packed_a + (size_t)a32 * a_kc_panel_floats + (size_t)(a16 & 1) * 16u;
                    cob_sgemm_16x64_sme_from_packed_b64(
                        b_panels64, kc, ap, packed_b64,
                        c + (size_t)row * (size_t)ldc + jc, ldc, pc != 0);
                }
            } else {
                if (use_n4096_large_k) {
                    cob_sgemm_16x64_sme_strided_b_pack_b32_tuple_prefetch2(
                        b_panels64, kc, packed_a, b + (size_t)pc * (size_t)ldb + jc,
                        ldb, packed_b64, c + jc, ldc, pc != 0);
                } else {
                    cob_sgemm_16x64_sme_strided_b_pack_b32_tuple(
                        b_panels64, kc, packed_a, b + (size_t)pc * (size_t)ldb + jc,
                        ldb, packed_b64, c + jc, ldc, pc != 0);
                }
                for (int a16 = 1; a16 < a16_panels; ++a16) {
                    const int a32 = a16 / 2;
                    const int row = a16 * 16;
                    const float* ap =
                        packed_a + (size_t)a32 * a_kc_panel_floats + (size_t)(a16 & 1) * 16u;
                    cob_sgemm_16x64_sme_from_packed_b64_tuple(
                        b_panels64, kc, ap, packed_b64,
                        c + (size_t)row * (size_t)ldc + jc, ldc, pc != 0);
                }
            }
        }
    }

    free(packed_a);
    free(packed_b64);
    return 1;
}

static int cob_sgemm_rowmajor_sme_from_packed_b32(
    int m,
    int n,
    int k,
    const float* a,
    int lda,
    const cob_packed_b_f32* packed_b,
    float* c,
    int ldc)
{
    if (packed_b->nr != COB_SGEMM_AMX_NR || m < 512 || n < 512 || k < 512 ||
        n > COB_SGEMM_SME_PACKED_MAX_N || n == 832 || n == 960 || n == 1088 ||
        (m % COB_SGEMM_AMX_MR) != 0 || (n % 64) != 0 ||
        !cob_apple_sme2p1_available()) {
        return 0;
    }

    const int max_a32_panels = COB_SGEMM_AMX_MC / COB_SGEMM_AMX_MR;
    const size_t a_panel_floats = (size_t)k * (size_t)COB_SGEMM_AMX_MR;
    float* packed_a_block =
        (float*)cob_aligned_alloc(128, (size_t)max_a32_panels * a_panel_floats * sizeof(float));
    if (packed_a_block == NULL) {
        return 0;
    }

    const int b_panels64 = n / 64;

    for (int ib = 0; ib < m; ib += COB_SGEMM_AMX_MC) {
        const int mc = cob_min_i32(COB_SGEMM_AMX_MC, m - ib);
        const int a32_panels = mc / COB_SGEMM_AMX_MR;
        cob_amx_set();
        for (int ap = 0; ap < a32_panels; ++ap) {
            const int ic = ib + ap * COB_SGEMM_AMX_MR;
            cob_sgemm_pack_a32_full(
                packed_a_block + (size_t)ap * a_panel_floats,
                k,
                a + (size_t)ic * (size_t)lda,
                lda);
        }
        cob_amx_clr();

        cob_sgemm_16x64_sme_packed_b32(
            a32_panels * 2,
            b_panels64,
            k,
            packed_a_block,
            packed_b->data,
            c + (size_t)ib * (size_t)ldc,
            ldc);
    }

    free(packed_a_block);
    return 1;
}

static int cob_sgemm_rowmajor_sme_medium_contiguous_strided_b32(
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
    const int use_square_384 = m == 384 && n == 384 && k == 384;
    const int use_square_768 = m == 768 && n == 768 && k == 768;
    const int use_extra_n = cob_sgemm_sme_direct_extra_n_shape(m, n, k);
    /* Tuned for medium contiguous cases where skipping one-shot B packing wins. */
    if ((!use_square_384 && !use_square_768 && !use_extra_n &&
            (m < 832 || m > COB_SGEMM_SME_DIRECT_MAX_N ||
                n < 832 || n > COB_SGEMM_SME_DIRECT_MAX_N ||
                k < 832 || k > COB_SGEMM_SME_DIRECT_MAX_N)) ||
        n == 1024 || lda != k || ldb != n ||
        (m % COB_SGEMM_AMX_MR) != 0 ||
        (n % 64) != 0 || !cob_apple_sme2p1_available()) {
        return 0;
    }

    const int b_panels64 = n / 64;
    const int max_a32_panels = COB_SGEMM_SME_DIRECT_MC / COB_SGEMM_AMX_MR;
    const size_t a_panel_floats = (size_t)k * (size_t)COB_SGEMM_AMX_MR;
    float* packed_a =
        (float*)cob_aligned_alloc(128, (size_t)max_a32_panels * a_panel_floats * sizeof(float));
    if (packed_a == NULL) {
        return 0;
    }

    for (int ib = 0; ib < m; ib += COB_SGEMM_SME_DIRECT_MC) {
        const int mc = cob_min_i32(COB_SGEMM_SME_DIRECT_MC, m - ib);
        const int a32_panels = mc / COB_SGEMM_AMX_MR;
        cob_amx_set();
        for (int ap = 0; ap < a32_panels; ++ap) {
            const int ic = ib + ap * COB_SGEMM_AMX_MR;
            cob_sgemm_pack_a32_full(
                packed_a + (size_t)ap * a_panel_floats,
                k,
                a + (size_t)ic * (size_t)lda,
                lda);
        }
        cob_amx_clr();

        cob_sgemm_16x64_sme_strided_b32(
            a32_panels * 2,
            b_panels64,
            k,
            packed_a,
            b,
            ldb,
            c + (size_t)ib * (size_t)ldc,
            ldc,
            0);
    }

    free(packed_a);
    return 1;
}

static int cob_sgemm_rowmajor_sme_skinny_contiguous_strided_b32(
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
    const int use_large_k_skinny =
        (n >= 1024 && n <= 4096 && k >= 2048) ||
        (n >= 1408 && n <= 4096 && k == 1536);
    const int use_long_n_k512 = n >= COB_SGEMM_M64_SME_LONG_N_K512_MIN_N && k == 512;
    const int use_prefetch = use_large_k_skinny && cob_sgemm_m64_sme_direct_prefetch_shape(n, k);
    if (m != 64 || (!use_large_k_skinny && !use_long_n_k512) ||
        lda != k || ldb != n ||
        (m % COB_SGEMM_AMX_MR) != 0 ||
        (n % 64) != 0 || !cob_apple_sme2p1_available()) {
        return 0;
    }

    const int b_panels64 = n / 64;
    const int a32_panels = m / COB_SGEMM_AMX_MR;
    const int kc_max =
        use_large_k_skinny && cob_sgemm_m64_sme_large_kc_shape(n, k) ?
            COB_SGEMM_SKINNY_SME_LARGE_KC : COB_SGEMM_SKINNY_SME_KC;
    const size_t a_panel_floats =
        (size_t)kc_max * (size_t)COB_SGEMM_AMX_MR;
    float* packed_a =
        (float*)cob_aligned_alloc(128, (size_t)a32_panels * a_panel_floats * sizeof(float));
    if (packed_a == NULL) {
        return 0;
    }

    for (int pc = 0; pc < k; pc += kc_max) {
        const int kc = cob_min_i32(kc_max, k - pc);
        const size_t a_kc_panel_floats = (size_t)kc * (size_t)COB_SGEMM_AMX_MR;
        cob_amx_set();
        for (int ap = 0; ap < a32_panels; ++ap) {
            const int ic = ap * COB_SGEMM_AMX_MR;
            cob_sgemm_pack_a32_full(
                packed_a + (size_t)ap * a_kc_panel_floats,
                kc,
                a + (size_t)ic * (size_t)lda + pc,
                lda);
        }
        cob_amx_clr();

        if (use_prefetch) {
            cob_sgemm_16x64_sme_strided_b32_prefetch2(
                a32_panels * 2,
                b_panels64,
                kc,
                packed_a,
                b + (size_t)pc * (size_t)ldb,
                ldb,
                c,
                ldc,
                pc != 0);
        } else {
            cob_sgemm_16x64_sme_strided_b32(
                a32_panels * 2,
                b_panels64,
                kc,
                packed_a,
                b + (size_t)pc * (size_t)ldb,
                ldb,
                c,
                ldc,
                pc != 0);
        }
    }

    free(packed_a);
    return 1;
}
#endif
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

#if defined(COB_USE_APPLE_AMX)
    if (nr_pack == COB_SGEMM_AMX_NR) {
        for (int panel = 0; panel < panels; ++panel) {
            const int col0 = panel * nr_pack;
            const int nr = cob_min_i32(nr_pack, n - col0);
            float* dst_panel = data + (size_t)panel * (size_t)k * (size_t)nr_pack;
            if (nr == COB_SGEMM_AMX_NR) {
                cob_sgemm_pack_b32_panel_full(dst_panel, k, b + col0, ldb);
            } else {
                cob_sgemm_pack_b32_panel_partial(dst_panel, k, b + col0, ldb, nr);
            }
        }

        packed->k = k;
        packed->n = n;
        packed->nr = nr_pack;
        packed->bytes = bytes;
        packed->data = data;
        return 0;
    }
#endif

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

    const int packed_mc = n >= 2048 ? COB_SGEMM_AMX_PACKED_LARGE_MC : COB_SGEMM_AMX_MC;
    if (m >= packed_mc && n >= 1152 && k >= 512 &&
        m % COB_SGEMM_AMX_MR == 0 && n % COB_SGEMM_AMX_NR == 0) {
        const int max_a_panels = packed_mc / COB_SGEMM_AMX_MR;
        const size_t a_block_floats = (size_t)max_a_panels * a_panel_floats;
        float* packed_a_block = (float*)cob_aligned_alloc(128, a_block_floats * sizeof(float));
        if (packed_a_block == NULL) {
            return 0;
        }

        cob_amx_set();
        for (int ib = 0; ib < m; ib += packed_mc) {
            const int mc = cob_min_i32(packed_mc, m - ib);
            const int a_panels = mc / COB_SGEMM_AMX_MR;
            for (int ap = 0; ap < a_panels; ++ap) {
                const int ic = ib + ap * COB_SGEMM_AMX_MR;
                cob_sgemm_pack_a32_full(
                    packed_a_block + (size_t)ap * a_panel_floats,
                    k,
                    a + (size_t)ic * (size_t)lda,
                    lda);
            }

            for (int panel = 0; panel < b_panels; ++panel) {
                const int jc = panel * COB_SGEMM_AMX_NR;
                const float* bp = packed_b->data + (size_t)panel * b_panel_floats;
                for (int ap = 0; ap < a_panels; ++ap) {
                    const int ic = ib + ap * COB_SGEMM_AMX_MR;
                    cob_sgemm_32x32_amx_packed_full(
                        k,
                        packed_a_block + (size_t)ap * a_panel_floats,
                        bp,
                        c + (size_t)ic * (size_t)ldc + jc,
                        ldc);
                }
            }
        }
        cob_amx_clr();

        free(packed_a_block);
        return 1;
    }

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

static void cob_sgemm_pack_b32_chunk_panels(
    float* packed,
    int k,
    int n,
    const float* b,
    int ldb)
{
    const int panels = (n + COB_SGEMM_AMX_NR - 1) / COB_SGEMM_AMX_NR;
    for (int panel = 0; panel < panels; ++panel) {
        const int col0 = panel * COB_SGEMM_AMX_NR;
        const int nr = cob_min_i32(COB_SGEMM_AMX_NR, n - col0);
        float* dst_panel = packed + (size_t)panel * (size_t)k * (size_t)COB_SGEMM_AMX_NR;
        if (nr == COB_SGEMM_AMX_NR) {
            cob_sgemm_pack_b32_panel_full(dst_panel, k, b + col0, ldb);
        } else {
            cob_sgemm_pack_b32_panel_partial(dst_panel, k, b + col0, ldb, nr);
        }
    }
}

static int cob_sgemm_rowmajor_amx_skinny_pack_b_chunks(
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
    if (m < 96 || m > 128 || n < 1024 || k < 512 ||
        lda != k || ldb != n ||
        (m % COB_SGEMM_AMX_MR) != 0 ||
        (n % COB_SGEMM_AMX_NR) != 0) {
        return 0;
    }

    const int skinny_nc = (m == 128 && k >= 1024) ? 512 : 256;
    const int a_panels = m / COB_SGEMM_AMX_MR;
    const int max_b_panels = skinny_nc / COB_SGEMM_AMX_NR;
    const size_t a_panel_floats = (size_t)k * (size_t)COB_SGEMM_AMX_MR;
    const size_t b_panel_floats = (size_t)k * (size_t)COB_SGEMM_AMX_NR;
    float* packed_a =
        (float*)cob_aligned_alloc(128, (size_t)a_panels * a_panel_floats * sizeof(float));
    float* packed_b =
        (float*)cob_aligned_alloc(128, (size_t)max_b_panels * b_panel_floats * sizeof(float));
    if (packed_a == NULL || packed_b == NULL) {
        free(packed_a);
        free(packed_b);
        return 0;
    }

    cob_amx_set();
    for (int ap = 0; ap < a_panels; ++ap) {
        const int ic = ap * COB_SGEMM_AMX_MR;
        cob_sgemm_pack_a32_full(
            packed_a + (size_t)ap * a_panel_floats,
            k,
            a + (size_t)ic * (size_t)lda,
            lda);
    }

    for (int jc = 0; jc < n; jc += skinny_nc) {
        const int nc = cob_min_i32(skinny_nc, n - jc);
        const int b_panels = nc / COB_SGEMM_AMX_NR;
        cob_sgemm_pack_b32_chunk_panels(packed_b, k, nc, b + jc, ldb);

        for (int panel = 0; panel < b_panels; ++panel) {
            const int col = jc + panel * COB_SGEMM_AMX_NR;
            const float* bp = packed_b + (size_t)panel * b_panel_floats;
            for (int ap = 0; ap < a_panels; ++ap) {
                const int ic = ap * COB_SGEMM_AMX_MR;
                cob_sgemm_32x32_amx_packed_full(
                    k,
                    packed_a + (size_t)ap * a_panel_floats,
                    bp,
                    c + (size_t)ic * (size_t)ldc + col,
                    ldc);
            }
        }
    }
    cob_amx_clr();

    free(packed_a);
    free(packed_b);
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
        lda >= k && ldc >= n) {
#if defined(COB_USE_APPLE_SME)
        if (cob_sgemm_rowmajor_sme_from_packed_b32(m, n, k, a, lda, packed_b, c, ldc)) {
            return;
        }
#endif
        if (cob_sgemm_rowmajor_amx_from_packed_b32(m, n, k, a, lda, packed_b, c, ldc) ||
            cob_sgemm_rowmajor_amx_from_packed_b8(m, n, k, a, lda, packed_b, c, ldc)) {
            return;
        }
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
    const int use_large_block =
        m >= COB_SGEMM_AMX_MC && n >= 1152 && k >= 512 &&
        m % COB_SGEMM_AMX_MR == 0 && n % COB_SGEMM_AMX_NR == 0;
    const int use_strided_b_large_extra =
        n == COB_SGEMM_AMX_STRIDED_B_EXTRA_N3 || n == COB_SGEMM_AMX_STRIDED_B_EXTRA_N4;
    const int use_strided_b_skinny_extra = m <= 128 && n >= 1024;
    const int use_strided_b_extra =
        n <= COB_SGEMM_AMX_STRIDED_B_MAX_N || n == COB_SGEMM_AMX_STRIDED_B_EXTRA_N ||
        n == COB_SGEMM_AMX_STRIDED_B_EXTRA_N2 || use_strided_b_large_extra ||
        use_strided_b_skinny_extra;
    const int use_strided_b =
        (!use_large_block || use_strided_b_large_extra) && use_strided_b_extra &&
        ldb != COB_SGEMM_AMX_STRIDED_B_CONFLICT_LDB &&
        m % COB_SGEMM_AMX_MR == 0 && n % COB_SGEMM_AMX_NR == 0;
    const int max_a_panels = COB_SGEMM_AMX_MC / COB_SGEMM_AMX_MR;
    const size_t a_block_floats = (size_t)max_a_panels * a_panel_floats;
    const size_t a_scratch_floats = use_large_block ? a_block_floats : a_panel_floats;
    const size_t scratch_bytes = b_bytes + a_scratch_floats * sizeof(float);

#if defined(COB_USE_APPLE_SME)
    if (cob_sgemm_rowmajor_sme_skinny_pack_b_reuse(m, n, k, a, lda, b, ldb, c, ldc)) {
        return 1;
    }
#endif

    if (cob_sgemm_rowmajor_amx_skinny_pack_b_chunks(m, n, k, a, lda, b, ldb, c, ldc)) {
        return 1;
    }

#if defined(COB_USE_APPLE_SME)
    if (cob_sgemm_rowmajor_sme_skinny_contiguous_strided_b32(m, n, k, a, lda, b, ldb, c, ldc)) {
        return 1;
    }

    if (((m == 384 && n == 384 && k == 384) || (m == 768 && n == 768 && k == 768)) &&
        cob_sgemm_rowmajor_sme_medium_contiguous_strided_b32(m, n, k, a, lda, b, ldb, c, ldc)) {
        return 1;
    }
#endif

    if (use_strided_b) {
        float* packed_a = (float*)cob_aligned_alloc(128, a_panel_floats * sizeof(float));
        if (packed_a == NULL) {
            return 0;
        }

        cob_amx_set();
        for (int ic = 0; ic < m; ic += COB_SGEMM_AMX_MR) {
            cob_sgemm_pack_a32_full(packed_a, k, a + (size_t)ic * (size_t)lda, lda);
            for (int panel = 0; panel < b_panels; ++panel) {
                const int jc = panel * COB_SGEMM_AMX_NR;
                cob_sgemm_32x32_amx_strided_b_full(
                    k,
                    packed_a,
                    b + jc,
                    ldb,
                    c + (size_t)ic * (size_t)ldc + jc,
                    ldc);
            }
        }
        cob_amx_clr();

        free(packed_a);
        return 1;
    }

#if defined(COB_USE_APPLE_SME)
    if (cob_sgemm_rowmajor_sme_medium_contiguous_strided_b32(m, n, k, a, lda, b, ldb, c, ldc)) {
        return 1;
    }
#endif

    float* scratch = (float*)cob_aligned_alloc(128, scratch_bytes);
    if (scratch == NULL) {
        return 0;
    }
    float* packed_b = scratch;
    float* packed_a_scratch = scratch + (size_t)b_panels * b_panel_floats;

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

#if defined(COB_USE_APPLE_SME)
    if (n == COB_SGEMM_AMX_STRIDED_B_CONFLICT_LDB &&
        m % COB_SGEMM_AMX_MR == 0 && k >= 512) {
        cob_packed_b_f32 packed_b_view;
        packed_b_view.k = k;
        packed_b_view.n = n;
        packed_b_view.nr = COB_SGEMM_AMX_NR;
        packed_b_view.bytes = b_bytes;
        packed_b_view.data = packed_b;
        if (cob_sgemm_rowmajor_sme_from_packed_b32(
                m, n, k, a, lda, &packed_b_view, c, ldc)) {
            free(scratch);
            return 1;
        }
    }
#endif

    if (use_large_block) {
        float* packed_a_block = packed_a_scratch;
        cob_amx_set();
        for (int ib = 0; ib < m; ib += COB_SGEMM_AMX_MC) {
            const int mc = cob_min_i32(COB_SGEMM_AMX_MC, m - ib);
            const int a_panels = mc / COB_SGEMM_AMX_MR;
            for (int ap = 0; ap < a_panels; ++ap) {
                const int ic = ib + ap * COB_SGEMM_AMX_MR;
                cob_sgemm_pack_a32_full(
                    packed_a_block + (size_t)ap * a_panel_floats,
                    k,
                    a + (size_t)ic * (size_t)lda,
                    lda);
            }

            for (int panel = 0; panel < b_panels; ++panel) {
                const int jc = panel * COB_SGEMM_AMX_NR;
                const float* bp = packed_b + (size_t)panel * b_panel_floats;
                for (int ap = 0; ap < a_panels; ++ap) {
                    const int ic = ib + ap * COB_SGEMM_AMX_MR;
                    cob_sgemm_32x32_amx_packed_full(
                        k,
                        packed_a_block + (size_t)ap * a_panel_floats,
                        bp,
                        c + (size_t)ic * (size_t)ldc + jc,
                        ldc);
                }
            }
        }
        cob_amx_clr();

        free(scratch);
        return 1;
    }

    float* packed_a = packed_a_scratch;

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

        free(scratch);
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

    free(scratch);

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
