#if !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200112L
#endif

#include "cob_gemm.h"

#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__)) && !defined(COB_DISABLE_X86_SIMD)
#include <immintrin.h>
#define COB_USE_X86_SIMD 1
#endif


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

#ifndef COB_SGEMM_M64_SME_SUBPANEL
#define COB_SGEMM_M64_SME_SUBPANEL 1
#endif

#ifndef COB_SGEMM_M64_SME_SUBPANEL_KC
#define COB_SGEMM_M64_SME_SUBPANEL_KC 512
#endif

#if COB_SGEMM_M64_SME_SUBPANEL != 0 && COB_SGEMM_M64_SME_SUBPANEL != 1
#error "COB_SGEMM_M64_SME_SUBPANEL must be 0 or 1"
#endif

#if COB_SGEMM_M64_SME_SUBPANEL_KC <= 0
#error "COB_SGEMM_M64_SME_SUBPANEL_KC must be positive"
#endif

#ifndef COB_SGEMM_M512_SME_REUSE_NC
#define COB_SGEMM_M512_SME_REUSE_NC 256
#endif

#ifndef COB_SGEMM_M512_SME_REUSE_KC
#define COB_SGEMM_M512_SME_REUSE_KC 1024
#endif

#ifndef COB_SGEMM_M768_2048_SME_REUSE_NC
#define COB_SGEMM_M768_2048_SME_REUSE_NC 256
#endif

#ifndef COB_SGEMM_M768_2048_SME_REUSE_KC
#define COB_SGEMM_M768_2048_SME_REUSE_KC 1024
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

#ifndef COB_SGEMM_M64_SME_PACK_B_PREFETCH_DISTANCE
#define COB_SGEMM_M64_SME_PACK_B_PREFETCH_DISTANCE 16
#endif

#ifndef COB_SGEMM_M64_SME_DIRECT_NC
#define COB_SGEMM_M64_SME_DIRECT_NC 1024
#endif

#ifndef COB_SGEMM_STRASSEN1
#define COB_SGEMM_STRASSEN1 1
#endif

#if COB_SGEMM_STRASSEN1 != 0 && COB_SGEMM_STRASSEN1 != 1
#error "COB_SGEMM_STRASSEN1 must be 0 or 1"
#endif

/* Each native Linux backend can select its own skinny-row ceiling and
   rectangular crossover independently. */
#if defined(COB_USE_X86_SIMD)
#define COB_DEFAULT_SKINNY_MAX_ROWS 8
#define COB_DEFAULT_RECTANGULAR_STRASSEN 1
#elif defined(COB_USE_NEON) && defined(__aarch64__) && !defined(COB_USE_APPLE_AMX)
#define COB_DEFAULT_SKINNY_MAX_ROWS 8
#define COB_DEFAULT_RECTANGULAR_STRASSEN 1
#else
#define COB_DEFAULT_SKINNY_MAX_ROWS 0
#define COB_DEFAULT_RECTANGULAR_STRASSEN 0
#endif

#ifndef COB_SKINNY_GEMM_MAX_ROWS
#define COB_SKINNY_GEMM_MAX_ROWS COB_DEFAULT_SKINNY_MAX_ROWS
#endif
#if COB_SKINNY_GEMM_MAX_ROWS != 0 && (COB_SKINNY_GEMM_MAX_ROWS < 2 || COB_SKINNY_GEMM_MAX_ROWS > 8)
#error "COB_SKINNY_GEMM_MAX_ROWS must be zero or in [2, 8]"
#endif
#ifndef COB_ENABLE_SKINNY_GEMM
#define COB_ENABLE_SKINNY_GEMM (COB_SKINNY_GEMM_MAX_ROWS >= 2)
#endif
#ifndef COB_ENABLE_RECTANGULAR_STRASSEN
#define COB_ENABLE_RECTANGULAR_STRASSEN COB_DEFAULT_RECTANGULAR_STRASSEN
#endif
#if COB_ENABLE_SKINNY_GEMM && (defined(COB_USE_X86_SIMD) || \
    (defined(COB_USE_NEON) && defined(__aarch64__)))
#define COB_USE_SKINNY_GEMM 1
#endif

/* The second level is an opt-in Linux experiment for native exact squares. */
#ifndef COB_ENABLE_RECURSIVE_STRASSEN
#define COB_ENABLE_RECURSIVE_STRASSEN 0
#endif

/* Experimental one-level schedule for native 64-bit x86 and Arm kernels. */
#if defined(COB_ENABLE_PORTABLE_STRASSEN) && COB_ENABLE_PORTABLE_STRASSEN && \
    !defined(COB_USE_APPLE_AMX) && (defined(COB_USE_X86_SIMD) || \
    (defined(COB_USE_NEON) && defined(__aarch64__)))
#define COB_USE_PORTABLE_STRASSEN 1
#ifndef COB_PORTABLE_STRASSEN_MIN_DIM
#define COB_PORTABLE_STRASSEN_MIN_DIM 1536
#endif
#if COB_PORTABLE_STRASSEN_MIN_DIM <= 0
#error "COB_PORTABLE_STRASSEN_MIN_DIM must be positive"
#endif
#endif

#ifndef COB_SGEMM_STRASSEN1_MIN_DIM
#define COB_SGEMM_STRASSEN1_MIN_DIM 6144
#define COB_SGEMM_STRASSEN1_MIN_DIM_DEFAULTED 1
#endif

#if COB_SGEMM_STRASSEN1_MIN_DIM <= 0
#error "COB_SGEMM_STRASSEN1_MIN_DIM must be positive"
#endif

#ifndef COB_SGEMM_STRASSEN1_SQUARE_MIN_DIM
#if defined(COB_SGEMM_STRASSEN1_MIN_DIM_DEFAULTED)
#define COB_SGEMM_STRASSEN1_SQUARE_MIN_DIM 5632
#else
#define COB_SGEMM_STRASSEN1_SQUARE_MIN_DIM COB_SGEMM_STRASSEN1_MIN_DIM
#endif
#endif

#if COB_SGEMM_STRASSEN1_SQUARE_MIN_DIM <= 0
#error "COB_SGEMM_STRASSEN1_SQUARE_MIN_DIM must be positive"
#endif

#if defined(COB_ENABLE_HK15) && COB_ENABLE_HK15
#define COB_USE_HK15 1
static void cob_sgemm_rowmajor_base(int, int, int, const float*, int,
    const float*, int, float*, int);
#else
#define cob_sgemm_rowmajor_base cob_sgemm_rowmajor
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

static int cob_sgemm_pack_mr(void)
{
#if defined(COB_USE_APPLE_AMX)
    return COB_SGEMM_AMX_MR;
#else
    return COB_SGEMM_MR;
#endif
}

static int cob_sgemm_sme_direct_extra_n_shape(int m, int n, int k)
{
    if (m >= 160 && m <= 384 && (m % 32) == 0 && n >= 1280 && n <= 1472 &&
        k >= 768 && k <= 1152) {
        return 1;
    }
    if (m >= 160 && m <= 320 && (m % 32) == 0 && n == 1536 &&
        k >= 832 && k <= 1152) {
        return 1;
    }
    if (m >= 160 && m <= 256 && (m % 32) == 0 && n == 1600 &&
        k >= 768 && k <= 1152) {
        return 1;
    }
    if (m >= 416 && m <= 480 && (m % 32) == 0 && n >= 1280 && n <= 1472 &&
        k >= 768 && k <= 1152) {
        return 1;
    }
    if (m == 512 && n == 896 && k >= 832 && k <= 1536) {
        return 1;
    }
    if (m == 512 && n >= 1280 && n <= 1408 &&
        (k == 768 || k == 832 || k == 960)) {
        return 1;
    }
    if (m == 544 && n >= 1280 && n <= 1408 &&
        (k == 768 || k == 832 || k == 960)) {
        return 1;
    }
    if (m == 576 && n >= 1280 && n <= 1408 &&
        (k == 768 || k == 832 || k == 960)) {
        return 1;
    }
    if (m == 608) {
        return (n == 1280 || n == 1408) && (k == 768 || k == 832 || k == 960);
    }
    if (m == 640 && n >= 1280 && n <= 1408 &&
        (k == 768 || k == 832 || k == 960)) {
        return 1;
    }
    if (m == 672) {
        if (n == 1280) {
            return k == 768 || k == 832 || k == 960;
        }
        if (n == 1344) {
            return k == 768 || k == 832;
        }
        return n == 1408 && (k == 768 || k == 832 || k == 960);
    }
    if (m == 704 && n >= 1280 && n <= 1408 && k >= 768 && k <= 960) {
        return 1;
    }
    if (m == 736 && n >= 1280 && n <= 1408 &&
        (k == 768 || k == 832 || k == 960)) {
        return 1;
    }
    if (m == 512 && n == 1024 && k == 1536) {
        return 1;
    }
    if (m == 512 && n == 512 && k == 2048) {
        return 1;
    }
    if (m == 512 && n == 512 && k == 3072) {
        return 1;
    }
    if (m == 512 && n == 512 && k == 4096) {
        return 1;
    }
    if (m == 384 && n == 512 && k == 3072) {
        return 1;
    }
    if (m == 384 && n == 896 && k == 1536) {
        return 1;
    }
    if (m == 512 && n == 1280 && k == 1536) {
        return 1;
    }
    if (m == 384 && n == 1280 && k == 1024) {
        return 1;
    }
    if (m == 384 && n == 1280 && k == 1536) {
        return 1;
    }
    if (n < COB_SGEMM_SME_DIRECT_EXTRA_N_MIN ||
        n > COB_SGEMM_SME_DIRECT_EXTRA_N_MAX ||
        ((n - COB_SGEMM_SME_DIRECT_EXTRA_N_MIN) % 64) != 0) {
        return 0;
    }
    if (m == 768) {
        if (n == 1280) {
            return k == 768 || k == 832 || k == 960;
        }
        if (n == 1344 || n == 1408) {
            return k >= 768 && k <= 1152;
        }
        return n == 1472 && (k == 768 || k == 832);
    }
    if (m == 800) {
        if (n == 1280) {
            return k == 768 || k == 832 || k == 960;
        }
        if (n == 1344 || n == 1408) {
            return k >= 768 && k <= 1152;
        }
        return n == 1472 && (k == 768 || k == 832);
    }
    if (m == 992) {
        if (n == 1280) {
            return k == 832 || k == 960;
        }
        if (n == 1344) {
            return k >= 832 && k <= 1152;
        }
        return n == 1472 && (k == 832 || k == 960);
    }
    if (m == 1024) {
        if (n == 1280) {
            return k == 832 || k == 960;
        }
        if (n == 1344) {
            return k >= 832 && k <= 1152;
        }
        return n == 1472 && (k == 832 || k == 960);
    }
    if (m == 1056) {
        if (n == 1280) {
            return k == 832 || k == 960;
        }
        if (n == 1344) {
            return k >= 832 && k <= 1152;
        }
        return n == 1472 && (k == 832 || k == 960);
    }
    if (m == 1088) {
        if (n == 1280) {
            return k == 832 || k == 960;
        }
        if (n == 1344) {
            return k >= 832 && k <= 1152;
        }
        return n == 1472 && (k == 832 || k == 960);
    }
    if (m == 1120) {
        if (n == 1280) {
            return k == 832 || k == 960;
        }
        if (n == 1344) {
            return k >= 832 && k <= 1152;
        }
        return n == 1472 && k == 960;
    }
    if (m == 1152) {
        return n == 1280 && (k == 832 || k == 960);
    }
    if (m == 1184) {
        return n == 1280 && (k == 832 || k == 960);
    }
    if (k == 768 && m >= 832 && m <= 960 && !(m == 928 && n == 1280)) {
        return 1;
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

static int cob_sgemm_m64_sme_direct_nc_shape(int n, int k)
{
    if (n == 2560) {
        return k >= 12288;
    }
    if (n == 3072) {
        return k >= 4096;
    }
    return n >= 3584 && n < 4096 && k >= 7168;
}

static int cob_sgemm_m64_sme_wide_reuse_shape(int n, int k)
{
    if (k < 1024) {
        return 0;
    }
    if (n >= 4160) {
        return 1;
    }
    if (n == 4096) {
        return k < 7168;
    }
    if (n == 2560 && k == 3072) {
        return 1;
    }
    if (n >= 3072 && n < 4096 && k == 1024) {
        return 1;
    }
    if (n == 3584) {
        return k < 7168;
    }
    return 0;
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

static int cob_sgemm_amx_strided_b_prefers_packed_shape(int m, int n, int k)
{
    if (m == COB_SGEMM_AMX_MC) {
        return (n >= COB_SGEMM_AMX_STRIDED_B_EXTRA_N3 && k >= 4096) ||
            (n == COB_SGEMM_AMX_STRIDED_B_EXTRA_N3 && k >= 3072) ||
            (n == COB_SGEMM_AMX_STRIDED_B_EXTRA_N4 &&
                (k >= 3072 || (m >= 768 && k >= 2048)));
    }
    return m >= 512 &&
        (k >= 4096 ||
            (n == COB_SGEMM_AMX_STRIDED_B_EXTRA_N4 &&
                (k >= 3072 || (m >= 768 && k >= 2048))) ||
            (n == COB_SGEMM_AMX_STRIDED_B_EXTRA_N3 &&
                (k >= 3072 || (m >= 512 && k == 2048))));
}

static int cob_sgemm_amx_packed_b_mc(int m, int n, int k)
{
    if (n >= 2048 && m == COB_SGEMM_AMX_MC && k >= 1024) {
        return COB_SGEMM_AMX_MC;
    }
    if (m == COB_SGEMM_AMX_PACKED_LARGE_MC && n == 1280 && k == 2048) {
        return COB_SGEMM_AMX_PACKED_LARGE_MC;
    }
    return n >= 2048 ? COB_SGEMM_AMX_PACKED_LARGE_MC : COB_SGEMM_AMX_MC;
}

static int cob_sgemm_amx_large_block_shape(int m, int n, int k)
{
    return n >= 1152 ||
        (n >= 768 && k >= 3072) ||
        (n == 512 && ((m == 768 && k == 4096) || (m >= 1024 && k >= 4096)));
}

static int cob_sgemm_sme_packed_b_excluded_shape(int m, int n, int k)
{
    if (n == 832 || n == 960 || n == 1088) {
        return 1;
    }
    if (m >= 1024 && n == 512 && k == 3072) {
        return 1;
    }
    if (n == 768 && (k == 2048 || k == 3072)) {
        return 1;
    }
    if ((n == 1024 && k >= 3072) || (n == 1024 && k == 2048)) {
        return 1;
    }
    if (m == 512 && n == 1024 && k == 1536) {
        return 1;
    }
    if (k >= 4096) {
        return 1;
    }
    if ((n == 1152 && k >= 2048) || (n == 1152 && k == 1536)) {
        return 1;
    }
    return 0;
}

static int cob_sgemm_amx_one_shot_large_mc(int m, int n, int k)
{
    if (k >= 4096 && m >= 768 && n >= 512 && n <= 1280) {
        return 256;
    }
    if (k >= 3072 && m == 512 && n >= 768 && n <= 1280) {
        return 256;
    }
    if (k >= 3072 && m >= 1024 && n >= 1152 && n <= 1280) {
        return 256;
    }
    if (m >= 768 && n == 768 && k >= 3072) {
        return 256;
    }
    if (m == 768 && n == 1024 && k == 3072) {
        return 256;
    }
    return COB_SGEMM_AMX_MC;
}

static void cob_sgemm_pack_a_generic(
    float* packed,
    int mr_pack,
    int rows,
    int k,
    const float* a,
    int lda)
{
    for (int p = 0; p < k; ++p) {
        float* dst = packed + (size_t)p * (size_t)mr_pack;
        for (int i = 0; i < mr_pack; ++i) {
            dst[i] = i < rows ? a[(size_t)i * (size_t)lda + p] : 0.0f;
        }
    }
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
    static atomic_int cached = ATOMIC_VAR_INIT(-1);
    const int cached_value = atomic_load_explicit(&cached, memory_order_acquire);
    if (cached_value >= 0) {
        return cached_value;
    }

    int value = 0;
    size_t size = sizeof(value);
    const int detected =
        sysctlbyname("hw.optional.arm.FEAT_SME2p1", &value, &size, NULL, 0) == 0 &&
        value != 0 && __arm_has_sme() && cob_apple_sme_f32_lanes() == 16;
    int expected = -1;
    if (atomic_compare_exchange_strong_explicit(
            &cached,
            &expected,
            detected,
            memory_order_release,
            memory_order_acquire)) {
        return detected;
    }
    return expected;
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
            if (p + COB_SGEMM_M64_SME_PACK_B_PREFETCH_DISTANCE < k) {
                const float* pf =
                    b + (size_t)(p + COB_SGEMM_M64_SME_PACK_B_PREFETCH_DISTANCE) *
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

static int cob_sgemm_rowmajor_sme_pack_b_reuse(
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
    const int use_m512_medium = m == 512 && n == 1216 && k == 3072;
    const int use_medium_high_k_reuse =
        (((m == 768 || m == 1024 || m == 1280 || (m == 1536 && n == 512)) &&
             (n == 512 || n == 768) && k == 4096) ||
            (m >= 512 && m <= 896 && (m % COB_SGEMM_AMX_MR) == 0 &&
                n >= 1024 && n <= 1280 && k >= 3072));
    const int use_long_n_k512 =
        (use_m64 && n >= COB_SGEMM_M64_SME_LONG_N_K512_MIN_N && k == 512) ||
        use_m96_128_k512;
    const int use_n4096_large_k = use_m64 && n == 4096 && k >= 7168;
    const int use_wide = use_m64 && cob_sgemm_m64_sme_wide_reuse_shape(n, k);
    const int use_m64_subpanel = COB_SGEMM_M64_SME_SUBPANEL &&
        use_m64 && n >= 192 && n <= 512 && (n % 64) == 0 && k >= 3072 && ldb > n;
    const int use_k1536_midwide_prefetch =
        k == 1536 && n >= 5120 && n <= 7680;
    const int use_wide_prefetch_pack = use_m64_subpanel ||
        (use_wide && (k == 2048 || use_k1536_midwide_prefetch ||
            (n == 24576 && k == 1536) ||
            (n == 7168 && k >= 8192)));
    if ((!use_m64 && !use_m96_128_k512 && !use_m96_128_k1024 &&
            !use_m512_medium && !use_medium_high_k_reuse) ||
        (!use_long_n_k512 && !use_n4096_large_k && !use_wide && !use_m64_subpanel &&
            !use_m96_128_k1024 && !use_m512_medium && !use_medium_high_k_reuse) ||
        lda != k || (!use_m64_subpanel && ldb != n) || ldb < n ||
        (n % 64) != 0 || !cob_apple_sme2p1_available()) {
        return 0;
    }

    const int a32_panels = m / COB_SGEMM_AMX_MR;
    const int a16_panels = m / 16;
    const int nc_max =
        use_m512_medium ? COB_SGEMM_M512_SME_REUSE_NC :
        use_medium_high_k_reuse ? COB_SGEMM_M768_2048_SME_REUSE_NC :
        use_m64_subpanel ? n :
        (use_wide && n >= 24576 && k == 1536) ?
            COB_SGEMM_M64_SME_LONG_WIDE_NC : COB_SGEMM_M64_SME_REUSE_NC;
    if (nc_max < 64 || (nc_max % 64) != 0) {
        return 0;
    }
    const int use_m96_128_large_kc = use_m96_128_k1024 && (k == 1024 || n >= 8192);
    const int use_large_kc = use_m96_128_large_kc || (use_m64 && use_wide && k == 1024);
    const int kc_max =
        use_m64_subpanel ? COB_SGEMM_M64_SME_SUBPANEL_KC :
        use_long_n_k512 ? k :
        use_m512_medium ? COB_SGEMM_M512_SME_REUSE_KC :
        use_medium_high_k_reuse ? COB_SGEMM_M768_2048_SME_REUSE_KC :
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
                if (use_n4096_large_k || use_wide_prefetch_pack ||
                    use_m512_medium || use_medium_high_k_reuse) {
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
        n > COB_SGEMM_SME_PACKED_MAX_N ||
        cob_sgemm_sme_packed_b_excluded_shape(m, n, k) ||
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
        (n == 1024 && !use_extra_n) || lda != k || ldb != n ||
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
        (n == 1024 && k == 1536) ||
        (n >= 1408 && n <= 4096 && k == 1536);
    const int use_mid_n_k512 =
        n >= 2048 && n < COB_SGEMM_M64_SME_LONG_N_K512_MIN_N &&
        (n % 512) == 0;
    const int use_long_n_k512 =
        (use_mid_n_k512 || n >= COB_SGEMM_M64_SME_LONG_N_K512_MIN_N) && k == 512;
    const int use_prefetch = use_large_k_skinny && cob_sgemm_m64_sme_direct_prefetch_shape(n, k);
    if (m != 64 || (!use_large_k_skinny && !use_long_n_k512) ||
        lda != k || ldb != n ||
        (m % COB_SGEMM_AMX_MR) != 0 ||
        (n % 64) != 0 || !cob_apple_sme2p1_available()) {
        return 0;
    }

    const int a32_panels = m / COB_SGEMM_AMX_MR;
    const int kc_max =
        use_large_k_skinny && cob_sgemm_m64_sme_large_kc_shape(n, k) ?
            COB_SGEMM_SKINNY_SME_LARGE_KC : COB_SGEMM_SKINNY_SME_KC;
    const int use_direct_nc = use_large_k_skinny && cob_sgemm_m64_sme_direct_nc_shape(n, k);
    const int nc_max =
        (COB_SGEMM_M64_SME_DIRECT_NC >= 64 && use_direct_nc) ?
            COB_SGEMM_M64_SME_DIRECT_NC : n;
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

        for (int jc = 0; jc < n; jc += nc_max) {
            const int nc = cob_min_i32(nc_max, n - jc);
            const int b_panels64 = nc / 64;
            if (use_prefetch) {
                cob_sgemm_16x64_sme_strided_b32_prefetch2(
                    a32_panels * 2,
                    b_panels64,
                    kc,
                    packed_a,
                    b + (size_t)pc * (size_t)ldb + jc,
                    ldb,
                    c + jc,
                    ldc,
                    pc != 0);
            } else {
                cob_sgemm_16x64_sme_strided_b32(
                    a32_panels * 2,
                    b_panels64,
                    kc,
                    packed_a,
                    b + (size_t)pc * (size_t)ldb + jc,
                    ldb,
                    c + jc,
                    ldc,
                    pc != 0);
            }
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

int cob_sgemm_pack_a(
    cob_packed_a_f32* packed,
    int m,
    int k,
    const float* a,
    int lda)
{
    if (packed == NULL) {
        return -1;
    }

    packed->m = 0;
    packed->k = 0;
    packed->mr = cob_sgemm_pack_mr();
    packed->bytes = 0;
    packed->data = NULL;

    if (m <= 0 || k <= 0 || a == NULL || lda < k) {
        return -1;
    }

    const int mr_pack = cob_sgemm_pack_mr();
    const int panels = (m + mr_pack - 1) / mr_pack;
    const size_t count = (size_t)panels * (size_t)k * (size_t)mr_pack;
    const size_t bytes = count * sizeof(float);
    const size_t alignment = mr_pack == COB_SGEMM_AMX_MR ? 128 : 64;
    float* data = (float*)cob_aligned_alloc(alignment, bytes);
    if (data == NULL) {
        return -1;
    }

#if defined(COB_USE_APPLE_AMX)
    const int use_amx_pack = mr_pack == COB_SGEMM_AMX_MR;
    if (use_amx_pack) {
        cob_amx_set();
    }
#endif
    for (int panel = 0; panel < panels; ++panel) {
        const int row0 = panel * mr_pack;
        const int rows = cob_min_i32(mr_pack, m - row0);
        float* dst_panel = data + (size_t)panel * (size_t)k * (size_t)mr_pack;
#if defined(COB_USE_APPLE_AMX)
        if (mr_pack == COB_SGEMM_AMX_MR && rows == COB_SGEMM_AMX_MR) {
            cob_sgemm_pack_a32_full(dst_panel, k, a + (size_t)row0 * (size_t)lda, lda);
        } else if (mr_pack == COB_SGEMM_AMX_MR) {
            cob_sgemm_pack_a32_partial(dst_panel, k, a + (size_t)row0 * (size_t)lda, lda, rows);
        } else
#endif
        {
            cob_sgemm_pack_a_generic(
                dst_panel,
                mr_pack,
                rows,
                k,
                a + (size_t)row0 * (size_t)lda,
                lda);
        }
    }
#if defined(COB_USE_APPLE_AMX)
    if (use_amx_pack) {
        cob_amx_clr();
    }
#endif

    packed->m = m;
    packed->k = k;
    packed->mr = mr_pack;
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

void cob_sgemm_free_packed_a(cob_packed_a_f32* packed)
{
    if (packed == NULL) {
        return;
    }

    free(packed->data);
    packed->m = 0;
    packed->k = 0;
    packed->mr = cob_sgemm_pack_mr();
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

#if defined(COB_USE_X86_SIMD) || (defined(COB_USE_NEON) && !defined(COB_USE_APPLE_AMX)) || defined(COB_USE_HK15)
static int cob_f32_ranges_overlap(const float* x, size_t nx, const float* y, size_t ny)
{
    const uintptr_t px = (uintptr_t)x;
    const uintptr_t py = (uintptr_t)y;
    return px <= py ? py - px < nx * sizeof(float) : px - py < ny * sizeof(float);
}
#endif

#if defined(COB_USE_NEON)
#if !defined(COB_USE_APPLE_AMX)
/* Fuse four consecutive K updates while streaming each B row in order. */
static void cob_sgemm_vector_matrix(
    int n, int k, const float* a, const float* b, int ldb, float* restrict c)
{
    memset(c, 0, (size_t)n * sizeof(float));
    int p = 0;
    for (; k - p >= 4; p += 4) {
        const float* b0 = b + (size_t)p * ldb;
        const float* b1 = b0 + ldb;
        const float* b2 = b1 + ldb;
        const float* b3 = b2 + ldb;
        const float a0 = a[p], a1 = a[p + 1], a2 = a[p + 2], a3 = a[p + 3];
        for (int j = 0; j < n; ++j) {
            float sum = c[j];
            sum += a0 * b0[j];
            sum += a1 * b1[j];
            sum += a2 * b2[j];
            sum += a3 * b3[j];
            c[j] = sum;
        }
    }
    for (; p < k; ++p) {
        const float* bp = b + (size_t)p * ldb;
        for (int j = 0; j < n; ++j) {
            c[j] += a[p] * bp[j];
        }
    }
}

#endif

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

#if !defined(COB_USE_NEON)
/* Each column is an independent FP32 dot product. Keeping the eight
   columns visible together lets the compiler use its native SIMD width. */
static void cob_sgemm_4x8_packed_b(int k, const float* a, int lda,
    const float* packed_b, float* c, int ldc)
{
    float c0[8] = {0}, c1[8] = {0}, c2[8] = {0}, c3[8] = {0};
    const float* a1 = a + lda;
    const float* a2 = a1 + lda;
    const float* a3 = a2 + lda;
    for (int p = 0; p < k; ++p) {
        const float* bp = packed_b + (size_t)p * 8;
        for (int j = 0; j < 8; ++j) {
            const float b = bp[j];
            c0[j] += a[p] * b;
            c1[j] += a1[p] * b;
            c2[j] += a2[p] * b;
            c3[j] += a3[p] * b;
        }
    }
    memcpy(c, c0, sizeof(c0));
    memcpy(c + ldc, c1, sizeof(c1));
    memcpy(c + (size_t)2 * ldc, c2, sizeof(c2));
    memcpy(c + (size_t)3 * ldc, c3, sizeof(c3));
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

    /* Keep a block of A rows in cache while traversing B's column panels. */
#if defined(COB_USE_APPLE_AMX)
    const int row_block = m;
#else
    const int row_block = 64;
#endif
    for (int row0 = 0; row0 < m;) {
        const int row_end = row0 + cob_min_i32(row_block, m - row0);
        for (int jc = 0; jc < n; jc += packed_nr) {
            const int nr = cob_min_i32(packed_nr, n - jc);
            const int panel = jc / packed_nr;
            const float* bp = packed_b->data +
                (size_t)panel * (size_t)k * (size_t)packed_nr;

            for (int ic = row0; ic < row_end; ic += COB_SGEMM_MR) {
                const int mr = cob_min_i32(COB_SGEMM_MR, row_end - ic);
                const float* at = a + (size_t)ic * (size_t)lda;
                float* ct = c + (size_t)ic * (size_t)ldc + jc;

                if (packed_nr == COB_SGEMM_NR && mr == COB_SGEMM_MR && nr == COB_SGEMM_NR) {
#if defined(COB_USE_NEON)
                    cob_sgemm_8x8_neon_packed_b(k, at, lda, bp, ct, ldc);
#else
                    cob_sgemm_4x8_packed_b(k, at, lda, bp, ct, ldc);
                    cob_sgemm_4x8_packed_b(k, at + (size_t)4 * lda, lda,
                                         bp, ct + (size_t)4 * ldc, ldc);
#endif
                } else {
                    cob_sgemm_tile_scalar_packed_b(mr, nr, k, at, lda, bp, packed_nr, ct, ldc);
                }
            }
        }
        row0 = row_end;
    }
}

#if defined(COB_USE_NEON) && !defined(COB_USE_APPLE_AMX)
/* B is already zero-padded to eight columns. Reuse both NEON vectors for
   each group of up to four real input rows, then copy only valid columns. */
static void cob_sgemm_neon_bottom_tile(int rows, int cols, int k,
    const float* a, int lda, const float* packed_b, float* c, int ldc)
{
    for (int row = 0; row < rows; row += 4) {
        const int active = cob_min_i32(4, rows - row);
        const float* a0 = a + (size_t)row * lda;
        const float* a1 = active >= 2 ? a0 + lda : a0;
        const float* a2 = active >= 3 ? a1 + lda : a0;
        const float* a3 = active == 4 ? a2 + lda : a0;
        float32x4_t lo0 = vdupq_n_f32(0), hi0 = vdupq_n_f32(0);
        float32x4_t lo1 = vdupq_n_f32(0), hi1 = vdupq_n_f32(0);
        float32x4_t lo2 = vdupq_n_f32(0), hi2 = vdupq_n_f32(0);
        float32x4_t lo3 = vdupq_n_f32(0), hi3 = vdupq_n_f32(0);
        for (int p = 0; p < k; ++p) {
            const float* b = packed_b + (size_t)p * 8;
            const float32x4_t lo = vld1q_f32(b);
            const float32x4_t hi = vld1q_f32(b + 4);
            lo0 = vfmaq_n_f32(lo0, lo, a0[p]); hi0 = vfmaq_n_f32(hi0, hi, a0[p]);
            lo1 = vfmaq_n_f32(lo1, lo, a1[p]); hi1 = vfmaq_n_f32(hi1, hi, a1[p]);
            lo2 = vfmaq_n_f32(lo2, lo, a2[p]); hi2 = vfmaq_n_f32(hi2, hi, a2[p]);
            lo3 = vfmaq_n_f32(lo3, lo, a3[p]); hi3 = vfmaq_n_f32(hi3, hi, a3[p]);
        }
        float tile[4][8];
        vst1q_f32(tile[0], lo0); vst1q_f32(tile[0] + 4, hi0);
        vst1q_f32(tile[1], lo1); vst1q_f32(tile[1] + 4, hi1);
        vst1q_f32(tile[2], lo2); vst1q_f32(tile[2] + 4, hi2);
        vst1q_f32(tile[3], lo3); vst1q_f32(tile[3] + 4, hi3);
        for (int r = 0; r < active; ++r) {
            memcpy(c + (size_t)(row + r) * ldc, tile[r], (size_t)cols * sizeof(float));
        }
    }
}

static void cob_sgemm_neon_ragged(int m, int n, int k,
    const float* a, int lda, const cob_packed_b_f32* packed, float* c, int ldc)
{
    const int full_rows = m / 8 * 8;
    const int full_cols = n / 8 * 8;
    if (full_rows && full_cols) {
        cob_sgemm_rowmajor_packed_b_fallback(full_rows, full_cols, k, a, lda, packed, c, ldc);
    }
    if (full_cols < n) {
        const float* b = packed->data + (size_t)full_cols * k;
        for (int i = 0; i < full_rows; i += 8) {
            float tile[64];
            cob_sgemm_8x8_neon_packed_b(k, a + (size_t)i * lda, lda, b, tile, 8);
            for (int r = 0; r < 8; ++r) {
                memcpy(c + (size_t)(i + r) * ldc + full_cols, tile + 8 * r,
                    (size_t)(n - full_cols) * sizeof(float));
            }
        }
    }
    if (full_rows < m) {
        for (int j = 0; j < n; j += 8) {
            cob_sgemm_neon_bottom_tile(m - full_rows, cob_min_i32(8, n - j), k,
                a + (size_t)full_rows * lda, lda, packed->data + (size_t)j * k,
                c + (size_t)full_rows * ldc + j, ldc);
        }
    }
}
#endif

static void cob_sgemm_rowmajor_packed_ab_fallback(
    int m,
    int n,
    int k,
    const cob_packed_a_f32* packed_a,
    const cob_packed_b_f32* packed_b,
    float* c,
    int ldc)
{
    if (m <= 0 || n <= 0 || k <= 0 || packed_a == NULL || packed_b == NULL ||
        c == NULL || packed_a->data == NULL || packed_b->data == NULL ||
        packed_a->k != k || packed_b->k != k || packed_a->m < m ||
        packed_b->n < n || packed_a->mr <= 0 || packed_b->nr <= 0 || ldc < n) {
        return;
    }

    const int packed_mr = packed_a->mr;
    const int packed_nr = packed_b->nr;
    for (int ic = 0; ic < m; ic += packed_mr) {
        const int mr = cob_min_i32(packed_mr, m - ic);
        const int a_panel = ic / packed_mr;
        const float* ap = packed_a->data +
            (size_t)a_panel * (size_t)k * (size_t)packed_mr;
        for (int jc = 0; jc < n; jc += packed_nr) {
            const int nr = cob_min_i32(packed_nr, n - jc);
            const int b_panel = jc / packed_nr;
            const float* bp = packed_b->data +
                (size_t)b_panel * (size_t)k * (size_t)packed_nr;
            float* ct = c + (size_t)ic * (size_t)ldc + jc;

            for (int i = 0; i < mr; ++i) {
                for (int j = 0; j < nr; ++j) {
                    float sum = 0.0f;
                    for (int p = 0; p < k; ++p) {
                        sum += ap[(size_t)p * (size_t)packed_mr + i] *
                            bp[(size_t)p * (size_t)packed_nr + j];
                    }
                    ct[(size_t)i * (size_t)ldc + j] = sum;
                }
            }
        }
    }
}

#if defined(COB_USE_APPLE_AMX)
static int cob_sgemm_rowmajor_amx_from_packed_ab32(
    int m,
    int n,
    int k,
    const cob_packed_a_f32* packed_a,
    const cob_packed_b_f32* packed_b,
    float* c,
    int ldc)
{
    if (packed_a->mr != COB_SGEMM_AMX_MR || packed_b->nr != COB_SGEMM_AMX_NR) {
        return 0;
    }
    if (packed_a->k != k || packed_b->k != k || packed_a->m < m ||
        packed_b->n < n || ldc < n) {
        return 0;
    }

    const int a_panels = (m + COB_SGEMM_AMX_MR - 1) / COB_SGEMM_AMX_MR;
    const int b_panels = (n + COB_SGEMM_AMX_NR - 1) / COB_SGEMM_AMX_NR;
    const size_t a_panel_floats = (size_t)k * (size_t)COB_SGEMM_AMX_MR;
    const size_t b_panel_floats = (size_t)k * (size_t)COB_SGEMM_AMX_NR;

    cob_amx_set();
    if (m >= COB_SGEMM_AMX_MC && cob_sgemm_amx_large_block_shape(m, n, k) &&
        k >= 512 && m % COB_SGEMM_AMX_MR == 0 && n % COB_SGEMM_AMX_NR == 0) {
        const int packed_mc = cob_sgemm_amx_packed_b_mc(m, n, k);
        for (int ib = 0; ib < m; ib += packed_mc) {
            const int mc = cob_min_i32(packed_mc, m - ib);
            const int block_a_panels = mc / COB_SGEMM_AMX_MR;
            for (int bpanel = 0; bpanel < b_panels; ++bpanel) {
                const int jc = bpanel * COB_SGEMM_AMX_NR;
                const float* bp = packed_b->data + (size_t)bpanel * b_panel_floats;
                for (int ap = 0; ap < block_a_panels; ++ap) {
                    const int ic = ib + ap * COB_SGEMM_AMX_MR;
                    const int apanel = ic / COB_SGEMM_AMX_MR;
                    const float* aptr = packed_a->data + (size_t)apanel * a_panel_floats;
                    cob_sgemm_32x32_amx_packed_full(
                        k,
                        aptr,
                        bp,
                        c + (size_t)ic * (size_t)ldc + jc,
                        ldc);
                }
            }
        }
        cob_amx_clr();
        return 1;
    }

    if (m % COB_SGEMM_AMX_MR == 0 && n % COB_SGEMM_AMX_NR == 0) {
        for (int bpanel = 0; bpanel < b_panels; ++bpanel) {
            const int jc = bpanel * COB_SGEMM_AMX_NR;
            const float* bp = packed_b->data + (size_t)bpanel * b_panel_floats;
            for (int apanel = 0; apanel < a_panels; ++apanel) {
                const int ic = apanel * COB_SGEMM_AMX_MR;
                const float* ap = packed_a->data + (size_t)apanel * a_panel_floats;
                cob_sgemm_32x32_amx_packed_full(
                    k,
                    ap,
                    bp,
                    c + (size_t)ic * (size_t)ldc + jc,
                    ldc);
            }
        }
        cob_amx_clr();
        return 1;
    }

    for (int apanel = 0; apanel < a_panels; ++apanel) {
        const int ic = apanel * COB_SGEMM_AMX_MR;
        const int mr = cob_min_i32(COB_SGEMM_AMX_MR, m - ic);
        const float* ap = packed_a->data + (size_t)apanel * a_panel_floats;
        for (int bpanel = 0; bpanel < b_panels; ++bpanel) {
            const int jc = bpanel * COB_SGEMM_AMX_NR;
            const int nr = cob_min_i32(COB_SGEMM_AMX_NR, n - jc);
            const float* bp = packed_b->data + (size_t)bpanel * b_panel_floats;
            float* ct = c + (size_t)ic * (size_t)ldc + jc;
            if (mr == COB_SGEMM_AMX_MR && nr == COB_SGEMM_AMX_NR) {
                cob_sgemm_32x32_amx_packed_full(k, ap, bp, ct, ldc);
            } else {
                cob_sgemm_32x32_amx_packed_partial(k, ap, bp, ct, ldc, mr, nr);
            }
        }
    }
    cob_amx_clr();
    return 1;
}

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

    const int packed_mc = cob_sgemm_amx_packed_b_mc(m, n, k);
    if (m >= packed_mc &&
        cob_sgemm_amx_large_block_shape(m, n, k) &&
        k >= 512 &&
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

    const int use_small_bouter =
        m % COB_SGEMM_AMX_MR == 0 && n % COB_SGEMM_AMX_NR == 0 &&
        ((m == 64 && n >= 2048 && k >= 1536) ||
            (m >= 96 && m <= 128 && n >= 4096 && k >= 1024));
    const int scratch_a_panels = use_small_bouter ? m / COB_SGEMM_AMX_MR : 1;
    float* packed_a = (float*)cob_aligned_alloc(
        128,
        (size_t)scratch_a_panels * a_panel_floats * sizeof(float));
    if (packed_a == NULL) {
        return 0;
    }

    if (use_small_bouter) {
        cob_amx_set();
        for (int ap = 0; ap < scratch_a_panels; ++ap) {
            const int ic = ap * COB_SGEMM_AMX_MR;
            cob_sgemm_pack_a32_full(
                packed_a + (size_t)ap * a_panel_floats,
                k,
                a + (size_t)ic * (size_t)lda,
                lda);
        }

        for (int panel = 0; panel < b_panels; ++panel) {
            const int jc = panel * COB_SGEMM_AMX_NR;
            const float* bp = packed_b->data + (size_t)panel * b_panel_floats;
            for (int ap = 0; ap < scratch_a_panels; ++ap) {
                const int ic = ap * COB_SGEMM_AMX_MR;
                cob_sgemm_32x32_amx_packed_full(
                    k,
                    packed_a + (size_t)ap * a_panel_floats,
                    bp,
                    c + (size_t)ic * (size_t)ldc + jc,
                    ldc);
            }
        }
        cob_amx_clr();

        free(packed_a);
        return 1;
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

static int cob_sgemm_rowmajor_amx_source_b_panel_reuse(
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
    const int selected_shape =
        (n == 896 && (m == 384 || m == 512 || m == 768)) ||
        (n == 1152 && m == 512) ||
        (n == 1280 && (m == 384 || m == 512));
    if (!selected_shape || k != 2048 || lda != k || ldb != n ||
        (m % COB_SGEMM_AMX_MR) != 0 || (n % COB_SGEMM_AMX_NR) != 0) {
        return 0;
    }

    const int a_panels = m / COB_SGEMM_AMX_MR;
    const int b_panels = n / COB_SGEMM_AMX_NR;
    const size_t a_panel_floats = (size_t)k * (size_t)COB_SGEMM_AMX_MR;
    float* packed_a =
        (float*)cob_aligned_alloc(128, (size_t)a_panels * a_panel_floats * sizeof(float));
    if (packed_a == NULL) {
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

    for (int panel = 0; panel < b_panels; ++panel) {
        const int jc = panel * COB_SGEMM_AMX_NR;
        const float* bp = b + jc;
        for (int ap = 0; ap < a_panels; ++ap) {
            const int ic = ap * COB_SGEMM_AMX_MR;
            cob_sgemm_32x32_amx_strided_b_full(
                k,
                packed_a + (size_t)ap * a_panel_floats,
                bp,
                ldb,
                c + (size_t)ic * (size_t)ldc + jc,
                ldc);
        }
    }
    cob_amx_clr();

    free(packed_a);
    return 1;
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
    const int use_skinny = m >= 96 && m <= 128 && n >= 1024 && k >= 512;
    const int use_m512 =
        m == 512 && k == 2048 && (n == 896 || n == 1024 || n == 1152 || n == 1280);
    if ((!use_skinny && !use_m512) || lda != k || ldb != n ||
        (m % COB_SGEMM_AMX_MR) != 0 || (n % COB_SGEMM_AMX_NR) != 0) {
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

void cob_sgemm_rowmajor_packed_ab(
    int m,
    int n,
    int k,
    const cob_packed_a_f32* packed_a,
    const cob_packed_b_f32* packed_b,
    float* c,
    int ldc)
{
#if defined(COB_USE_APPLE_AMX)
    if (m > 0 && n > 0 && k > 0 && packed_a != NULL && packed_b != NULL && c != NULL &&
        packed_a->data != NULL && packed_b->data != NULL && packed_a->k == k &&
        packed_b->k == k && packed_a->m >= m && packed_b->n >= n && ldc >= n) {
        if (cob_sgemm_rowmajor_amx_from_packed_ab32(m, n, k, packed_a, packed_b, c, ldc)) {
            return;
        }
    }
#endif

    cob_sgemm_rowmajor_packed_ab_fallback(m, n, k, packed_a, packed_b, c, ldc);
}

#if defined(COB_USE_X86_SIMD)

/* 6 independent rows, each holding two 8-float accumulation vectors. */
__attribute__((target("avx2,fma")))
static void cob_sgemm_tile_avx2(int k, const float* a, const float* b,
    float* c, int ldc, int resume)
{
    __m256 left[6], right[6];
    for (int i = 0; i < 6; ++i) {
        const float* row = c + (size_t)i * ldc;
        left[i] = resume ? _mm256_loadu_ps(row) : _mm256_setzero_ps();
        right[i] = resume ? _mm256_loadu_ps(row + 8) : _mm256_setzero_ps();
    }
    for (int p = 0; p < k; ++p) {
        const __m256 b0 = _mm256_load_ps(b);
        const __m256 b1 = _mm256_load_ps(b + 8);
#pragma GCC unroll 6
        for (int i = 0; i < 6; ++i) {
            const __m256 av = _mm256_set1_ps(a[i]);
            left[i] = _mm256_fmadd_ps(av, b0, left[i]);
            right[i] = _mm256_fmadd_ps(av, b1, right[i]);
        }
        a += 6;
        b += 16;
    }
    for (int i = 0; i < 6; ++i) {
        float* row = c + (size_t)i * ldc;
        _mm256_storeu_ps(row, left[i]);
        _mm256_storeu_ps(row + 8, right[i]);
    }
}

/* 8 independent rows, each holding two 16-float accumulation vectors. */
__attribute__((target("avx512f,fma")))
static void cob_sgemm_tile_avx512(int k, const float* a, const float* b,
    float* c, int ldc, int resume)
{
    __m512 left[8], right[8];
    for (int i = 0; i < 8; ++i) {
        const float* row = c + (size_t)i * ldc;
        left[i] = resume ? _mm512_loadu_ps(row) : _mm512_setzero_ps();
        right[i] = resume ? _mm512_loadu_ps(row + 16) : _mm512_setzero_ps();
    }
    for (int p = 0; p < k; ++p) {
        const __m512 b0 = _mm512_load_ps(b);
        const __m512 b1 = _mm512_load_ps(b + 16);
#pragma GCC unroll 8
        for (int i = 0; i < 8; ++i) {
            const __m512 av = _mm512_set1_ps(a[i]);
            left[i] = _mm512_fmadd_ps(av, b0, left[i]);
            right[i] = _mm512_fmadd_ps(av, b1, right[i]);
        }
        a += 8;
        b += 32;
    }
    for (int i = 0; i < 8; ++i) {
        float* row = c + (size_t)i * ldc;
        _mm512_storeu_ps(row, left[i]);
        _mm512_storeu_ps(row + 16, right[i]);
    }
}
typedef void (*cob_sgemm_x86_tile)(int, const float*, const float*, float*, int, int);

/* Only the final row or column tile needs a temporary. No vector read or
   write may extend into a caller's row padding. */
static void cob_sgemm_x86_edge(cob_sgemm_x86_tile tile, int k,
    const float* a, const float* b, float* c, int ldc, int rows, int cols,
    int nr, int resume)
{
    _Alignas(64) float edge[8 * 32] = {0};
    if (resume) {
        for (int i = 0; i < rows; ++i) {
            memcpy(edge + (size_t)i * nr, c + (size_t)i * ldc, (size_t)cols * sizeof(float));
        }
    }
    tile(k, a, b, edge, nr, resume);
    for (int i = 0; i < rows; ++i) {
        memcpy(c + (size_t)i * ldc, edge + (size_t)i * nr, (size_t)cols * sizeof(float));
    }
}

static void cob_sgemm_x86_pack_b(float* dst, const float* b, int ldb,
    int k, int n, int nr)
{
    for (int j = 0; j < n; j += nr) {
        const int cols = cob_min_i32(nr, n - j);
        for (int p = 0; p < k; ++p) {
            memcpy(dst, b + (size_t)p * ldb + j, (size_t)cols * sizeof(float));
            if (cols < nr) {
                memset(dst + cols, 0, (size_t)(nr - cols) * sizeof(float));
            }
            dst += nr;
        }
    }
}

static void cob_sgemm_x86_pack_a(float* dst, const float* a, int lda,
    int m, int k, int mr)
{
    for (int i = 0; i < m; i += mr) {
        const int rows = cob_min_i32(mr, m - i);
        for (int p = 0; p < k; ++p) {
            for (int r = 0; r < rows; ++r) {
                dst[r] = a[(size_t)(i + r) * lda + p];
            }
            for (int r = rows; r < mr; ++r) {
                dst[r] = 0;
            }
            dst += mr;
        }
    }
}

/* One row needs no panel packing. Four B rows are streamed while each
   output vector stays in a register, in increasing K order. */
__attribute__((target("avx2,fma")))
static void cob_sgemm_row_avx2(int n, int k, const float* a, const float* b,
    int ldb, float* restrict c)
{
    memset(c, 0, (size_t)n * sizeof(float));
    int p = 0;
    for (; k - p >= 4; p += 4) {
        const float* b0 = b + (size_t)p * ldb;
        const float* b1 = b0 + ldb;
        const float* b2 = b1 + ldb;
        const float* b3 = b2 + ldb;
        const __m256 a0 = _mm256_set1_ps(a[p]);
        const __m256 a1 = _mm256_set1_ps(a[p + 1]);
        const __m256 a2 = _mm256_set1_ps(a[p + 2]);
        const __m256 a3 = _mm256_set1_ps(a[p + 3]);
        int j = 0;
#pragma GCC unroll 4
        for (; n - j >= 8; j += 8) {
            __m256 acc = _mm256_loadu_ps(c + j);
            acc = _mm256_fmadd_ps(a0, _mm256_loadu_ps(b0 + j), acc);
            acc = _mm256_fmadd_ps(a1, _mm256_loadu_ps(b1 + j), acc);
            acc = _mm256_fmadd_ps(a2, _mm256_loadu_ps(b2 + j), acc);
            acc = _mm256_fmadd_ps(a3, _mm256_loadu_ps(b3 + j), acc);
            _mm256_storeu_ps(c + j, acc);
        }
        for (; j < n; ++j) {
            float acc = __builtin_fmaf(a[p], b0[j], c[j]);
            acc = __builtin_fmaf(a[p + 1], b1[j], acc);
            acc = __builtin_fmaf(a[p + 2], b2[j], acc);
            c[j] = __builtin_fmaf(a[p + 3], b3[j], acc);
        }
    }
    for (; p < k; ++p) {
        const float* row = b + (size_t)p * ldb;
        const __m256 av = _mm256_set1_ps(a[p]);
        int j = 0;
        for (; n - j >= 8; j += 8) {
            const __m256 acc = _mm256_fmadd_ps(av, _mm256_loadu_ps(row + j), _mm256_loadu_ps(c + j));
            _mm256_storeu_ps(c + j, acc);
        }
        for (; j < n; ++j) {
            c[j] = __builtin_fmaf(a[p], row[j], c[j]);
        }
    }
}

static int cob_sgemm_one_shot_x86(int m, int n, int k, const float* a, int lda,
    const float* b, int ldb, float* c, int ldc)
{
    if ((m != 1 && m < 6) || n < 16 || k < 16 || !__builtin_cpu_supports("fma")) {
        return 0;
    }
    const size_t c_count = (size_t)(m - 1) * ldc + n;
    if (cob_f32_ranges_overlap(c, c_count, a, (size_t)(m - 1) * lda + k) ||
        cob_f32_ranges_overlap(c, c_count, b, (size_t)(k - 1) * ldb + n)) {
        return 0;
    }
    if (m == 1) {
        if (!__builtin_cpu_supports("avx2")) {
            return 0;
        }
        cob_sgemm_row_avx2(n, k, a, b, ldb, c);
        return 1;
    }
    int mr, nr, mc;
    cob_sgemm_x86_tile tile;
#if !defined(COB_DISABLE_X86_AVX512)
    if (__builtin_cpu_supports("avx512f")) {
        mr = 8; nr = 32; mc = 128;
        tile = cob_sgemm_tile_avx512;
    } else
#endif
    if (__builtin_cpu_supports("avx2")) {
        mr = 6; nr = 16; mc = 120;
        tile = cob_sgemm_tile_avx2;
    } else {
        return 0;
    }
    const int kc = cob_min_i32(k, 256);
    const int nc = cob_min_i32(n, 256);
    const int packed_n = (nc + nr - 1) / nr * nr;
    const int packed_m = (cob_min_i32(m, mc) + mr - 1) / mr * mr;
    float* pack = cob_aligned_alloc(64, (size_t)(packed_n + packed_m) * kc * sizeof(float));
    if (pack == NULL) {
        return 0;
    }
    float* pa = pack + (size_t)packed_n * kc;
    for (int j = 0; j < n;) {
        const int nb = cob_min_i32(nc, n - j);
        for (int p = 0; p < k;) {
            const int kb = cob_min_i32(kc, k - p);
            cob_sgemm_x86_pack_b(pack, b + (size_t)p * ldb + j, ldb, kb, nb, nr);
            for (int i = 0; i < m;) {
                const int mb = cob_min_i32(mc, m - i);
                cob_sgemm_x86_pack_a(pa, a + (size_t)i * lda + p, lda, mb, kb, mr);
                for (int col = 0; col < nb; col += nr) {
                    const float* bp = pack + (size_t)col * kb;
                    const int cols = cob_min_i32(nr, nb - col);
                    for (int row = 0; row < mb; row += mr) {
                        const int rows = cob_min_i32(mr, mb - row);
                        float* ct = c + (size_t)(i + row) * ldc + j + col;
                        const float* ap = pa + (size_t)row * kb;
                        if (rows == mr && cols == nr) {
                            tile(kb, ap, bp, ct, ldc, p != 0);
                        } else {
                            cob_sgemm_x86_edge(tile, kb, ap, bp, ct, ldc, rows, cols, nr, p != 0);
                        }
                    }
                }
                i += mb;
            }
            p += kb;
        }
        j += nb;
    }
    free(pack);
    return 1;
}

#endif

#if defined(COB_ENABLE_BLOCKED_LEAF) && COB_ENABLE_BLOCKED_LEAF && !defined(COB_USE_APPLE_AMX) && \
    (defined(COB_USE_X86_SIMD) || (defined(COB_USE_NEON) && defined(__aarch64__)))
#define COB_USE_BLOCKED_LEAF 1
#if defined(COB_ENABLE_X86_MEDIUM_ROUTE) && COB_ENABLE_X86_MEDIUM_ROUTE && defined(COB_USE_X86_SIMD)
#define COB_USE_SCOPED_X86_MEDIUM 1
#endif
#include "cob_blocked_leaf.inc"
#endif

#if defined(COB_ENABLE_MEDIUM_DIRECT_A) && COB_ENABLE_MEDIUM_DIRECT_A && defined(COB_USE_X86_SIMD)
#include "cob_medium_direct_a.inc"
#endif

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

#if defined(COB_ENABLE_MEDIUM_DIRECT_A) && COB_ENABLE_MEDIUM_DIRECT_A && defined(COB_USE_X86_SIMD)
    if (cob_try_medium_direct_a(m, n, k, a, lda, b, ldb, c, ldc)) return;
#endif

#if defined(COB_USE_BLOCKED_LEAF)
    if (cob_sgemm_one_shot_blocked(m, n, k, a, lda, b, ldb, c, ldc)) return;
#endif

#if defined(COB_USE_X86_SIMD)
    if (cob_sgemm_one_shot_x86(m, n, k, a, lda, b, ldb, c, ldc)) {
        return;
    }
#endif

    cob_packed_b_f32 packed_b;
    if (cob_sgemm_pack_b(&packed_b, k, n, b, ldb) != 0) {
        cob_sgemm_ref_rowmajor(m, n, k, a, lda, b, ldb, c, ldc);
        return;
    }

#if defined(COB_USE_NEON) && !defined(COB_USE_APPLE_AMX)
    if (packed_b.nr == 8 && (m % 8 != 0 || n % 8 != 0) &&
        !cob_f32_ranges_overlap(c, (size_t)(m - 1) * ldc + n, a, (size_t)(m - 1) * lda + k) &&
        !cob_f32_ranges_overlap(c, (size_t)(m - 1) * ldc + n, b, (size_t)(k - 1) * ldb + n)) {
        cob_sgemm_neon_ragged(m, n, k, a, lda, &packed_b, c, ldc);
        cob_sgemm_free_packed_b(&packed_b);
        return;
    }
#endif
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
        m >= COB_SGEMM_AMX_MC &&
        cob_sgemm_amx_large_block_shape(m, n, k) &&
        k >= 512 &&
        m % COB_SGEMM_AMX_MR == 0 && n % COB_SGEMM_AMX_NR == 0;
    const int use_strided_b_large_extra =
        n == COB_SGEMM_AMX_STRIDED_B_EXTRA_N3 || n == COB_SGEMM_AMX_STRIDED_B_EXTRA_N4;
    const int use_strided_b_skinny_extra = m <= 128 && n >= 1024;
    const int use_strided_b_extra =
        n <= COB_SGEMM_AMX_STRIDED_B_MAX_N || n == COB_SGEMM_AMX_STRIDED_B_EXTRA_N ||
        n == COB_SGEMM_AMX_STRIDED_B_EXTRA_N2 || use_strided_b_large_extra ||
        use_strided_b_skinny_extra;
    const int use_sme_before_strided_n1152 =
        n == COB_SGEMM_AMX_STRIDED_B_EXTRA_N3 && k == 832 && (m == 1024 || m == 1088);
    const int use_strided_b =
        (!use_large_block || use_strided_b_large_extra) && use_strided_b_extra &&
        !use_sme_before_strided_n1152 &&
        !cob_sgemm_amx_strided_b_prefers_packed_shape(m, n, k) &&
        ldb != COB_SGEMM_AMX_STRIDED_B_CONFLICT_LDB &&
        m % COB_SGEMM_AMX_MR == 0 && n % COB_SGEMM_AMX_NR == 0;
    const int large_mc = cob_sgemm_amx_one_shot_large_mc(m, n, k);
    const int max_a_panels = large_mc / COB_SGEMM_AMX_MR;
    const size_t a_block_floats = (size_t)max_a_panels * a_panel_floats;
    const size_t a_scratch_floats = use_large_block ? a_block_floats : a_panel_floats;
    const size_t scratch_bytes = b_bytes + a_scratch_floats * sizeof(float);

#if defined(COB_USE_APPLE_SME)
    if (cob_sgemm_rowmajor_sme_pack_b_reuse(m, n, k, a, lda, b, ldb, c, ldc)) {
        return 1;
    }
#endif

    if (cob_sgemm_rowmajor_amx_source_b_panel_reuse(m, n, k, a, lda, b, ldb, c, ldc)) {
        return 1;
    }

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
        !(m == 768 && k == 1536) &&
        m % COB_SGEMM_AMX_MR == 0 && k >= 512 && k < 2048) {
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
        for (int ib = 0; ib < m; ib += large_mc) {
            const int mc = cob_min_i32(large_mc, m - ib);
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

#endif /* COB_USE_APPLE_AMX */

#if COB_SGEMM_STRASSEN1 && (defined(COB_USE_APPLE_AMX) || defined(COB_USE_PORTABLE_STRASSEN))
static void cob_strassen_linear2(
    float* dst,
    int rows,
    int cols,
    const float* x,
    int ldx,
    const float* y,
    int ldy,
    int y_sign)
{
    for (int i = 0; i < rows; ++i) {
        const float* xr = x + (size_t)i * (size_t)ldx;
        const float* yr = y + (size_t)i * (size_t)ldy;
        float* dr = dst + (size_t)i * (size_t)cols;
        int j = 0;
#if defined(COB_USE_NEON)
        for (; j + 4 <= cols; j += 4) {
            const float32x4_t xv = vld1q_f32(xr + j);
            const float32x4_t yv = vld1q_f32(yr + j);
            vst1q_f32(dr + j, y_sign > 0 ? vaddq_f32(xv, yv) : vsubq_f32(xv, yv));
        }
#endif
        for (; j < cols; ++j) {
            dr[j] = y_sign > 0 ? xr[j] + yr[j] : xr[j] - yr[j];
        }
    }
}

/* Applies M1 to all four C quadrants in one pass. On entry the quadrants
   hold direct-written products: c11 = M4, c12 = M3, c21 = M2. The pass
   reads each product row once and evaluates in the same floating-point
   order as the previous one-merge-per-pass schedule, so the output is
   bit-identical to it. */
static void cob_strassen_merge_m1(
    float* c11,
    float* c12,
    float* c21,
    float* c22,
    int ldc,
    const float* product,
    int rows,
    int cols)
{
    for (int i = 0; i < rows; ++i) {
        const float* pr = product + (size_t)i * (size_t)cols;
        float* r11 = c11 + (size_t)i * (size_t)ldc;
        const float* r12 = c12 + (size_t)i * (size_t)ldc;
        float* r21 = c21 + (size_t)i * (size_t)ldc;
        float* r22 = c22 + (size_t)i * (size_t)ldc;
        int j = 0;
#if defined(COB_USE_NEON)
        for (; j + 4 <= cols; j += 4) {
            const float32x4_t m1 = vld1q_f32(pr + j);
            const float32x4_t m4 = vld1q_f32(r11 + j);
            const float32x4_t m3 = vld1q_f32(r12 + j);
            const float32x4_t m2 = vld1q_f32(r21 + j);
            vst1q_f32(r11 + j, vaddq_f32(m1, m4));
            vst1q_f32(r21 + j, vaddq_f32(m2, m4));
            vst1q_f32(r22 + j, vaddq_f32(vsubq_f32(m1, m2), m3));
        }
#endif
        for (; j < cols; ++j) {
            const float m1 = pr[j];
            const float m4 = r11[j];
            const float m3 = r12[j];
            const float m2 = r21[j];
            r11[j] = m1 + m4;
            r21[j] = m2 + m4;
            r22[j] = (m1 - m2) + m3;
        }
    }
}

/* Applies M5 to both destination quadrants in one pass over the product. */
static void cob_strassen_merge_m5(
    float* c11,
    float* c12,
    int ldc,
    const float* product,
    int rows,
    int cols)
{
    for (int i = 0; i < rows; ++i) {
        const float* pr = product + (size_t)i * (size_t)cols;
        float* r11 = c11 + (size_t)i * (size_t)ldc;
        float* r12 = c12 + (size_t)i * (size_t)ldc;
        int j = 0;
#if defined(COB_USE_NEON)
        for (; j + 4 <= cols; j += 4) {
            const float32x4_t m5 = vld1q_f32(pr + j);
            vst1q_f32(r11 + j, vsubq_f32(vld1q_f32(r11 + j), m5));
            vst1q_f32(r12 + j, vaddq_f32(vld1q_f32(r12 + j), m5));
        }
#endif
        for (; j < cols; ++j) {
            const float m5 = pr[j];
            r11[j] -= m5;
            r12[j] += m5;
        }
    }
}

static void cob_strassen_merge_product(
    float* c,
    int ldc,
    const float* product,
    int rows,
    int cols,
    int operation)
{
    for (int i = 0; i < rows; ++i) {
        float* cr = c + (size_t)i * (size_t)ldc;
        const float* pr = product + (size_t)i * (size_t)cols;
        int j = 0;
#if defined(COB_USE_NEON)
        for (; j + 4 <= cols; j += 4) {
            const float32x4_t pv = vld1q_f32(pr + j);
            if (operation == 0) {
                vst1q_f32(cr + j, pv);
            } else {
                const float32x4_t cv = vld1q_f32(cr + j);
                vst1q_f32(
                    cr + j,
                    operation > 0 ? vaddq_f32(cv, pv) : vsubq_f32(cv, pv));
            }
        }
#endif
        for (; j < cols; ++j) {
            if (operation == 0) {
                cr[j] = pr[j];
            } else if (operation > 0) {
                cr[j] += pr[j];
            } else {
                cr[j] -= pr[j];
            }
        }
    }
}

#if defined(COB_ENABLE_FUSED_STRASSEN) && COB_ENABLE_FUSED_STRASSEN && defined(COB_USE_BLOCKED_LEAF)
#define COB_USE_FUSED_STRASSEN 1
#include "cob_fused_strassen.inc"
#endif

#if defined(COB_USE_PORTABLE_STRASSEN) && defined(__linux__) && defined(COB_ENABLE_OPENBLAS_LEAF) && COB_ENABLE_OPENBLAS_LEAF
#define COB_USE_OPENBLAS_LEAF 1
#include <cblas.h>
#endif

static int cob_strassen_leaf(int m, int n, int k, const float* a, int lda,
    const float* b, int ldb, float* c, int ldc)
{
#if defined(COB_USE_APPLE_AMX)
    return cob_sgemm_rowmajor_amx(m, n, k, a, lda, b, ldb, c, ldc);
#else
    /* Bypass the public dispatcher. The ordinary one-shot native kernel
       owns its packing, including the existing allocation fallback. */
#if defined(COB_USE_OPENBLAS_LEAF)
    if (openblas_get_num_threads() == 1) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k,
            1.0f, a, lda, b, ldb, 0.0f, c, ldc);
        return 1;
    }
#endif
    cob_sgemm_rowmajor_fallback(m, n, k, a, lda, b, ldb, c, ldc);
    return 1;
#endif
}

#if COB_ENABLE_RECTANGULAR_STRASSEN && defined(COB_USE_PORTABLE_STRASSEN)
#define COB_USE_RECTANGULAR_STRASSEN 1
#ifndef COB_RECTANGULAR_MIN_HARMONIC_DIM
#define COB_RECTANGULAR_MIN_HARMONIC_DIM 1536
#endif
#if COB_RECTANGULAR_MIN_HARMONIC_DIM < 128
#error "COB_RECTANGULAR_MIN_HARMONIC_DIM must be at least 128"
#endif

/* Strassen saves work proportional to M*N*K and forms temporaries of size
   M*N + M*K + N*K. Their ratio scales with the harmonic dimension. */
static int cob_strassen_rectangle_is_eligible(int m, int n, int k)
{
    const int smallest = cob_min_i32(m, cob_min_i32(n, k));
    const int largest = m > n ? (m > k ? m : k) : (n > k ? n : k);
    if (smallest < COB_RECTANGULAR_MIN_HARMONIC_DIM / 2 ||
        (int64_t)largest > 8 * (int64_t)smallest || n < 128) return 0;

    const double areas = (double)m*n + (double)m*k + (double)n*k;
    /* Three half-product buffers in FP32 consume one byte per element of
       the three full matrix areas. Bound the newly enabled temporary. */
    return areas <= 256.0 * 1024 * 1024 &&
           3.0 * m * n * k >= COB_RECTANGULAR_MIN_HARMONIC_DIM * areas;
}

static void cob_strassen_complete_edges(int m, int n, int k,
    int core_m, int core_n, int core_k, const float* a, int lda,
    const float* b, int ldb, float* c, int ldc)
{
    if (k != core_k) {
        const float* last_b = b + (size_t)core_k * ldb;
        for (int i = 0; i < core_m; ++i) {
            const float last_a = a[(size_t)i * lda + core_k];
            float* row = c + (size_t)i * ldc;
            for (int j = 0; j < core_n; ++j) row[j] += last_a * last_b[j];
        }
    }
    /* Both edges use the complete K. Their narrow dimensions preclude
       redispatch to this seven-product decomposition. */
    if (n != core_n) {
        cob_sgemm_rowmajor_base(core_m, n - core_n, k, a, lda, b + core_n, ldb,
            c + core_n, ldc);
    }
    if (m != core_m) {
        cob_sgemm_rowmajor_base(1, n, k, a + (size_t)core_m * lda, lda, b, ldb,
            c + (size_t)core_m * ldc, ldc);
    }
}
#endif

#if COB_ENABLE_RECURSIVE_STRASSEN && defined(COB_USE_PORTABLE_STRASSEN) && defined(__linux__)
#define COB_USE_RECURSIVE_STRASSEN 1
#ifndef COB_RECURSIVE_MIN_DIM
#define COB_RECURSIVE_MIN_DIM 3072
#endif
#ifndef COB_RECURSIVE_MAX_DIM
#define COB_RECURSIVE_MAX_DIM 5120
#endif
#if COB_RECURSIVE_MIN_DIM < 128 || COB_RECURSIVE_MAX_DIM < COB_RECURSIVE_MIN_DIM
#error "Invalid recursive Strassen dimension bounds"
#endif

static int cob_recursive_square_is_eligible(int m, int n, int k)
{
    return m == n && n == k && m % 128 == 0 &&
           m >= COB_RECURSIVE_MIN_DIM && m <= COB_RECURSIVE_MAX_DIM;
}

typedef int (*cob_product_fn)(int, int, int, const float*, int,
    const float*, int, float*, int);

/* Winograd's seven-product identity uses eight operand additions and
   seven output additions. Reuse S1/T1 as S2/T2 and then as S4/T4.
   The outer Strassen level owns these three quarter-matrix temporaries. */
static int cob_winograd_product(int m, int n, int k, const float* a, int lda,
    const float* b, int ldb, float* c, int ldc, float* scratch, float* child_scratch,
    cob_product_fn leaf)
{
    if (!scratch) return leaf(m, n, k, a, lda, b, ldb, c, ldc);
    const int mr = m / 2, nc = n / 2, kr = k / 2;
    const float* a11 = a;
    const float* a12 = a + kr;
    const float* a21 = a + (size_t)mr * lda;
    const float* a22 = a21 + kr;
    const float* b11 = b;
    const float* b12 = b + nc;
    const float* b21 = b + (size_t)kr * ldb;
    const float* b22 = b21 + nc;
    float* c11 = c;
    float* c12 = c + nc;
    float* c21 = c + (size_t)mr * ldc;
    float* c22 = c21 + nc;
    float* s = scratch;
    float* t = s + (size_t)mr * kr;
    float* p = t + (size_t)kr * nc;

    /* A second level uses the same schedule and reuses one child workspace. */
#if defined(COB_USE_OPENBLAS_LEAF) && defined(COB_USE_RECTANGULAR_STRASSEN)
#define COB_WINOGRAD_LEAF(...) (child_scratch ? \
    cob_winograd_product(__VA_ARGS__, child_scratch, NULL, leaf) : leaf(__VA_ARGS__))
#else
    (void)child_scratch;
#define COB_WINOGRAD_LEAF leaf
#endif

    /* P1 and P2 go to C quadrants directly. */
    if (!COB_WINOGRAD_LEAF(mr, nc, kr, a11, lda, b11, ldb, c21, ldc) ||
        !COB_WINOGRAD_LEAF(mr, nc, kr, a12, lda, b21, ldb, c11, ldc)) return 0;
    cob_strassen_linear2(s, mr, kr, a21, lda, a22, lda, 1);
    cob_strassen_linear2(t, kr, nc, b12, ldb, b11, ldb, -1);
    if (!COB_WINOGRAD_LEAF(mr, nc, kr, s, kr, t, nc, c22, ldc)) return 0;
    cob_strassen_linear2(s, mr, kr, s, kr, a11, lda, -1);
    cob_strassen_linear2(t, kr, nc, b22, ldb, t, nc, -1);
    if (!COB_WINOGRAD_LEAF(mr, nc, kr, s, kr, t, nc, c12, ldc)) return 0;

    /* C contains P2, P6, P1, P5. Share U2=P1+P6 and U4=U2+P5. */
    for (int i = 0; i < mr; ++i) {
        for (int j = 0; j < nc; ++j) {
            const size_t z = (size_t)i * ldc + j;
            const float p1 = c21[z];
            const float u2 = p1 + c12[z];
            const float u4 = u2 + c22[z];
            c11[z] += p1;
            c12[z] = u4;
            c21[z] = u2;
            c22[z] = u4;
        }
    }
    cob_strassen_linear2(s, mr, kr, a12, lda, s, kr, -1);
    if (!COB_WINOGRAD_LEAF(mr, nc, kr, s, kr, b22, ldb, p, nc)) return 0;
    cob_strassen_merge_product(c12, ldc, p, mr, nc, 1);
    cob_strassen_linear2(t, kr, nc, t, nc, b21, ldb, -1);
    if (!COB_WINOGRAD_LEAF(mr, nc, kr, a22, lda, t, nc, p, nc)) return 0;
    cob_strassen_merge_product(c21, ldc, p, mr, nc, -1);
    cob_strassen_linear2(s, mr, kr, a11, lda, a21, lda, -1);
    cob_strassen_linear2(t, kr, nc, b22, ldb, b12, ldb, -1);
    if (!COB_WINOGRAD_LEAF(mr, nc, kr, s, kr, t, nc, p, nc)) return 0;
    for (int i = 0; i < mr; ++i) {
        for (int j = 0; j < nc; ++j) {
            const float p7 = p[(size_t)i * nc + j];
            const size_t z = (size_t)i * ldc + j;
            c21[z] += p7;
            c22[z] += p7;
        }
    }
#undef COB_WINOGRAD_LEAF
    return 1;
}
#if defined(COB_USE_OPENBLAS_LEAF) && defined(COB_USE_RECTANGULAR_STRASSEN)
#include "cob_openblas_winograd.inc"
#endif

#endif

static int cob_sgemm_rowmajor_strassen1(
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
    const int min_dim = cob_min_i32(cob_min_i32(m, n), k);
    const int max_dim = m > n ? (m > k ? m : k) : (n > k ? n : k);
    const int legacy_geometry = 3 * (int64_t)max_dim <= 4 * (int64_t)min_dim &&
        lda == k && ldb == n && m % 64 == 0 && n % 64 == 0 && k % 64 == 0;
#if defined(COB_USE_RECTANGULAR_STRASSEN)
    const int full_m = m, full_n = n, full_k = k;
#endif
    if (legacy_geometry) {
#if defined(COB_USE_PORTABLE_STRASSEN)
        if (min_dim < COB_PORTABLE_STRASSEN_MIN_DIM) return 0;
        const size_t c_span = (size_t)(m - 1) * ldc + n;
        if (cob_f32_ranges_overlap(c, c_span, a, (size_t)m * k) ||
            cob_f32_ranges_overlap(c, c_span, b, (size_t)k * n)) return 0;
#else
        const int lower_square = m == n && n == k &&
            min_dim >= COB_SGEMM_STRASSEN1_SQUARE_MIN_DIM;
        if (!lower_square && min_dim < COB_SGEMM_STRASSEN1_MIN_DIM) return 0;
#endif
    } else {
#if defined(COB_USE_RECTANGULAR_STRASSEN)
        if (lda < k || ldb < n || ldc < n || !cob_strassen_rectangle_is_eligible(m, n, k)) {
            return 0;
        }
        const size_t c_span = (size_t)(m - 1) * ldc + n;
        if (cob_f32_ranges_overlap(c, c_span, a, (size_t)(m - 1) * lda + k) ||
            cob_f32_ranges_overlap(c, c_span, b, (size_t)(k - 1) * ldb + n)) return 0;
        m -= m % 2;
        n -= n % 128;
        k -= k % 2;
#else
        return 0;
#endif
    }

    const int mh = m / 2;
    const int nh = n / 2;
    const int kh = k / 2;
    const size_t x_values = (size_t)mh * (size_t)kh;
    const size_t y_values = (size_t)kh * (size_t)nh;
    const size_t p_values = (size_t)mh * (size_t)nh;
    const size_t parent_values = x_values + y_values + p_values;
#if defined(COB_USE_RECURSIVE_STRASSEN)
    size_t child_values = legacy_geometry && cob_recursive_square_is_eligible(m, n, k)
        ? parent_values / 4 : 0;
    float* workspace = cob_aligned_alloc(128, (parent_values + child_values) * sizeof(float));
    if (workspace == NULL && child_values != 0) {
        child_values = 0;
        workspace = cob_aligned_alloc(128, parent_values * sizeof(float));
    }
    float* child = workspace && child_values ? workspace + parent_values : NULL;
#define COB_STRASSEN_CHILD(...) cob_winograd_product(__VA_ARGS__, child, NULL, cob_strassen_leaf)
#else
    float* workspace = cob_aligned_alloc(128, parent_values * sizeof(float));
#define COB_STRASSEN_CHILD cob_strassen_leaf
#endif
    if (workspace == NULL) {
        return 0;
    }
    float* x = workspace;
    float* y = x + x_values;
    float* p = y + y_values;

    const float* a11 = a;
    const float* a12 = a + kh;
    const float* a21 = a + (size_t)mh * (size_t)lda;
    const float* a22 = a21 + kh;
    const float* b11 = b;
    const float* b12 = b + nh;
    const float* b21 = b + (size_t)kh * (size_t)ldb;
    const float* b22 = b21 + nh;
    float* c11 = c;
    float* c12 = c + nh;
    float* c21 = c + (size_t)mh * (size_t)ldc;
    float* c22 = c21 + nh;

    /* One level reduces eight half-size products to seven. The native
       scheduler handles each product with three reusable half-matrix buffers. M2, M3, and M4 are written straight
       into their first destination quadrant instead of the product buffer,
       and the remaining recombination is fused so each product buffer pass
       is read once. Every quadrant keeps the previous schedule's
       floating-point evaluation order, so results stay bit-identical. */
    cob_strassen_linear2(x, mh, kh, a21, lda, a22, lda, 1);
    if (!COB_STRASSEN_CHILD(mh, nh, kh, x, kh, b11, ldb, c21, ldc)) {
        free(workspace);
        return 0;
    }

    cob_strassen_linear2(y, kh, nh, b12, ldb, b22, ldb, -1);
    if (!COB_STRASSEN_CHILD(mh, nh, kh, a11, lda, y, nh, c12, ldc)) {
        free(workspace);
        return 0;
    }

    cob_strassen_linear2(y, kh, nh, b21, ldb, b11, ldb, -1);
    if (!COB_STRASSEN_CHILD(mh, nh, kh, a22, lda, y, nh, c11, ldc)) {
        free(workspace);
        return 0;
    }

    cob_strassen_linear2(x, mh, kh, a11, lda, a22, lda, 1);
    cob_strassen_linear2(y, kh, nh, b11, ldb, b22, ldb, 1);
    if (!COB_STRASSEN_CHILD(mh, nh, kh, x, kh, y, nh, p, nh)) {
        free(workspace);
        return 0;
    }
    cob_strassen_merge_m1(c11, c12, c21, c22, ldc, p, mh, nh);

    cob_strassen_linear2(x, mh, kh, a11, lda, a12, lda, 1);
    if (!COB_STRASSEN_CHILD(mh, nh, kh, x, kh, b22, ldb, p, nh)) {
        free(workspace);
        return 0;
    }
    cob_strassen_merge_m5(c11, c12, ldc, p, mh, nh);

    cob_strassen_linear2(x, mh, kh, a21, lda, a11, lda, -1);
    cob_strassen_linear2(y, kh, nh, b11, ldb, b12, ldb, 1);
    if (!COB_STRASSEN_CHILD(mh, nh, kh, x, kh, y, nh, p, nh)) {
        free(workspace);
        return 0;
    }
    cob_strassen_merge_product(c22, ldc, p, mh, nh, 1);

    cob_strassen_linear2(x, mh, kh, a12, lda, a22, lda, -1);
    cob_strassen_linear2(y, kh, nh, b21, ldb, b22, ldb, 1);
    if (!COB_STRASSEN_CHILD(mh, nh, kh, x, kh, y, nh, p, nh)) {
        free(workspace);
        return 0;
    }
    cob_strassen_merge_product(c11, ldc, p, mh, nh, 1);

#undef COB_STRASSEN_CHILD
    free(workspace);
#if defined(COB_USE_RECTANGULAR_STRASSEN)
    if (!legacy_geometry) {
        cob_strassen_complete_edges(full_m, full_n, full_k, m, n, k, a, lda, b, ldb, c, ldc);
    }
#endif
    return 1;
}

#endif

#if defined(COB_USE_SKINNY_GEMM) && !defined(COB_USE_APPLE_AMX)
/* For a handful of output rows each B value has little reuse. Stream B in
   short K strips directly; all accumulators keep increasing K order. */
#if defined(COB_USE_X86_SIMD)
#define COB_THIN_TARGET __attribute__((target("avx2,fma")))
typedef __m256 cob_thin_vector;
enum { COB_THIN_LANES = 8 };
#define COB_THIN_LOAD(p) _mm256_loadu_ps(p)
#define COB_THIN_STORE(p, v) _mm256_storeu_ps((p), (v))
#define COB_THIN_ZERO() _mm256_setzero_ps()
#define COB_THIN_FMA(c, b, a) _mm256_fmadd_ps(_mm256_set1_ps(a), (b), (c))
#else
#define COB_THIN_TARGET
typedef float32x4_t cob_thin_vector;
enum { COB_THIN_LANES = 4 };
#define COB_THIN_LOAD(p) vld1q_f32(p)
#define COB_THIN_STORE(p, v) vst1q_f32((p), (v))
#define COB_THIN_ZERO() vdupq_n_f32(0)
#define COB_THIN_FMA(c, b, a) vfmaq_n_f32((c), (b), (a))
#endif
enum { COB_THIN_COLS = 16, COB_THIN_VECTORS = COB_THIN_COLS / COB_THIN_LANES };

COB_THIN_TARGET __attribute__((always_inline)) static inline void cob_thin_tile(
    int rows, int k, const float* a, int lda, const float* b, int ldb,
    float* c, int ldc, int first)
{
    cob_thin_vector accum[4][COB_THIN_VECTORS];
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < COB_THIN_VECTORS; ++j) {
            accum[i][j] = first ? COB_THIN_ZERO() :
                COB_THIN_LOAD(c + (size_t)i * ldc + j * COB_THIN_LANES);
        }
    }
    for (int p = 0; p < k; ++p) {
        const float* bp = b + (size_t)p * ldb;
        cob_thin_vector bv[COB_THIN_VECTORS];
        for (int j = 0; j < COB_THIN_VECTORS; ++j) {
            bv[j] = COB_THIN_LOAD(bp + j * COB_THIN_LANES);
        }
        for (int i = 0; i < rows; ++i) {
            const float av = a[(size_t)i * lda + p];
            for (int j = 0; j < COB_THIN_VECTORS; ++j) {
                accum[i][j] = COB_THIN_FMA(accum[i][j], bv[j], av);
            }
        }
    }
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < COB_THIN_VECTORS; ++j) {
            COB_THIN_STORE(c + (size_t)i * ldc + j * COB_THIN_LANES, accum[i][j]);
        }
    }
}

COB_THIN_TARGET static void cob_sgemm_few_rows(int m, int n, int k,
    const float* a, int lda, const float* b, int ldb, float* c, int ldc)
{
    const int full_cols = n / COB_THIN_COLS * COB_THIN_COLS;
    for (int p = 0; p < k;) {
        const int kb = cob_min_i32(16, k - p);
        for (int j = 0; j < full_cols; j += COB_THIN_COLS) {
            const float* bp = b + (size_t)p * ldb + j;
            for (int i = 0; i < m;) {
                const float* ap = a + (size_t)i * lda + p;
                float* cp = c + (size_t)i * ldc + j;
                const int remain = m - i;
                /* Split five as three plus two to avoid a single-row tail. */
                if (remain == 2) {
                    cob_thin_tile(2, kb, ap, lda, bp, ldb, cp, ldc, p == 0);
                    i += 2;
                } else if (remain == 3 || remain == 5) {
                    cob_thin_tile(3, kb, ap, lda, bp, ldb, cp, ldc, p == 0);
                    i += 3;
                } else {
                    cob_thin_tile(4, kb, ap, lda, bp, ldb, cp, ldc, p == 0);
                    i += 4;
                }
            }
        }
        p += kb;
    }
    /* B's final row need not have accessible storage after its last column. */
    for (int i = 0; i < m; ++i) {
        for (int j = full_cols; j < n; ++j) {
            float sum = 0;
            for (int p = 0; p < k; ++p) {
                sum = __builtin_fmaf(a[(size_t)i * lda + p], b[(size_t)p * ldb + j], sum);
            }
            c[(size_t)i * ldc + j] = sum;
        }
    }
}
#undef COB_THIN_TARGET
#undef COB_THIN_LOAD
#undef COB_THIN_STORE
#undef COB_THIN_ZERO
#undef COB_THIN_FMA
#endif

#include "cob_small_direct.inc"

#if defined(COB_ENABLE_SMALL_X86) && COB_ENABLE_SMALL_X86 && defined(COB_USE_X86_SIMD) && \
    ((defined(__AVX2__) && defined(__FMA__)) || (defined(__AVX512F__) && !defined(COB_DISABLE_X86_AVX512)))
#define COB_USE_SMALL_X86 1
#include "cob_small_x86.inc"
#endif

#if defined(COB_ENABLE_SHORT_AXES) && COB_ENABLE_SHORT_AXES && !defined(COB_USE_APPLE_AMX) && \
    (defined(COB_USE_X86_SIMD) || (defined(COB_USE_NEON) && defined(__aarch64__)))
#define COB_USE_SHORT_AXES 1
#include "cob_short_axes.inc"
#endif

#if defined(COB_ENABLE_FEW_ROW_PANEL) && COB_ENABLE_FEW_ROW_PANEL && !defined(COB_USE_APPLE_AMX) && \
    (defined(COB_USE_X86_SIMD) || (defined(COB_USE_NEON) && defined(__aarch64__)))
#define COB_USE_FEW_ROW_PANEL 1
#include "cob_few_row_panel.inc"
#endif

#if defined(COB_ENABLE_N2_SMALL) && COB_ENABLE_N2_SMALL && defined(__linux__) && \
    defined(__aarch64__) && defined(COB_USE_NEON) && defined(__ARM_NEON)
#define COB_USE_N2_SMALL 1
#include "cob_n2_small.inc"
#endif

#if defined(COB_USE_HK15) && defined(COB_USE_FUSED_STRASSEN) && \
    defined(COB_ENABLE_HK15_PARENT_960) && COB_ENABLE_HK15_PARENT_960
#define COB_USE_HK15_PARENT_960 1
#if defined(COB_HK15_CHILD_HARMONIC)
#error "Select the parent policy or the first-stage child cutoff separately"
#endif
#endif
#if defined(COB_USE_HK15) && defined(COB_USE_FUSED_STRASSEN) && defined(COB_HK15_CHILD_TRACE) && COB_HK15_CHILD_TRACE
#define COB_TRACE_HK15_CHILD 1
static _Thread_local int cob_hk15_children, cob_hk15_child_fused, cob_hk15_other_fused;

#if defined(COB_USE_HK15_PARENT_960)
static _Thread_local int cob_hk15_trace_inside, cob_hk15_lower_children;
#endif

int rank_hk15_route_count(int id)
{
    switch (id) {
        case 0: return cob_hk15_children;
        case 1: return cob_hk15_child_fused;
        case 2: return cob_hk15_children - cob_hk15_child_fused;
        case 3: return cob_hk15_other_fused;
#if defined(COB_USE_HK15_PARENT_960)
        case 4: return cob_hk15_lower_children;
#else
        case 4: return 0;
#endif
        default: return -1;
    }
}
#endif
#if defined(COB_USE_HK15) && defined(COB_USE_FUSED_STRASSEN) && !defined(COB_USE_HK15_PARENT_960) && \
    (defined(COB_HK15_CHILD_HARMONIC) || (defined(COB_HK15_CHILD_TRACE) && COB_HK15_CHILD_TRACE))
#define COB_USE_HK15_CHILD 1
#ifndef COB_HK15_CHILD_HARMONIC
#define COB_HK15_CHILD_HARMONIC COB_FUSED_STRASSEN_MIN_HARMONIC_DIM
#endif
#if COB_HK15_CHILD_HARMONIC <= 0
#error "COB_HK15_CHILD_HARMONIC must be positive"
#endif
static void cob_sgemm_dispatch(int hk_child,
#elif defined(COB_USE_HK_X86_LEAF)
static void cob_hk_x86_base(int hk_fused,
#else
#ifdef COB_USE_HK15
static
#endif
void cob_sgemm_rowmajor_base(
#endif
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
#if defined(COB_USE_OPENBLAS_LEAF) && defined(COB_USE_RECTANGULAR_STRASSEN) && defined(COB_USE_RECURSIVE_STRASSEN)
    if (cob_openblas_winograd_try(m, n, k, a, lda, b, ldb, c, ldc, 2048)) return;
#endif
#ifdef COB_USE_N2_SMALL
    if (cob_n2_try_sgemm(m, n, k, a, lda, b, ldb, c, ldc)) return;
#endif
#ifdef COB_USE_FEW_ROW_PANEL
    if (cob_try_panel_few_rows(m, n, k, a, lda, b, ldb, c, ldc)) return;
#endif
#ifdef COB_USE_SMALL_X86
    if (cob_try_small_x86(m, n, k, a, lda, b, ldb, c, ldc)) return;
#endif
#ifdef COB_USE_SHORT_AXES
    if (cob_try_short_axes(m, n, k, a, lda, b, ldb, c, ldc)) return;
#endif
#if defined(COB_USE_DIRECT_SMALL_GEMM)
    if (m >= 4 && m <= 128 && n >= 16 && n <= 128 && k > 0 && k <= 128 &&
        m % 4 == 0 && n % 16 == 0 && a && b && c && lda >= k && ldb >= n && ldc >= n &&
#if defined(COB_USE_X86_SIMD)
        __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") &&
#endif
        !cob_f32_ranges_overlap(c, (size_t)(m - 1) * ldc + n, a, (size_t)(m - 1) * lda + k) &&
        !cob_f32_ranges_overlap(c, (size_t)(m - 1) * ldc + n, b, (size_t)(k - 1) * ldb + n)) {
        cob_sgemm_small_direct(m, n, k, a, lda, b, ldb, c, ldc);
        return;
    }
#endif
#if defined(COB_USE_SKINNY_GEMM) && !defined(COB_USE_APPLE_AMX)
    if (m >= 2 && m <= COB_SKINNY_GEMM_MAX_ROWS && n >= 256 && k >= 64 &&
        a && b && c && lda >= k && ldb >= n && ldc >= n &&
#if defined(COB_USE_X86_SIMD)
        __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") &&
        /* Let a complete row tile use the ordinary packed SIMD kernel. */
#if !defined(COB_DISABLE_X86_AVX512)
        (m < 6 || (m < 8 && __builtin_cpu_supports("avx512f"))) &&
#else
        m < 6 &&
#endif
#endif
        !cob_f32_ranges_overlap(c, (size_t)(m - 1) * ldc + n, a, (size_t)(m - 1) * lda + k) &&
        !cob_f32_ranges_overlap(c, (size_t)(m - 1) * ldc + n, b, (size_t)(k - 1) * ldb + n)) {
        cob_sgemm_few_rows(m, n, k, a, lda, b, ldb, c, ldc);
        return;
    }
#endif

#if defined(COB_USE_APPLE_AMX)
    if (m > 0 && n > 0 && k > 0 && a != NULL && b != NULL && c != NULL &&
        lda >= k && ldb >= n && ldc >= n &&
#if COB_SGEMM_STRASSEN1
        (cob_sgemm_rowmajor_strassen1(m, n, k, a, lda, b, ldb, c, ldc) ||
#endif
        cob_sgemm_rowmajor_amx(m, n, k, a, lda, b, ldb, c, ldc)
#if COB_SGEMM_STRASSEN1
        )
#endif
        ) {
        return;
    }
#endif

#if defined(COB_USE_NEON) && !defined(COB_USE_APPLE_AMX)
    if (m == 1 && n > 0 && k > 0 && a != NULL && b != NULL && c != NULL &&
        lda >= k && ldb >= n && ldc >= n &&
        !cob_f32_ranges_overlap(c, (size_t)n, a, (size_t)k) &&
        !cob_f32_ranges_overlap(c, (size_t)n, b, (size_t)(k - 1) * ldb + n)) {
        cob_sgemm_vector_matrix(n, k, a, b, ldb, c);
        return;
    }
#endif

#if defined(COB_USE_FUSED_STRASSEN)
    if (
#if defined(COB_USE_RECURSIVE_STRASSEN)
        !(lda == k && ldb == n && cob_recursive_square_is_eligible(m, n, k)) &&
#endif
#if defined(COB_USE_HK15_CHILD)
        cob_sgemm_strassen_fused_at(m, n, k, a, lda, b, ldb, c, ldc,
            hk_child ? COB_HK15_CHILD_HARMONIC : COB_FUSED_STRASSEN_MIN_HARMONIC_DIM)) {
#ifdef COB_TRACE_HK15_CHILD
        if (hk_child) ++cob_hk15_child_fused;
        else ++cob_hk15_other_fused;
#endif
        return;
    }
#else
#if defined(COB_USE_HK_X86_LEAF)
        cob_sgemm_strassen_fused_impl(m, n, k, a, lda, b, ldb, c, ldc,
            COB_FUSED_STRASSEN_MIN_HARMONIC_DIM, hk_fused)) {
#else
        cob_sgemm_strassen_fused(m, n, k, a, lda, b, ldb, c, ldc)) {
#endif
#if defined(COB_USE_HK15_PARENT_960) && defined(COB_TRACE_HK15_CHILD)
        if (cob_hk15_trace_inside) ++cob_hk15_child_fused;
        else ++cob_hk15_other_fused;
#endif
        return;
    }
#endif
#endif

#if COB_SGEMM_STRASSEN1 && defined(COB_USE_PORTABLE_STRASSEN)
    if (m > 0 && n > 0 && k > 0 && a != NULL && b != NULL && c != NULL && ldc >= n &&
        cob_sgemm_rowmajor_strassen1(m, n, k, a, lda, b, ldb, c, ldc)) {
        return;
    }
#endif

#if defined(COB_USE_SCOPED_X86_MEDIUM)
    if (m >= 64 && m <= 256 && n >= 128 && n <= 4096 &&
        k >= 128 && k <= 1024 && n >= m && (2 * n >= 3 * m || 2 * k >= 3 * m) &&
        a && b && c && lda >= k && ldb >= n && ldc >= n &&
        cob_sgemm_blocked_impl(m, n, k, a, lda, b, ldb, c, ldc, 1)) return;
#endif

    cob_sgemm_rowmajor_fallback(m, n, k, a, lda, b, ldb, c, ldc);
}

#ifdef COB_USE_HK_X86_LEAF
static void cob_sgemm_rowmajor_base(int m, int n, int k, const float* a, int lda,
    const float* b, int ldb, float* c, int ldc)
{
    cob_hk_x86_base(0, m, n, k, a, lda, b, ldb, c, ldc);
}
#endif

#ifdef COB_USE_HK15_CHILD
static void cob_sgemm_rowmajor_base(int m, int n, int k, const float* a, int lda,
    const float* b, int ldb, float* c, int ldc)
{
    cob_sgemm_dispatch(0,m,n,k,a,lda,b,ldb,c,ldc);
}

static void cob_hk15_child_product(int m, int n, int k, const float* a, int lda,
    const float* b, int ldb, float* c, int ldc)
{
#ifdef COB_TRACE_HK15_CHILD
    ++cob_hk15_children;
#endif
    cob_sgemm_dispatch(1,m,n,k,a,lda,b,ldb,c,ldc);
}
#endif

#ifdef COB_USE_HK15
#include "cob_hk15.inc"

void cob_sgemm_rowmajor(int m, int n, int k, const float* a, int lda,
    const float* b, int ldb, float* c, int ldc)
{
#ifdef COB_TRACE_HK15_CHILD
    cob_hk15_children = cob_hk15_child_fused = cob_hk15_other_fused = 0;
#ifdef COB_USE_HK15_PARENT_960
    cob_hk15_trace_inside = cob_hk15_lower_children = 0;
#endif
#endif
#ifdef COB_TRACE_HK_X86_LEAF
    cob_hk_x86_children = cob_hk_x86_products = cob_hk_x86_nr = 0;
#endif
#ifdef COB_TRACE_HK_PACK_B
    cob_hk_pack_hits = cob_hk_panel_uses = cob_hk_pack_fallbacks = 0;
#endif
    if (cob_hk15_try(m, n, k, a, lda, b, ldb, c, ldc)) return;
    cob_sgemm_rowmajor_base(m, n, k, a, lda, b, ldb, c, ldc);
}
#endif
