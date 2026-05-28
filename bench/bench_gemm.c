#if !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200112L
#endif

#include "cob_gemm.h"

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__APPLE__)
#include <pthread.h>
#include <pthread/qos.h>
#endif

#if defined(COB_HAVE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#endif

#if defined(COB_HAVE_EXTERNAL_CBLAS)
#ifndef COB_EXTERNAL_CBLAS_HEADER
#define COB_EXTERNAL_CBLAS_HEADER "cblas.h"
#endif
#include COB_EXTERNAL_CBLAS_HEADER
#ifndef COB_EXTERNAL_CBLAS_NAME
#define COB_EXTERNAL_CBLAS_NAME "external_CBLAS"
#endif
#endif

#if defined(COB_HAVE_EXTERNAL_FORTRAN_SGEMM)
#ifndef COB_EXTERNAL_FORTRAN_SGEMM_NAME
#define COB_EXTERNAL_FORTRAN_SGEMM_NAME "external_sgemm_"
#endif
extern void sgemm_(
    char* transa,
    char* transb,
    int* m,
    int* n,
    int* k,
    float* alpha,
    float* a,
    int* lda,
    float* b,
    int* ldb,
    float* beta,
    float* c,
    int* ldc);
#endif

typedef struct bench_shape {
    int m;
    int n;
    int k;
} bench_shape;

typedef void (*bench_fn)(bench_shape shape, const float* a, const float* b, float* c);

typedef struct bench_stats {
    double best;
    double median;
} bench_stats;

enum {
    COB_BENCH_MAX_REPEATS = 31,
    COB_BENCH_AMX_MR = 32,
    COB_BENCH_AMX_NR = 32
};

#ifndef COB_SGEMM_AMX_MC
#define COB_SGEMM_AMX_MC 384
#endif

#ifndef COB_SGEMM_AMX_PACKED_LARGE_MC
#define COB_SGEMM_AMX_PACKED_LARGE_MC 512
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

#ifndef COB_SGEMM_AMX_STRIDED_B_CONFLICT_LDB
#define COB_SGEMM_AMX_STRIDED_B_CONFLICT_LDB 512
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

#ifndef COB_SGEMM_M64_SME_DIRECT_NC
#define COB_SGEMM_M64_SME_DIRECT_NC 1024
#endif

static void configure_benchmark_thread(void)
{
#if defined(__APPLE__) && defined(QOS_CLASS_USER_INTERACTIVE)
    (void)pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
}

static int cob_amx_strided_b_prefers_packed_shape(int m, int n, int k)
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

static int cob_amx_packed_b_mc(int m, int n, int k)
{
    if (n >= 2048 && m == COB_SGEMM_AMX_MC && k >= 1024) {
        return COB_SGEMM_AMX_MC;
    }
    if (m == COB_SGEMM_AMX_PACKED_LARGE_MC && n == 1280 && k == 2048) {
        return COB_SGEMM_AMX_PACKED_LARGE_MC;
    }
    return n >= 2048 ? COB_SGEMM_AMX_PACKED_LARGE_MC : COB_SGEMM_AMX_MC;
}

static int cob_amx_large_block_shape(int m, int n, int k)
{
    return n >= 1152 ||
        (n >= 768 && k >= 3072) ||
        (n == 512 && ((m == 768 && k == 4096) || (m >= 1024 && k >= 4096)));
}

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

static void fill_random(float* x, size_t count, uint32_t* state)
{
    for (size_t i = 0; i < count; ++i) {
        x[i] = rng_f32(state);
    }
}

static float* alloc_f32_aligned(size_t count)
{
    void* ptr = NULL;
    if (posix_memalign(&ptr, 128, count * sizeof(float)) != 0) {
        return NULL;
    }
    return (float*)ptr;
}

static double now_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static float checksum(const float* c, size_t count)
{
    float sum = 0.0f;
    for (size_t i = 0; i < count; i += 97) {
        sum += c[i];
    }
    return sum;
}

static int shape_max_dim(bench_shape shape)
{
    int max_dim = shape.m;
    if (shape.n > max_dim) {
        max_dim = shape.n;
    }
    if (shape.k > max_dim) {
        max_dim = shape.k;
    }
    return max_dim;
}

static int env_int_clamped(const char* name, int fallback, int min_value, int max_value);

static int timed_iterations(bench_shape shape)
{
    const int forced = env_int_clamped("COB_BENCH_ITERS", 0, 0, 100000000);
    if (forced > 0) {
        return forced;
    }

    const int n = shape_max_dim(shape);
    if (n <= 64) {
        return 8192;
    }
    if (n <= 128) {
        return 2048;
    }
    if (n <= 256) {
        return 256;
    }
    if (n <= 512) {
        return 32;
    }
    return 1;
}

static int env_int_clamped(const char* name, int fallback, int min_value, int max_value)
{
    const char* value = getenv(name);
    if (value == NULL || value[0] == '\0') {
        return fallback;
    }

    char* end = NULL;
    const long parsed = strtol(value, &end, 10);
    if (end == value || *end != '\0') {
        return fallback;
    }
    if (parsed < min_value) {
        return min_value;
    }
    if (parsed > max_value) {
        return max_value;
    }
    return (int)parsed;
}

static int timed_repeats(bench_shape shape)
{
    const int n = shape_max_dim(shape);
    const int fallback = n <= 256 ? 7 : 5;
    return env_int_clamped("COB_BENCH_REPEATS", fallback, 1, COB_BENCH_MAX_REPEATS);
}

static int bench_pack_setup_enabled(void)
{
    return env_int_clamped("COB_BENCH_PACK_SETUP", 0, 0, 1) != 0;
}

static int bench_csv_enabled(void)
{
    return env_int_clamped("COB_BENCH_CSV", 0, 0, 1) != 0;
}

static int bench_route_enabled(void)
{
    return env_int_clamped("COB_BENCH_ROUTE", 0, 0, 1) != 0;
}

static int bench_hot_seconds(void)
{
    return env_int_clamped("COB_BENCH_HOT_SECONDS", 0, 0, 3600);
}

static int bench_cooldown_us(void)
{
    return env_int_clamped("COB_BENCH_COOLDOWN_US", 0, 0, 10000000);
}

static void sleep_microseconds(int microseconds)
{
    if (microseconds <= 0) {
        return;
    }
    struct timespec request;
    request.tv_sec = microseconds / 1000000;
    request.tv_nsec = (long)(microseconds % 1000000) * 1000L;
    while (nanosleep(&request, &request) != 0) {
    }
}

static int bench_only_matches(const char* only, const char* name)
{
    if (only == NULL || only[0] == '\0') {
        return 1;
    }
    if (strcmp(only, name) == 0) {
        return 1;
    }
    if (strcmp(only, "one-shot") == 0 && strcmp(name, "cob one-shot") == 0) {
        return 1;
    }
    if (strcmp(only, "packed") == 0 && strcmp(name, "cob packed-B") == 0) {
        return 1;
    }
    if (strcmp(only, "packed-B") == 0 && strcmp(name, "cob packed-B") == 0) {
        return 1;
    }
    if (strcmp(only, "packed-AB") == 0 && strcmp(name, "cob packed-AB") == 0) {
        return 1;
    }
    if (strcmp(only, "packed_ab") == 0 && strcmp(name, "cob packed-AB") == 0) {
        return 1;
    }
    if (strcmp(only, "pack-setup") == 0 && strcmp(name, "cob pack-B setup") == 0) {
        return 1;
    }
    if (strcmp(only, "accelerate") == 0 && strcmp(name, "Accelerate") == 0) {
        return 1;
    }
    return 0;
}

static int pack_setup_iterations(bench_shape shape)
{
    const int n = shape_max_dim(shape);
    if (n <= 128) {
        return 512;
    }
    if (n <= 256) {
        return 128;
    }
    if (n <= 512) {
        return 16;
    }
    return 1;
}

static void sort_doubles(double* values, int count)
{
    for (int i = 1; i < count; ++i) {
        const double value = values[i];
        int j = i - 1;
        while (j >= 0 && values[j] > value) {
            values[j + 1] = values[j];
            --j;
        }
        values[j + 1] = value;
    }
}

static bench_stats summarize_times(double* times, int count)
{
    sort_doubles(times, count);
    bench_stats stats;
    stats.best = times[0];
    stats.median = times[count / 2];
    return stats;
}

static void format_shape(char* dst, size_t dst_size, bench_shape shape)
{
    snprintf(dst, dst_size, "%dx%dx%d", shape.m, shape.n, shape.k);
}

static void bench_cob_direct(bench_shape shape, const float* a, const float* b, float* c)
{
    cob_sgemm_rowmajor(shape.m, shape.n, shape.k, a, shape.k, b, shape.n, c, shape.n);
}

static int is_apple_amx_build(void)
{
#if defined(__APPLE__) && defined(__aarch64__) && !defined(COB_DISABLE_APPLE_AMX)
    return 1;
#else
    return 0;
#endif
}

static int is_apple_sme_build(void)
{
#if defined(__APPLE__) && defined(__aarch64__) && !defined(COB_DISABLE_APPLE_SME)
    return 1;
#else
    return 0;
#endif
}

static int cob_sme_direct_extra_n_shape(int m, int n, int k)
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
    if (m == 512 && n == 512 && k == 3072) {
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

static int cob_m64_sme_direct_nc_shape(int n, int k)
{
    return ((n == 2560 && k >= 12288) || (n == 3072 && k >= 4096) ||
               (n >= 3584 && n < 4096 && k >= 7168)) &&
        COB_SGEMM_M64_SME_DIRECT_NC >= 64;
}

static int cob_m64_sme_wide_reuse_shape(int n, int k)
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

static const char* cob_one_shot_route(bench_shape shape)
{
    const int m = shape.m;
    const int n = shape.n;
    const int k = shape.k;
    if (!is_apple_amx_build()) {
        return "fallback";
    }

    const int aligned32 = (m % COB_BENCH_AMX_MR) == 0 && (n % COB_BENCH_AMX_NR) == 0;
    if (is_apple_sme_build()) {
        const int use_m64 = m == 64;
        const int use_m96_128_k512 =
            (m == 96 || m == 128) && n >= COB_SGEMM_M96_128_SME_REUSE_K512_MIN_N && k == 512;
        const int use_m96_128_k1024 =
            (m == 96 || m == 128) && n >= COB_SGEMM_M96_128_SME_REUSE_K1024_MIN_N && k >= 1024;
        if (m == 512 && n == 1216 && k == 3072 && (n % 64) == 0) {
            return "sme_medium_reuse";
        }
        if ((m == 768 || m == 1024 || m == 1280 || (m == 1536 && n == 512)) &&
            (n == 512 || n == 768) && k == 4096 && (n % 64) == 0) {
            return "sme_large_reuse";
        }
        const int use_long_n_k512 =
            (use_m64 && n >= COB_SGEMM_M64_SME_LONG_N_K512_MIN_N && k == 512) ||
            use_m96_128_k512;
        const int use_n4096_large_k = use_m64 && n == 4096 && k >= 7168;
        const int use_wide = use_m64 && cob_m64_sme_wide_reuse_shape(n, k);
        if ((use_m64 || use_m96_128_k512 || use_m96_128_k1024) &&
            (use_long_n_k512 || use_n4096_large_k || use_wide || use_m96_128_k1024) &&
            (n % 64) == 0) {
            return "sme_skinny_reuse";
        }
    }

    if (m >= 96 && m <= 128 && n >= 1024 && k >= 512 && aligned32) {
        return "amx_skinny_chunks";
    }
    if (m == 512 && k == 2048 &&
        (n == 896 || n == 1024 || n == 1152 || n == 1280) && aligned32) {
        return "amx_chunked_b";
    }

    if (is_apple_sme_build()) {
        const int use_large_k_skinny =
            m == 64 &&
            ((n >= 1024 && n <= 4096 && k >= 2048) ||
                (n == 1024 && k == 1536) ||
                (n >= 1408 && n <= 4096 && k == 1536));
        const int use_mid_n_k512 =
            n >= 2048 && n < COB_SGEMM_M64_SME_LONG_N_K512_MIN_N &&
            (n % 512) == 0;
        const int use_long_n_k512 =
            m == 64 && (use_mid_n_k512 || n >= COB_SGEMM_M64_SME_LONG_N_K512_MIN_N) &&
            k == 512;
        if ((use_large_k_skinny || use_long_n_k512) && (n % 64) == 0) {
            if (use_large_k_skinny && cob_m64_sme_direct_nc_shape(n, k)) {
                return "sme_skinny_strided_nc";
            }
            return "sme_skinny_strided";
        }

        if ((m == 384 && n == 384 && k == 384) || (m == 768 && n == 768 && k == 768)) {
            return "sme_medium_direct";
        }
    }

    const int use_large_block =
        m >= COB_SGEMM_AMX_MC &&
        cob_amx_large_block_shape(m, n, k) &&
        k >= 512 && aligned32;
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
        !cob_amx_strided_b_prefers_packed_shape(m, n, k) &&
        n != COB_SGEMM_AMX_STRIDED_B_CONFLICT_LDB && aligned32;
    if (use_strided_b) {
        return "amx_strided_b";
    }

    if (is_apple_sme_build()) {
        const int use_square_384 = m == 384 && n == 384 && k == 384;
        const int use_square_768 = m == 768 && n == 768 && k == 768;
        const int use_extra_n = cob_sme_direct_extra_n_shape(m, n, k);
        if ((use_square_384 || use_square_768 || use_extra_n ||
                (m >= 832 && m <= COB_SGEMM_SME_DIRECT_MAX_N &&
                    n >= 832 && n <= COB_SGEMM_SME_DIRECT_MAX_N &&
                    k >= 832 && k <= COB_SGEMM_SME_DIRECT_MAX_N)) &&
            (n != 1024 || use_extra_n) && (n % 64) == 0 && (m % COB_BENCH_AMX_MR) == 0) {
            return "sme_medium_direct";
        }
    }

    if (n == COB_SGEMM_AMX_STRIDED_B_CONFLICT_LDB && m >= 512 &&
        !(m == 768 && k == 1536) && k < 2048 &&
        (m % COB_BENCH_AMX_MR) == 0 && k >= 512 && is_apple_sme_build()) {
        return "pack_b_then_sme";
    }
    if (use_large_block) {
        return "amx_packed_large_block";
    }
    if (aligned32) {
        return "amx_packed_full";
    }
    return "amx_packed_partial";
}

static int cob_packed_b_sme_excluded_shape(int m, int n, int k)
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

static int cob_packed_b_small_bouter_shape(int m, int n, int k)
{
    return (m == 64 && n >= 2048 && k >= 1536) ||
        (m >= 96 && m <= 128 && n >= 4096 && k >= 1024);
}

static const char* cob_packed_b_route(bench_shape shape)
{
    const int m = shape.m;
    const int n = shape.n;
    const int k = shape.k;
    if (!is_apple_amx_build()) {
        return "packed_fallback";
    }
    if (is_apple_sme_build() && m >= 512 && n >= 512 && k >= 512 &&
        n <= COB_SGEMM_SME_PACKED_MAX_N &&
        !cob_packed_b_sme_excluded_shape(m, n, k) &&
        (m % COB_BENCH_AMX_MR) == 0 && (n % 64) == 0) {
        return "packed_sme";
    }
    if ((m % COB_BENCH_AMX_MR) == 0 && (n % COB_BENCH_AMX_NR) == 0) {
        if (m >= cob_amx_packed_b_mc(m, n, k) &&
            cob_amx_large_block_shape(m, n, k) &&
            k >= 512) {
            return "packed_amx_large_block";
        }
        if (cob_packed_b_small_bouter_shape(m, n, k)) {
            return "packed_amx_small_bouter";
        }
        return "packed_amx_full";
    }
    return "packed_amx_partial";
}

static const char* cob_packed_ab_route(bench_shape shape)
{
    (void)shape;
    return is_apple_amx_build() ? "packed_ab_amx" : "packed_ab_fallback";
}

#if defined(COB_HAVE_ACCELERATE)
static void bench_accelerate(bench_shape shape, const float* a, const float* b, float* c)
{
    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        shape.m,
        shape.n,
        shape.k,
        1.0f,
        a,
        shape.k,
        b,
        shape.n,
        0.0f,
        c,
        shape.n);
}
#endif

#if defined(COB_HAVE_EXTERNAL_CBLAS)
static void bench_external_cblas(bench_shape shape, const float* a, const float* b, float* c)
{
    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        shape.m,
        shape.n,
        shape.k,
        1.0f,
        a,
        shape.k,
        b,
        shape.n,
        0.0f,
        c,
        shape.n);
}
#endif

#if defined(COB_HAVE_EXTERNAL_FORTRAN_SGEMM)
static void bench_external_fortran_sgemm(bench_shape shape, const float* a, const float* b, float* c)
{
    char trans = 'N';
    int fm = shape.n;
    int fn = shape.m;
    int fk = shape.k;
    int lda = shape.n;
    int ldb = shape.k;
    int ldc = shape.n;
    const float alpha_in = 1.0f;
    const float beta_in = 0.0f;
    float alpha = alpha_in;
    float beta = beta_in;

    /*
     * Row-major C = A * B is column-major C^T = B^T * A^T over the same buffers.
     */
    sgemm_(
        &trans,
        &trans,
        &fm,
        &fn,
        &fk,
        &alpha,
        (float*)b,
        &lda,
        (float*)a,
        &ldb,
        &beta,
        c,
        &ldc);
}
#endif

static double run_case(
    const char* name,
    const char* route,
    bench_fn fn,
    bench_shape shape,
    const float* a,
    const float* b,
    float* c,
    int csv,
    int route_enabled)
{
    const int max_dim = shape_max_dim(shape);
    const int warmups = max_dim <= 256 ? 2 : 1;
    const int repeats = timed_repeats(shape);
    const int iters = timed_iterations(shape);
    const int cooldown_us = bench_cooldown_us();
    double times[COB_BENCH_MAX_REPEATS];

    for (int i = 0; i < warmups; ++i) {
        fn(shape, a, b, c);
    }

    const int hot_seconds = bench_hot_seconds();
    if (hot_seconds > 0) {
        const double ops = 2.0 * (double)shape.m * (double)shape.n * (double)shape.k;
        const double t0 = now_seconds();
        const double deadline = t0 + (double)hot_seconds;
        uint64_t calls = 0;
        do {
            fn(shape, a, b, c);
            ++calls;
        } while (now_seconds() < deadline);
        const double elapsed = now_seconds() - t0;
        const double gflops = ops * (double)calls / elapsed / 1.0e9;
        const double sum = (double)checksum(c, (size_t)shape.m * (size_t)shape.n);
        if (csv) {
            if (route_enabled) {
                printf(
                    "gemm,%s,%d,%d,%d,%.6f,%.6f,GF/s,%.9f,%.9f,%.9e,%s\n",
                    name,
                    shape.m,
                    shape.n,
                    shape.k,
                    gflops,
                    gflops,
                    elapsed / (double)calls,
                    elapsed / (double)calls,
                    sum,
                    route);
            } else {
                printf(
                    "gemm,%s,%d,%d,%d,%.6f,%.6f,GF/s,%.9f,%.9f,%.9e\n",
                    name,
                    shape.m,
                    shape.n,
                    shape.k,
                    gflops,
                    gflops,
                    elapsed / (double)calls,
                    elapsed / (double)calls,
                    sum);
            }
        } else {
            char shape_text[32];
            format_shape(shape_text, sizeof(shape_text), shape);
            printf(
                "%-18s %-14s  hot %8.2f GF/s  %9.3f s  calls=%llu  checksum=% .5e",
                name,
                shape_text,
                gflops,
                elapsed,
                (unsigned long long)calls,
                sum);
            if (route_enabled && route[0] != '\0') {
                printf("  route=%s", route);
            }
            printf("\n");
        }
        return gflops;
    }

    for (int i = 0; i < repeats; ++i) {
        const double t0 = now_seconds();
        for (int iter = 0; iter < iters; ++iter) {
            fn(shape, a, b, c);
        }
        const double t1 = now_seconds();
        times[i] = (t1 - t0) / (double)iters;
        sleep_microseconds(cooldown_us);
    }

    const bench_stats stats = summarize_times(times, repeats);
    const double ops = 2.0 * (double)shape.m * (double)shape.n * (double)shape.k;
    const double best_gflops = ops / stats.best / 1.0e9;
    const double median_gflops = ops / stats.median / 1.0e9;
    const double sum = (double)checksum(c, (size_t)shape.m * (size_t)shape.n);
    if (csv) {
        if (route_enabled) {
            printf(
                "gemm,%s,%d,%d,%d,%.6f,%.6f,GF/s,%.9f,%.9f,%.9e,%s\n",
                name,
                shape.m,
                shape.n,
                shape.k,
                best_gflops,
                median_gflops,
                stats.best,
                stats.median,
                sum,
                route);
        } else {
            printf(
                "gemm,%s,%d,%d,%d,%.6f,%.6f,GF/s,%.9f,%.9f,%.9e\n",
                name,
                shape.m,
                shape.n,
                shape.k,
                best_gflops,
                median_gflops,
                stats.best,
                stats.median,
                sum);
        }
    } else {
        char shape_text[32];
        format_shape(shape_text, sizeof(shape_text), shape);
        printf(
            "%-18s %-14s  best %8.2f GF/s  med %8.2f GF/s  %9.6f s  checksum=% .5e",
            name,
            shape_text,
            best_gflops,
            median_gflops,
            stats.best,
            sum);
        if (route_enabled && route[0] != '\0') {
            printf("  route=%s", route);
        }
        printf("\n");
    }
    return best_gflops;
}

static double run_case_cob_packed_reuse(
    bench_shape shape,
    const float* a,
    const float* b,
    float* c,
    int csv,
    int route_enabled)
{
    cob_packed_b_f32 packed_b;
    if (cob_sgemm_pack_b(&packed_b, shape.k, shape.n, b, shape.n) != 0) {
        fprintf(stderr, "packed-B allocation failed for %dx%dx%d\n", shape.m, shape.n, shape.k);
        return 0.0;
    }

    const int max_dim = shape_max_dim(shape);
    const int warmups = max_dim <= 256 ? 2 : 1;
    const int repeats = timed_repeats(shape);
    const int iters = timed_iterations(shape);
    const int cooldown_us = bench_cooldown_us();
    double times[COB_BENCH_MAX_REPEATS];

    for (int i = 0; i < warmups; ++i) {
        cob_sgemm_rowmajor_packed_b(
            shape.m, shape.n, shape.k, a, shape.k, &packed_b, c, shape.n);
    }

    const int hot_seconds = bench_hot_seconds();
    if (hot_seconds > 0) {
        const double ops = 2.0 * (double)shape.m * (double)shape.n * (double)shape.k;
        const double t0 = now_seconds();
        const double deadline = t0 + (double)hot_seconds;
        uint64_t calls = 0;
        do {
            cob_sgemm_rowmajor_packed_b(
                shape.m, shape.n, shape.k, a, shape.k, &packed_b, c, shape.n);
            ++calls;
        } while (now_seconds() < deadline);
        const double elapsed = now_seconds() - t0;
        const double gflops = ops * (double)calls / elapsed / 1.0e9;
        const double sum = (double)checksum(c, (size_t)shape.m * (size_t)shape.n);
        if (csv) {
            if (route_enabled) {
                printf(
                    "gemm,cob packed-B,%d,%d,%d,%.6f,%.6f,GF/s,%.9f,%.9f,%.9e,%s\n",
                    shape.m,
                    shape.n,
                    shape.k,
                    gflops,
                    gflops,
                    elapsed / (double)calls,
                    elapsed / (double)calls,
                    sum,
                    cob_packed_b_route(shape));
            } else {
                printf(
                    "gemm,cob packed-B,%d,%d,%d,%.6f,%.6f,GF/s,%.9f,%.9f,%.9e\n",
                    shape.m,
                    shape.n,
                    shape.k,
                    gflops,
                    gflops,
                    elapsed / (double)calls,
                    elapsed / (double)calls,
                    sum);
            }
        } else {
            char shape_text[32];
            format_shape(shape_text, sizeof(shape_text), shape);
            printf(
                "%-18s %-14s  hot %8.2f GF/s  %9.3f s  calls=%llu  checksum=% .5e",
                "cob packed-B",
                shape_text,
                gflops,
                elapsed,
                (unsigned long long)calls,
                sum);
            if (route_enabled) {
                printf("  route=%s", cob_packed_b_route(shape));
            }
            printf("\n");
        }

        cob_sgemm_free_packed_b(&packed_b);
        return gflops;
    }

    for (int i = 0; i < repeats; ++i) {
        const double t0 = now_seconds();
        for (int iter = 0; iter < iters; ++iter) {
            cob_sgemm_rowmajor_packed_b(
                shape.m, shape.n, shape.k, a, shape.k, &packed_b, c, shape.n);
        }
        const double t1 = now_seconds();
        times[i] = (t1 - t0) / (double)iters;
        sleep_microseconds(cooldown_us);
    }

    const bench_stats stats = summarize_times(times, repeats);
    const double ops = 2.0 * (double)shape.m * (double)shape.n * (double)shape.k;
    const double best_gflops = ops / stats.best / 1.0e9;
    const double median_gflops = ops / stats.median / 1.0e9;
    const double sum = (double)checksum(c, (size_t)shape.m * (size_t)shape.n);
    if (csv) {
        if (route_enabled) {
            printf(
                "gemm,cob packed-B,%d,%d,%d,%.6f,%.6f,GF/s,%.9f,%.9f,%.9e,%s\n",
                shape.m,
                shape.n,
                shape.k,
                best_gflops,
                median_gflops,
                stats.best,
                stats.median,
                sum,
                cob_packed_b_route(shape));
        } else {
            printf(
                "gemm,cob packed-B,%d,%d,%d,%.6f,%.6f,GF/s,%.9f,%.9f,%.9e\n",
                shape.m,
                shape.n,
                shape.k,
                best_gflops,
                median_gflops,
                stats.best,
                stats.median,
                sum);
        }
    } else {
        char shape_text[32];
        format_shape(shape_text, sizeof(shape_text), shape);
        printf("%-18s %-14s  best %8.2f GF/s  med %8.2f GF/s  %9.6f s  checksum=% .5e",
            "cob packed-B",
            shape_text,
            best_gflops,
            median_gflops,
            stats.best,
            sum);
        if (route_enabled) {
            printf("  route=%s", cob_packed_b_route(shape));
        }
        printf("\n");
    }

    cob_sgemm_free_packed_b(&packed_b);
    return best_gflops;
}

static double run_case_cob_packed_ab_reuse(
    bench_shape shape,
    const float* a,
    const float* b,
    float* c,
    int csv,
    int route_enabled)
{
    cob_packed_a_f32 packed_a;
    cob_packed_b_f32 packed_b;
    if (cob_sgemm_pack_a(&packed_a, shape.m, shape.k, a, shape.k) != 0) {
        fprintf(stderr, "packed-A allocation failed for %dx%dx%d\n", shape.m, shape.n, shape.k);
        return 0.0;
    }
    if (cob_sgemm_pack_b(&packed_b, shape.k, shape.n, b, shape.n) != 0) {
        fprintf(stderr, "packed-B allocation failed for %dx%dx%d\n", shape.m, shape.n, shape.k);
        cob_sgemm_free_packed_a(&packed_a);
        return 0.0;
    }

    const int max_dim = shape_max_dim(shape);
    const int warmups = max_dim <= 256 ? 2 : 1;
    const int repeats = timed_repeats(shape);
    const int iters = timed_iterations(shape);
    const int cooldown_us = bench_cooldown_us();
    double times[COB_BENCH_MAX_REPEATS];

    for (int i = 0; i < warmups; ++i) {
        cob_sgemm_rowmajor_packed_ab(
            shape.m, shape.n, shape.k, &packed_a, &packed_b, c, shape.n);
    }

    for (int i = 0; i < repeats; ++i) {
        const double t0 = now_seconds();
        for (int iter = 0; iter < iters; ++iter) {
            cob_sgemm_rowmajor_packed_ab(
                shape.m, shape.n, shape.k, &packed_a, &packed_b, c, shape.n);
        }
        const double t1 = now_seconds();
        times[i] = (t1 - t0) / (double)iters;
        sleep_microseconds(cooldown_us);
    }

    const bench_stats stats = summarize_times(times, repeats);
    const double ops = 2.0 * (double)shape.m * (double)shape.n * (double)shape.k;
    const double best_gflops = ops / stats.best / 1.0e9;
    const double median_gflops = ops / stats.median / 1.0e9;
    const double sum = (double)checksum(c, (size_t)shape.m * (size_t)shape.n);
    if (csv) {
        if (route_enabled) {
            printf(
                "gemm,cob packed-AB,%d,%d,%d,%.6f,%.6f,GF/s,%.9f,%.9f,%.9e,%s\n",
                shape.m,
                shape.n,
                shape.k,
                best_gflops,
                median_gflops,
                stats.best,
                stats.median,
                sum,
                cob_packed_ab_route(shape));
        } else {
            printf(
                "gemm,cob packed-AB,%d,%d,%d,%.6f,%.6f,GF/s,%.9f,%.9f,%.9e\n",
                shape.m,
                shape.n,
                shape.k,
                best_gflops,
                median_gflops,
                stats.best,
                stats.median,
                sum);
        }
    } else {
        char shape_text[32];
        format_shape(shape_text, sizeof(shape_text), shape);
        printf("%-18s %-14s  best %8.2f GF/s  med %8.2f GF/s  %9.6f s  checksum=% .5e",
            "cob packed-AB",
            shape_text,
            best_gflops,
            median_gflops,
            stats.best,
            sum);
        if (route_enabled) {
            printf("  route=%s", cob_packed_ab_route(shape));
        }
        printf("\n");
    }

    cob_sgemm_free_packed_b(&packed_b);
    cob_sgemm_free_packed_a(&packed_a);
    return best_gflops;
}

static double run_case_cob_pack_setup(bench_shape shape, const float* b, int csv, int route_enabled)
{
    const int repeats = timed_repeats(shape);
    const int iters = pack_setup_iterations(shape);
    const int cooldown_us = bench_cooldown_us();
    double times[COB_BENCH_MAX_REPEATS];
    size_t bytes_per_iter = 0;

    cob_packed_b_f32 warmup;
    if (cob_sgemm_pack_b(&warmup, shape.k, shape.n, b, shape.n) != 0) {
        fprintf(stderr, "packed-B setup allocation failed for %dx%dx%d\n",
            shape.m,
            shape.n,
            shape.k);
        return 0.0;
    }
    bytes_per_iter = (size_t)shape.k * (size_t)shape.n * sizeof(float) + warmup.bytes;
    cob_sgemm_free_packed_b(&warmup);

    for (int i = 0; i < repeats; ++i) {
        const double t0 = now_seconds();
        for (int iter = 0; iter < iters; ++iter) {
            cob_packed_b_f32 packed_b;
            if (cob_sgemm_pack_b(&packed_b, shape.k, shape.n, b, shape.n) != 0) {
                fprintf(stderr, "packed-B setup allocation failed for %dx%dx%d\n",
                    shape.m,
                    shape.n,
                    shape.k);
                return 0.0;
            }
            cob_sgemm_free_packed_b(&packed_b);
        }
        const double t1 = now_seconds();
        times[i] = (t1 - t0) / (double)iters;
        sleep_microseconds(cooldown_us);
    }

    const bench_stats stats = summarize_times(times, repeats);
    const double best_gbs = (double)bytes_per_iter / stats.best / 1.0e9;
    const double median_gbs = (double)bytes_per_iter / stats.median / 1.0e9;
    if (csv) {
        if (route_enabled) {
            printf(
                "pack_b_setup,cob,%d,%d,%d,%.6f,%.6f,GB/s,%.9f,%.9f,,pack_b_setup\n",
                shape.m,
                shape.n,
                shape.k,
                best_gbs,
                median_gbs,
                stats.best,
                stats.median);
        } else {
            printf(
                "pack_b_setup,cob,%d,%d,%d,%.6f,%.6f,GB/s,%.9f,%.9f,\n",
                shape.m,
                shape.n,
                shape.k,
                best_gbs,
                median_gbs,
                stats.best,
                stats.median);
        }
    } else {
        char shape_text[32];
        format_shape(shape_text, sizeof(shape_text), shape);
        printf("%-18s %-14s  best %8.2f GB/s  med %8.2f GB/s  %9.6f s\n",
            "cob pack-B setup",
            shape_text,
            best_gbs,
            median_gbs,
            stats.best);
    }
    return best_gbs;
}

static int is_shape_separator(char ch)
{
    return ch == 'x' || ch == 'X' || ch == ',' || ch == ':';
}

static int parse_positive_int(const char** cursor, int* out)
{
    char* end = NULL;
    const long parsed = strtol(*cursor, &end, 10);
    if (end == *cursor || parsed <= 0 || parsed > INT_MAX) {
        return 0;
    }

    *cursor = end;
    *out = (int)parsed;
    return 1;
}

static int parse_shape_arg(const char* arg, bench_shape* shape)
{
    const char* cursor = arg;
    if (!parse_positive_int(&cursor, &shape->m)) {
        return 0;
    }

    if (*cursor == '\0') {
        shape->n = shape->m;
        shape->k = shape->m;
        return 1;
    }

    if (!is_shape_separator(*cursor)) {
        return 0;
    }
    ++cursor;
    if (!parse_positive_int(&cursor, &shape->n)) {
        return 0;
    }

    if (!is_shape_separator(*cursor)) {
        return 0;
    }
    ++cursor;
    if (!parse_positive_int(&cursor, &shape->k)) {
        return 0;
    }

    return *cursor == '\0';
}

int main(int argc, char** argv)
{
    static const int default_sizes[] = {64, 128, 192, 256, 384, 512, 768, 1024};

    configure_benchmark_thread();

    setenv("VECLIB_MAXIMUM_THREADS", "1", 1);
    setenv("ACCELERATE_MAXIMUM_THREADS", "1", 1);
    setenv("BLIS_NUM_THREADS", "1", 1);
    setenv("OPENBLAS_NUM_THREADS", "1", 1);
    setenv("OMP_NUM_THREADS", "1", 1);

    bench_shape shapes[32];
    int shape_count = 0;
    if (argc > 1) {
        for (int i = 1; i < argc && shape_count < 32; ++i) {
            bench_shape shape;
            if (parse_shape_arg(argv[i], &shape)) {
                shapes[shape_count++] = shape;
            } else {
                fprintf(stderr, "invalid shape: %s (use N or MxNxK)\n", argv[i]);
                return 1;
            }
        }
    } else {
        shape_count = (int)(sizeof(default_sizes) / sizeof(default_sizes[0]));
        for (int i = 0; i < shape_count; ++i) {
            shapes[i].m = default_sizes[i];
            shapes[i].n = default_sizes[i];
            shapes[i].k = default_sizes[i];
        }
    }

    const int show_pack_setup = bench_pack_setup_enabled();
    const int csv = bench_csv_enabled();
    const int route_enabled = bench_route_enabled();
    const char* bench_only = getenv("COB_BENCH_ONLY");
    if (csv) {
        if (route_enabled) {
            printf(
                "kind,implementation,m,n,k,best_throughput,median_throughput,unit,best_seconds,median_seconds,checksum,route\n");
        } else {
            printf(
                "kind,implementation,m,n,k,best_throughput,median_throughput,unit,best_seconds,median_seconds,checksum\n");
        }
    } else {
        printf("single-thread FP32 row-major GEMM\n");
        printf(
            "cob one-shot includes packing; cob packed-B excludes B packing; cob packed-AB excludes A/B packing\n\n");
    }

    for (int si = 0; si < shape_count; ++si) {
        const bench_shape shape = shapes[si];
        const size_t a_count = (size_t)shape.m * (size_t)shape.k;
        const size_t b_count = (size_t)shape.k * (size_t)shape.n;
        const size_t c_count = (size_t)shape.m * (size_t)shape.n;
        float* a = alloc_f32_aligned(a_count);
        float* b = alloc_f32_aligned(b_count);
        float* c = alloc_f32_aligned(c_count);
        if (a == NULL || b == NULL || c == NULL) {
            fprintf(stderr, "allocation failed for %dx%dx%d\n", shape.m, shape.n, shape.k);
            free(a);
            free(b);
            free(c);
            return 1;
        }

        uint32_t state =
            0x9e3779b9u ^ (uint32_t)(shape.m * 131 + shape.n * 17 + shape.k);
        fill_random(a, a_count, &state);
        fill_random(b, b_count, &state);

        if (bench_only_matches(bench_only, "cob one-shot")) {
            run_case(
                "cob one-shot",
                cob_one_shot_route(shape),
                bench_cob_direct,
                shape,
                a,
                b,
                c,
                csv,
                route_enabled);
        }
        if (bench_only_matches(bench_only, "cob packed-B")) {
            run_case_cob_packed_reuse(shape, a, b, c, csv, route_enabled);
        }
        if (bench_only_matches(bench_only, "cob packed-AB")) {
            run_case_cob_packed_ab_reuse(shape, a, b, c, csv, route_enabled);
        }
        if (show_pack_setup && bench_only_matches(bench_only, "cob pack-B setup")) {
            run_case_cob_pack_setup(shape, b, csv, route_enabled);
        }
#if defined(COB_HAVE_ACCELERATE)
        if (bench_only_matches(bench_only, "Accelerate")) {
            run_case("Accelerate", "", bench_accelerate, shape, a, b, c, csv, route_enabled);
        }
#endif
#if defined(COB_HAVE_EXTERNAL_CBLAS)
        if (bench_only_matches(bench_only, COB_EXTERNAL_CBLAS_NAME)) {
            run_case(
                COB_EXTERNAL_CBLAS_NAME,
                "",
                bench_external_cblas,
                shape,
                a,
                b,
                c,
                csv,
                route_enabled);
        }
#endif
#if defined(COB_HAVE_EXTERNAL_FORTRAN_SGEMM)
        if (bench_only_matches(bench_only, COB_EXTERNAL_FORTRAN_SGEMM_NAME)) {
            run_case(
                COB_EXTERNAL_FORTRAN_SGEMM_NAME,
                "",
                bench_external_fortran_sgemm,
                shape,
                a,
                b,
                c,
                csv,
                route_enabled);
        }
#endif
        if (!csv) {
            printf("\n");
        }

        free(a);
        free(b);
        free(c);
    }

    return 0;
}
