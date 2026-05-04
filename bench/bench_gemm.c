#if !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200112L
#endif

#include "cob_gemm.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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

typedef void (*bench_fn)(int n, const float* a, const float* b, float* c);

typedef struct bench_stats {
    double best;
    double median;
} bench_stats;

enum {
    COB_BENCH_MAX_REPEATS = 31
};

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

static float checksum(const float* c, int count)
{
    float sum = 0.0f;
    for (int i = 0; i < count; i += 97) {
        sum += c[i];
    }
    return sum;
}

static int timed_iterations(int n)
{
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

static int timed_repeats(int n)
{
    const int fallback = n <= 256 ? 7 : 5;
    return env_int_clamped("COB_BENCH_REPEATS", fallback, 1, COB_BENCH_MAX_REPEATS);
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

static void bench_cob_direct(int n, const float* a, const float* b, float* c)
{
    cob_sgemm_rowmajor(n, n, n, a, n, b, n, c, n);
}

#if defined(COB_HAVE_ACCELERATE)
static void bench_accelerate(int n, const float* a, const float* b, float* c)
{
    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        n,
        n,
        n,
        1.0f,
        a,
        n,
        b,
        n,
        0.0f,
        c,
        n);
}
#endif

#if defined(COB_HAVE_EXTERNAL_CBLAS)
static void bench_external_cblas(int n, const float* a, const float* b, float* c)
{
    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        n,
        n,
        n,
        1.0f,
        a,
        n,
        b,
        n,
        0.0f,
        c,
        n);
}
#endif

#if defined(COB_HAVE_EXTERNAL_FORTRAN_SGEMM)
static void bench_external_fortran_sgemm(int n, const float* a, const float* b, float* c)
{
    char trans = 'N';
    int dim = n;
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
        &dim,
        &dim,
        &dim,
        &alpha,
        (float*)b,
        &dim,
        (float*)a,
        &dim,
        &beta,
        c,
        &dim);
}
#endif

static double run_case(const char* name, bench_fn fn, int n, const float* a, const float* b, float* c)
{
    const int warmups = n <= 256 ? 2 : 1;
    const int repeats = timed_repeats(n);
    const int iters = timed_iterations(n);
    double times[COB_BENCH_MAX_REPEATS];

    for (int i = 0; i < warmups; ++i) {
        fn(n, a, b, c);
    }

    for (int i = 0; i < repeats; ++i) {
        const double t0 = now_seconds();
        for (int iter = 0; iter < iters; ++iter) {
            fn(n, a, b, c);
        }
        const double t1 = now_seconds();
        times[i] = (t1 - t0) / (double)iters;
    }

    const bench_stats stats = summarize_times(times, repeats);
    const double ops = 2.0 * (double)n * (double)n * (double)n;
    const double best_gflops = ops / stats.best / 1.0e9;
    const double median_gflops = ops / stats.median / 1.0e9;
    printf("%-18s n=%4d  best %8.2f GF/s  med %8.2f GF/s  %9.6f s  checksum=% .5e\n",
        name,
        n,
        best_gflops,
        median_gflops,
        stats.best,
        (double)checksum(c, n * n));
    return best_gflops;
}

static double run_case_cob_packed_reuse(int n, const float* a, const float* b, float* c)
{
    cob_packed_b_f32 packed_b;
    if (cob_sgemm_pack_b(&packed_b, n, n, b, n) != 0) {
        fprintf(stderr, "packed-B allocation failed for n=%d\n", n);
        return 0.0;
    }

    const int warmups = n <= 256 ? 2 : 1;
    const int repeats = timed_repeats(n);
    const int iters = timed_iterations(n);
    double times[COB_BENCH_MAX_REPEATS];

    for (int i = 0; i < warmups; ++i) {
        cob_sgemm_rowmajor_packed_b(n, n, n, a, n, &packed_b, c, n);
    }

    for (int i = 0; i < repeats; ++i) {
        const double t0 = now_seconds();
        for (int iter = 0; iter < iters; ++iter) {
            cob_sgemm_rowmajor_packed_b(n, n, n, a, n, &packed_b, c, n);
        }
        const double t1 = now_seconds();
        times[i] = (t1 - t0) / (double)iters;
    }

    const bench_stats stats = summarize_times(times, repeats);
    const double ops = 2.0 * (double)n * (double)n * (double)n;
    const double best_gflops = ops / stats.best / 1.0e9;
    const double median_gflops = ops / stats.median / 1.0e9;
    printf("%-18s n=%4d  best %8.2f GF/s  med %8.2f GF/s  %9.6f s  checksum=% .5e\n",
        "cob packed-B",
        n,
        best_gflops,
        median_gflops,
        stats.best,
        (double)checksum(c, n * n));

    cob_sgemm_free_packed_b(&packed_b);
    return best_gflops;
}

int main(int argc, char** argv)
{
    static const int default_sizes[] = {64, 128, 192, 256, 384, 512, 768, 1024};

    setenv("VECLIB_MAXIMUM_THREADS", "1", 1);
    setenv("ACCELERATE_MAXIMUM_THREADS", "1", 1);
    setenv("BLIS_NUM_THREADS", "1", 1);
    setenv("OPENBLAS_NUM_THREADS", "1", 1);
    setenv("OMP_NUM_THREADS", "1", 1);

    int sizes[32];
    int size_count = 0;
    if (argc > 1) {
        for (int i = 1; i < argc && size_count < 32; ++i) {
            const int n = atoi(argv[i]);
            if (n > 0) {
                sizes[size_count++] = n;
            }
        }
    } else {
        size_count = (int)(sizeof(default_sizes) / sizeof(default_sizes[0]));
        memcpy(sizes, default_sizes, sizeof(default_sizes));
    }

    printf("single-thread FP32 row-major square GEMM\n");
    printf("cob one-shot includes B packing; cob packed-B excludes B packing after setup\n\n");

    for (int si = 0; si < size_count; ++si) {
        const int n = sizes[si];
        const size_t count = (size_t)n * (size_t)n;
        float* a = alloc_f32_aligned(count);
        float* b = alloc_f32_aligned(count);
        float* c = alloc_f32_aligned(count);
        if (a == NULL || b == NULL || c == NULL) {
            fprintf(stderr, "allocation failed for n=%d\n", n);
            free(a);
            free(b);
            free(c);
            return 1;
        }

        uint32_t state = 0x9e3779b9u ^ (uint32_t)n;
        fill_random(a, (int)count, &state);
        fill_random(b, (int)count, &state);

        run_case("cob one-shot", bench_cob_direct, n, a, b, c);
        run_case_cob_packed_reuse(n, a, b, c);
#if defined(COB_HAVE_ACCELERATE)
        run_case("Accelerate", bench_accelerate, n, a, b, c);
#endif
#if defined(COB_HAVE_EXTERNAL_CBLAS)
        run_case(COB_EXTERNAL_CBLAS_NAME, bench_external_cblas, n, a, b, c);
#endif
#if defined(COB_HAVE_EXTERNAL_FORTRAN_SGEMM)
        run_case(COB_EXTERNAL_FORTRAN_SGEMM_NAME, bench_external_fortran_sgemm, n, a, b, c);
#endif
        printf("\n");

        free(a);
        free(b);
        free(c);
    }

    return 0;
}
