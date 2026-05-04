#include "cob_gemm.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(COB_HAVE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#endif

typedef void (*bench_fn)(int n, const float* a, const float* b, float* c);

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

static double run_case(const char* name, bench_fn fn, int n, const float* a, const float* b, float* c)
{
    const int warmups = n <= 256 ? 2 : 1;
    const int repeats = n <= 256 ? 7 : 4;
    double best = 1.0e100;

    for (int i = 0; i < warmups; ++i) {
        memset(c, 0, (size_t)n * (size_t)n * sizeof(float));
        fn(n, a, b, c);
    }

    for (int i = 0; i < repeats; ++i) {
        memset(c, 0, (size_t)n * (size_t)n * sizeof(float));
        const double t0 = now_seconds();
        fn(n, a, b, c);
        const double t1 = now_seconds();
        const double elapsed = t1 - t0;
        if (elapsed < best) {
            best = elapsed;
        }
    }

    const double gflops = (2.0 * (double)n * (double)n * (double)n) / best / 1.0e9;
    printf("%-18s n=%4d  %8.2f GF/s  %9.6f s  checksum=% .5e\n",
        name,
        n,
        gflops,
        best,
        (double)checksum(c, n * n));
    return gflops;
}

static double run_case_cob_packed_reuse(int n, const float* a, const float* b, float* c)
{
    cob_packed_b_f32 packed_b;
    if (cob_sgemm_pack_b(&packed_b, n, n, b, n) != 0) {
        fprintf(stderr, "packed-B allocation failed for n=%d\n", n);
        return 0.0;
    }

    const int warmups = n <= 256 ? 2 : 1;
    const int repeats = n <= 256 ? 7 : 4;
    double best = 1.0e100;

    for (int i = 0; i < warmups; ++i) {
        memset(c, 0, (size_t)n * (size_t)n * sizeof(float));
        cob_sgemm_rowmajor_packed_b(n, n, n, a, n, &packed_b, c, n);
    }

    for (int i = 0; i < repeats; ++i) {
        memset(c, 0, (size_t)n * (size_t)n * sizeof(float));
        const double t0 = now_seconds();
        cob_sgemm_rowmajor_packed_b(n, n, n, a, n, &packed_b, c, n);
        const double t1 = now_seconds();
        const double elapsed = t1 - t0;
        if (elapsed < best) {
            best = elapsed;
        }
    }

    const double gflops = (2.0 * (double)n * (double)n * (double)n) / best / 1.0e9;
    printf("%-18s n=%4d  %8.2f GF/s  %9.6f s  checksum=% .5e\n",
        "cob packed-B",
        n,
        gflops,
        best,
        (double)checksum(c, n * n));

    cob_sgemm_free_packed_b(&packed_b);
    return gflops;
}

int main(int argc, char** argv)
{
    static const int default_sizes[] = {64, 128, 192, 256, 384, 512, 768, 1024};

    setenv("VECLIB_MAXIMUM_THREADS", "1", 1);

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
        float* a = (float*)malloc(count * sizeof(float));
        float* b = (float*)malloc(count * sizeof(float));
        float* c = (float*)malloc(count * sizeof(float));
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
        printf("\n");

        free(a);
        free(b);
        free(c);
    }

    return 0;
}
