#if !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200112L
#endif

#include "cob_gemm.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void cob_a_sgemm_rowmajor(
    int m,
    int n,
    int k,
    const float* a,
    int lda,
    const float* b,
    int ldb,
    float* c,
    int ldc);
int cob_a_sgemm_pack_b(cob_packed_b_f32* packed, int k, int n, const float* b, int ldb);
void cob_a_sgemm_free_packed_b(cob_packed_b_f32* packed);
void cob_a_sgemm_rowmajor_packed_b(
    int m,
    int n,
    int k,
    const float* a,
    int lda,
    const cob_packed_b_f32* packed_b,
    float* c,
    int ldc);

void cob_b_sgemm_rowmajor(
    int m,
    int n,
    int k,
    const float* a,
    int lda,
    const float* b,
    int ldb,
    float* c,
    int ldc);
int cob_b_sgemm_pack_b(cob_packed_b_f32* packed, int k, int n, const float* b, int ldb);
void cob_b_sgemm_free_packed_b(cob_packed_b_f32* packed);
void cob_b_sgemm_rowmajor_packed_b(
    int m,
    int n,
    int k,
    const float* a,
    int lda,
    const cob_packed_b_f32* packed_b,
    float* c,
    int ldc);

typedef struct bench_shape {
    int m;
    int n;
    int k;
} bench_shape;

typedef struct impl_api {
    const char* name;
    void (*rowmajor)(int, int, int, const float*, int, const float*, int, float*, int);
    int (*pack_b)(cob_packed_b_f32*, int, int, const float*, int);
    void (*free_packed_b)(cob_packed_b_f32*);
    void (*rowmajor_packed_b)(int, int, int, const float*, int, const cob_packed_b_f32*, float*, int);
} impl_api;

typedef struct sample_stats {
    double mean;
    double median;
    double best;
    double cv_percent;
} sample_stats;

enum {
    COB_AB_MAX_REPEATS = 1001
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

static int default_iterations(bench_shape shape)
{
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

static sample_stats summarize_samples(const double* values, int count)
{
    double* sorted = (double*)malloc((size_t)count * sizeof(double));
    if (sorted == NULL) {
        sample_stats empty = {0.0, 0.0, 0.0, 0.0};
        return empty;
    }

    double sum = 0.0;
    for (int i = 0; i < count; ++i) {
        sorted[i] = values[i];
        sum += values[i];
    }
    sort_doubles(sorted, count);

    const double mean = sum / (double)count;
    double variance = 0.0;
    for (int i = 0; i < count; ++i) {
        const double delta = values[i] - mean;
        variance += delta * delta;
    }
    variance /= count > 1 ? (double)(count - 1) : 1.0;

    sample_stats stats;
    stats.mean = mean;
    stats.median = sorted[count / 2];
    stats.best = sorted[count - 1];
    stats.cv_percent = mean > 0.0 ? 100.0 * sqrt(variance) / mean : 0.0;

    free(sorted);
    return stats;
}

static int parse_shape(const char* text, bench_shape* shape)
{
    int m = 0;
    int n = 0;
    int k = 0;
    char extra = '\0';
    if (sscanf(text, "%dx%dx%d%c", &m, &n, &k, &extra) == 3 ||
        sscanf(text, "%d,%d,%d%c", &m, &n, &k, &extra) == 3) {
        if (m > 0 && n > 0 && k > 0) {
            shape->m = m;
            shape->n = n;
            shape->k = k;
            return 1;
        }
        return 0;
    }

    if (sscanf(text, "%d%c", &m, &extra) == 1 && m > 0) {
        shape->m = m;
        shape->n = m;
        shape->k = m;
        return 1;
    }
    return 0;
}

static double time_direct(const impl_api* impl, bench_shape shape, int iterations, const float* a, const float* b, float* c)
{
    const double t0 = now_seconds();
    for (int i = 0; i < iterations; ++i) {
        impl->rowmajor(shape.m, shape.n, shape.k, a, shape.k, b, shape.n, c, shape.n);
    }
    return now_seconds() - t0;
}

static double time_packed(
    const impl_api* impl,
    bench_shape shape,
    int iterations,
    const float* a,
    const cob_packed_b_f32* packed_b,
    float* c)
{
    const double t0 = now_seconds();
    for (int i = 0; i < iterations; ++i) {
        impl->rowmajor_packed_b(shape.m, shape.n, shape.k, a, shape.k, packed_b, c, shape.n);
    }
    return now_seconds() - t0;
}

static void bootstrap_log_speedup_ci(const double* log_speedups, int repeats, int draws, double* low, double* high)
{
    *low = 0.0;
    *high = 0.0;
    if (draws <= 0 || repeats <= 1) {
        return;
    }

    double* means = (double*)malloc((size_t)draws * sizeof(double));
    if (means == NULL) {
        return;
    }

    uint32_t rng = 0x9e3779b9u;
    for (int i = 0; i < draws; ++i) {
        double sum = 0.0;
        for (int j = 0; j < repeats; ++j) {
            const int pick = (int)(rng_next(&rng) % (uint32_t)repeats);
            sum += log_speedups[pick];
        }
        means[i] = sum / (double)repeats;
    }

    sort_doubles(means, draws);
    const int low_index = (draws * 25) / 1000;
    const int high_index = (draws * 975) / 1000;
    *low = exp(means[low_index]);
    *high = exp(means[high_index < draws ? high_index : draws - 1]);

    free(means);
}

static int bench_one_shape(bench_shape shape, int repeats, int iterations, int warmups, int packed_mode, int bootstrap_draws)
{
    const impl_api impl_a = {
        "A",
        cob_a_sgemm_rowmajor,
        cob_a_sgemm_pack_b,
        cob_a_sgemm_free_packed_b,
        cob_a_sgemm_rowmajor_packed_b,
    };
    const impl_api impl_b = {
        "B",
        cob_b_sgemm_rowmajor,
        cob_b_sgemm_pack_b,
        cob_b_sgemm_free_packed_b,
        cob_b_sgemm_rowmajor_packed_b,
    };

    const size_t a_count = (size_t)shape.m * (size_t)shape.k;
    const size_t b_count = (size_t)shape.k * (size_t)shape.n;
    const size_t c_count = (size_t)shape.m * (size_t)shape.n;

    float* a = alloc_f32_aligned(a_count);
    float* b = alloc_f32_aligned(b_count);
    float* c_a = alloc_f32_aligned(c_count);
    float* c_b = alloc_f32_aligned(c_count);
    double* times_a = (double*)malloc((size_t)repeats * sizeof(double));
    double* times_b = (double*)malloc((size_t)repeats * sizeof(double));
    double* gf_a = (double*)malloc((size_t)repeats * sizeof(double));
    double* gf_b = (double*)malloc((size_t)repeats * sizeof(double));
    double* speedups = (double*)malloc((size_t)repeats * sizeof(double));
    double* log_speedups = (double*)malloc((size_t)repeats * sizeof(double));

    if (a == NULL || b == NULL || c_a == NULL || c_b == NULL || times_a == NULL ||
        times_b == NULL || gf_a == NULL || gf_b == NULL || speedups == NULL ||
        log_speedups == NULL) {
        fprintf(stderr, "allocation failed for %dx%dx%d\n", shape.m, shape.n, shape.k);
        free(a);
        free(b);
        free(c_a);
        free(c_b);
        free(times_a);
        free(times_b);
        free(gf_a);
        free(gf_b);
        free(speedups);
        free(log_speedups);
        return 1;
    }

    uint32_t rng = 0x12345678u;
    fill_random(a, a_count, &rng);
    fill_random(b, b_count, &rng);
    memset(c_a, 0, c_count * sizeof(float));
    memset(c_b, 0, c_count * sizeof(float));

    cob_packed_b_f32 packed_a;
    cob_packed_b_f32 packed_b;
    memset(&packed_a, 0, sizeof(packed_a));
    memset(&packed_b, 0, sizeof(packed_b));
    if (packed_mode) {
        if (impl_a.pack_b(&packed_a, shape.k, shape.n, b, shape.n) != 0 ||
            impl_b.pack_b(&packed_b, shape.k, shape.n, b, shape.n) != 0) {
            fprintf(stderr, "packed-B setup failed for %dx%dx%d\n", shape.m, shape.n, shape.k);
            free(a);
            free(b);
            free(c_a);
            free(c_b);
            free(times_a);
            free(times_b);
            free(gf_a);
            free(gf_b);
            free(speedups);
            free(log_speedups);
            impl_a.free_packed_b(&packed_a);
            impl_b.free_packed_b(&packed_b);
            return 1;
        }
    }

    for (int i = 0; i < warmups; ++i) {
        if (packed_mode) {
            (void)time_packed(&impl_a, shape, 1, a, &packed_a, c_a);
            (void)time_packed(&impl_b, shape, 1, a, &packed_b, c_b);
        } else {
            (void)time_direct(&impl_a, shape, 1, a, b, c_a);
            (void)time_direct(&impl_b, shape, 1, a, b, c_b);
        }
    }

    const double ops = 2.0 * (double)shape.m * (double)shape.n * (double)shape.k * (double)iterations;
    int b_faster = 0;
    for (int r = 0; r < repeats; ++r) {
        if ((r & 1) == 0) {
            times_a[r] = packed_mode ? time_packed(&impl_a, shape, iterations, a, &packed_a, c_a)
                                     : time_direct(&impl_a, shape, iterations, a, b, c_a);
            times_b[r] = packed_mode ? time_packed(&impl_b, shape, iterations, a, &packed_b, c_b)
                                     : time_direct(&impl_b, shape, iterations, a, b, c_b);
        } else {
            times_b[r] = packed_mode ? time_packed(&impl_b, shape, iterations, a, &packed_b, c_b)
                                     : time_direct(&impl_b, shape, iterations, a, b, c_b);
            times_a[r] = packed_mode ? time_packed(&impl_a, shape, iterations, a, &packed_a, c_a)
                                     : time_direct(&impl_a, shape, iterations, a, b, c_a);
        }

        gf_a[r] = ops / times_a[r] * 1.0e-9;
        gf_b[r] = ops / times_b[r] * 1.0e-9;
        speedups[r] = times_a[r] / times_b[r];
        log_speedups[r] = log(speedups[r]);
        if (speedups[r] > 1.0) {
            ++b_faster;
        }
    }

    const sample_stats a_stats = summarize_samples(gf_a, repeats);
    const sample_stats b_stats = summarize_samples(gf_b, repeats);
    const sample_stats speedup_stats = summarize_samples(speedups, repeats);
    double mean_log = 0.0;
    for (int r = 0; r < repeats; ++r) {
        mean_log += log_speedups[r];
    }
    mean_log /= (double)repeats;

    double ci_low = 0.0;
    double ci_high = 0.0;
    bootstrap_log_speedup_ci(log_speedups, repeats, bootstrap_draws, &ci_low, &ci_high);

    printf(
        "%dx%dx%d mode=%s repeats=%d iterations=%d\n",
        shape.m,
        shape.n,
        shape.k,
        packed_mode ? "packed-b" : "one-shot",
        repeats,
        iterations);
    printf(
        "  A median %8.2f GF/s best %8.2f GF/s cv %5.2f%% checksum=% .6e\n",
        a_stats.median,
        a_stats.best,
        a_stats.cv_percent,
        checksum(c_a, c_count));
    printf(
        "  B median %8.2f GF/s best %8.2f GF/s cv %5.2f%% checksum=% .6e\n",
        b_stats.median,
        b_stats.best,
        b_stats.cv_percent,
        checksum(c_b, c_count));
    printf(
        "  paired B/A median %.4fx mean-log %.4fx",
        speedup_stats.median,
        exp(mean_log));
    if (ci_low > 0.0 && ci_high > 0.0) {
        printf(" bootstrap95 [%.4fx, %.4fx]", ci_low, ci_high);
    }
    printf(" B-faster %d/%d\n", b_faster, repeats);
    if (a_stats.cv_percent > 2.0 || b_stats.cv_percent > 2.0) {
        printf("  warning: sample CV above 2%%; treat this shape as noisy\n");
    }

    impl_a.free_packed_b(&packed_a);
    impl_b.free_packed_b(&packed_b);
    free(a);
    free(b);
    free(c_a);
    free(c_b);
    free(times_a);
    free(times_b);
    free(gf_a);
    free(gf_b);
    free(speedups);
    free(log_speedups);
    return 0;
}

int main(int argc, char** argv)
{
    const int repeats = env_int_clamped("COB_AB_REPEATS", 31, 1, COB_AB_MAX_REPEATS);
    const int warmups = env_int_clamped("COB_AB_WARMUPS", 2, 0, 100);
    const int bootstrap_draws = env_int_clamped("COB_AB_BOOTSTRAP", 2000, 0, 100000);
    const int forced_iterations = env_int_clamped("COB_AB_ITERS", 0, 0, 100000000);
    const char* mode = getenv("COB_AB_MODE");
    const int packed_mode = mode != NULL && strcmp(mode, "packed") == 0;

    bench_shape defaults[] = {
        {384, 384, 384},
        {768, 768, 768},
        {1024, 1024, 1024},
        {64, 4096, 7168},
    };

    int status = 0;
    if (argc <= 1) {
        const int count = (int)(sizeof(defaults) / sizeof(defaults[0]));
        for (int i = 0; i < count; ++i) {
            const int iterations = forced_iterations > 0 ? forced_iterations : default_iterations(defaults[i]);
            status |= bench_one_shape(defaults[i], repeats, iterations, warmups, packed_mode, bootstrap_draws);
        }
        return status;
    }

    for (int i = 1; i < argc; ++i) {
        bench_shape shape;
        if (!parse_shape(argv[i], &shape)) {
            fprintf(stderr, "invalid shape '%s' (use N, MxNxK, or M,N,K)\n", argv[i]);
            status = 1;
            continue;
        }
        const int iterations = forced_iterations > 0 ? forced_iterations : default_iterations(shape);
        status |= bench_one_shape(shape, repeats, iterations, warmups, packed_mode, bootstrap_draws);
    }
    return status;
}
