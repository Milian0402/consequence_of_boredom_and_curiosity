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

#if defined(__APPLE__)
#include <pthread.h>
#include <pthread/qos.h>
#endif

#if defined(COB_AB_B_ACCELERATE)
#include <Accelerate/Accelerate.h>
#endif

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
int cob_a_sgemm_pack_a(cob_packed_a_f32* packed, int m, int k, const float* a, int lda);
void cob_a_sgemm_free_packed_a(cob_packed_a_f32* packed);
void cob_a_sgemm_rowmajor_packed_b(
    int m,
    int n,
    int k,
    const float* a,
    int lda,
    const cob_packed_b_f32* packed_b,
    float* c,
    int ldc);
void cob_a_sgemm_rowmajor_packed_ab(
    int m,
    int n,
    int k,
    const cob_packed_a_f32* packed_a,
    const cob_packed_b_f32* packed_b,
    float* c,
    int ldc);

#if !defined(COB_AB_B_ACCELERATE)
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
int cob_b_sgemm_pack_a(cob_packed_a_f32* packed, int m, int k, const float* a, int lda);
void cob_b_sgemm_free_packed_a(cob_packed_a_f32* packed);
void cob_b_sgemm_rowmajor_packed_b(
    int m,
    int n,
    int k,
    const float* a,
    int lda,
    const cob_packed_b_f32* packed_b,
    float* c,
    int ldc);
void cob_b_sgemm_rowmajor_packed_ab(
    int m,
    int n,
    int k,
    const cob_packed_a_f32* packed_a,
    const cob_packed_b_f32* packed_b,
    float* c,
    int ldc);
#else
static void cob_b_sgemm_rowmajor(
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
    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        m,
        n,
        k,
        1.0f,
        a,
        lda,
        b,
        ldb,
        0.0f,
        c,
        ldc);
}

static int cob_b_sgemm_pack_b(
    cob_packed_b_f32* packed,
    int k,
    int n,
    const float* b,
    int ldb)
{
    (void)packed;
    (void)k;
    (void)n;
    (void)b;
    (void)ldb;
    return -1;
}

static void cob_b_sgemm_free_packed_b(cob_packed_b_f32* packed)
{
    (void)packed;
}

static int cob_b_sgemm_pack_a(
    cob_packed_a_f32* packed,
    int m,
    int k,
    const float* a,
    int lda)
{
    (void)packed;
    (void)m;
    (void)k;
    (void)a;
    (void)lda;
    return -1;
}

static void cob_b_sgemm_free_packed_a(cob_packed_a_f32* packed)
{
    (void)packed;
}

static void cob_b_sgemm_rowmajor_packed_b(
    int m,
    int n,
    int k,
    const float* a,
    int lda,
    const cob_packed_b_f32* packed_b,
    float* c,
    int ldc)
{
    (void)m;
    (void)n;
    (void)k;
    (void)a;
    (void)lda;
    (void)packed_b;
    (void)c;
    (void)ldc;
}

static void cob_b_sgemm_rowmajor_packed_ab(
    int m,
    int n,
    int k,
    const cob_packed_a_f32* packed_a,
    const cob_packed_b_f32* packed_b,
    float* c,
    int ldc)
{
    (void)m;
    (void)n;
    (void)k;
    (void)packed_a;
    (void)packed_b;
    (void)c;
    (void)ldc;
}
#endif

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
    int (*pack_a)(cob_packed_a_f32*, int, int, const float*, int);
    void (*free_packed_a)(cob_packed_a_f32*);
    void (*rowmajor_packed_b)(int, int, int, const float*, int, const cob_packed_b_f32*, float*, int);
    void (*rowmajor_packed_ab)(int, int, int, const cob_packed_a_f32*, const cob_packed_b_f32*, float*, int);
} impl_api;

typedef struct sample_stats {
    double mean;
    double median;
    double best;
    double cv_percent;
} sample_stats;

typedef struct paired_stats {
    sample_stats speedup;
    double mean_log_speedup;
    double ci_low;
    double ci_high;
    double sign_pvalue;
    int b_faster;
} paired_stats;

enum {
    COB_AB_MAX_REPEATS = 1001
};

typedef enum bench_mode {
    BENCH_MODE_DIRECT,
    BENCH_MODE_PACKED_B,
    BENCH_MODE_PACKED_AB
} bench_mode;

static void configure_benchmark_thread(void)
{
#if defined(__APPLE__) && defined(QOS_CLASS_USER_INTERACTIVE)
    (void)pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
}

static uint32_t rng_next(uint32_t* state)
{
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

static int rng_bounded(uint32_t* state, int bound)
{
    return (int)(((uint64_t)rng_next(state) * (uint64_t)(uint32_t)bound) >> 32);
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

static double env_double_clamped(const char* name, double fallback, double min_value, double max_value)
{
    const char* value = getenv(name);
    if (value == NULL || value[0] == '\0') {
        return fallback;
    }

    char* end = NULL;
    const double parsed = strtod(value, &end);
    if (end == value || *end != '\0' || !isfinite(parsed)) {
        return fallback;
    }
    if (parsed < min_value) {
        return min_value;
    }
    if (parsed > max_value) {
        return max_value;
    }
    return parsed;
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

static double time_packed_ab(
    const impl_api* impl,
    bench_shape shape,
    int iterations,
    const cob_packed_a_f32* packed_a,
    const cob_packed_b_f32* packed_b,
    float* c)
{
    const double t0 = now_seconds();
    for (int i = 0; i < iterations; ++i) {
        impl->rowmajor_packed_ab(shape.m, shape.n, shape.k, packed_a, packed_b, c, shape.n);
    }
    return now_seconds() - t0;
}

static const char* bench_mode_name(bench_mode mode)
{
    switch (mode) {
    case BENCH_MODE_PACKED_B:
        return "packed-b";
    case BENCH_MODE_PACKED_AB:
        return "packed-ab";
    case BENCH_MODE_DIRECT:
    default:
        return "one-shot";
    }
}

static double time_impl_sample(
    const impl_api* impl,
    bench_shape shape,
    int iterations,
    bench_mode mode,
    const float* a,
    const float* b,
    const cob_packed_a_f32* packed_a,
    const cob_packed_b_f32* packed_b,
    float* c)
{
    if (mode == BENCH_MODE_PACKED_AB) {
        return time_packed_ab(impl, shape, iterations, packed_a, packed_b, c);
    }
    if (mode == BENCH_MODE_PACKED_B) {
        return time_packed(impl, shape, iterations, a, packed_b, c);
    }
    return time_direct(impl, shape, iterations, a, b, c);
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
            const int pick = rng_bounded(&rng, repeats);
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

static double sign_test_two_tailed_pvalue(int successes, int trials)
{
    if (trials <= 0) {
        return 1.0;
    }

    int tail = successes;
    if (trials - successes < tail) {
        tail = trials - successes;
    }

    double probability = ldexp(1.0, -trials);
    double tail_sum = probability;
    for (int i = 0; i < tail; ++i) {
        probability *= (double)(trials - i) / (double)(i + 1);
        tail_sum += probability;
    }

    const double pvalue = 2.0 * tail_sum;
    return pvalue < 1.0 ? pvalue : 1.0;
}

static int checksums_match(float checksum_a, float checksum_b)
{
    const double a = fabs((double)checksum_a);
    const double b = fabs((double)checksum_b);
    double scale = a > b ? a : b;
    if (scale < 1.0) {
        scale = 1.0;
    }
    return fabs((double)checksum_a - (double)checksum_b) / scale <= 1.0e-4;
}

static paired_stats summarize_paired(
    const double* speedups,
    const double* log_speedups,
    int repeats,
    int bootstrap_draws)
{
    paired_stats stats;
    stats.speedup = summarize_samples(speedups, repeats);
    stats.mean_log_speedup = 0.0;
    stats.ci_low = 0.0;
    stats.ci_high = 0.0;
    stats.sign_pvalue = 1.0;
    stats.b_faster = 0;

    for (int r = 0; r < repeats; ++r) {
        stats.mean_log_speedup += log_speedups[r];
        if (speedups[r] > 1.0) {
            ++stats.b_faster;
        }
    }
    stats.mean_log_speedup /= (double)repeats;
    bootstrap_log_speedup_ci(
        log_speedups,
        repeats,
        bootstrap_draws,
        &stats.ci_low,
        &stats.ci_high);
    stats.sign_pvalue = sign_test_two_tailed_pvalue(stats.b_faster, repeats);
    return stats;
}

static int bench_one_shape(
    bench_shape shape,
    int min_repeats,
    int max_repeats,
    int repeat_batch,
    double cv_target_percent,
    int iterations,
    int warmups,
    bench_mode mode,
    int bootstrap_draws,
    int cooldown_us,
    int holdout_reporting)
{
    const impl_api impl_a = {
#if defined(COB_AB_B_ACCELERATE)
        "COB",
#else
        "A",
#endif
        cob_a_sgemm_rowmajor,
        cob_a_sgemm_pack_b,
        cob_a_sgemm_free_packed_b,
        cob_a_sgemm_pack_a,
        cob_a_sgemm_free_packed_a,
        cob_a_sgemm_rowmajor_packed_b,
        cob_a_sgemm_rowmajor_packed_ab,
    };
    const impl_api impl_b = {
#if defined(COB_AB_B_ACCELERATE)
        "Accelerate",
#else
        "B",
#endif
        cob_b_sgemm_rowmajor,
        cob_b_sgemm_pack_b,
        cob_b_sgemm_free_packed_b,
        cob_b_sgemm_pack_a,
        cob_b_sgemm_free_packed_a,
        cob_b_sgemm_rowmajor_packed_b,
        cob_b_sgemm_rowmajor_packed_ab,
    };

    const size_t a_count = (size_t)shape.m * (size_t)shape.k;
    const size_t b_count = (size_t)shape.k * (size_t)shape.n;
    const size_t c_count = (size_t)shape.m * (size_t)shape.n;

    float* a = alloc_f32_aligned(a_count);
    float* b = alloc_f32_aligned(b_count);
    float* c_a = alloc_f32_aligned(c_count);
    float* c_b = alloc_f32_aligned(c_count);
    double* times_a = (double*)malloc((size_t)max_repeats * sizeof(double));
    double* times_b = (double*)malloc((size_t)max_repeats * sizeof(double));
    double* gf_a = (double*)malloc((size_t)max_repeats * sizeof(double));
    double* gf_b = (double*)malloc((size_t)max_repeats * sizeof(double));
    double* speedups = (double*)malloc((size_t)max_repeats * sizeof(double));
    double* log_speedups = (double*)malloc((size_t)max_repeats * sizeof(double));

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

    cob_packed_a_f32 packed_a_a;
    cob_packed_a_f32 packed_a_b;
    cob_packed_b_f32 packed_b_a;
    cob_packed_b_f32 packed_b_b;
    memset(&packed_a_a, 0, sizeof(packed_a_a));
    memset(&packed_a_b, 0, sizeof(packed_a_b));
    memset(&packed_b_a, 0, sizeof(packed_b_a));
    memset(&packed_b_b, 0, sizeof(packed_b_b));
    if (mode == BENCH_MODE_PACKED_B || mode == BENCH_MODE_PACKED_AB) {
        if (impl_a.pack_b(&packed_b_a, shape.k, shape.n, b, shape.n) != 0 ||
            impl_b.pack_b(&packed_b_b, shape.k, shape.n, b, shape.n) != 0) {
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
            impl_a.free_packed_b(&packed_b_a);
            impl_b.free_packed_b(&packed_b_b);
            return 1;
        }
    }
    if (mode == BENCH_MODE_PACKED_AB) {
        if (impl_a.pack_a(&packed_a_a, shape.m, shape.k, a, shape.k) != 0 ||
            impl_b.pack_a(&packed_a_b, shape.m, shape.k, a, shape.k) != 0) {
            fprintf(stderr, "packed-A setup failed for %dx%dx%d\n", shape.m, shape.n, shape.k);
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
            impl_a.free_packed_a(&packed_a_a);
            impl_b.free_packed_a(&packed_a_b);
            impl_a.free_packed_b(&packed_b_a);
            impl_b.free_packed_b(&packed_b_b);
            return 1;
        }
    }

    for (int i = 0; i < warmups; ++i) {
        if (mode == BENCH_MODE_PACKED_AB) {
            (void)time_packed_ab(&impl_a, shape, 1, &packed_a_a, &packed_b_a, c_a);
            (void)time_packed_ab(&impl_b, shape, 1, &packed_a_b, &packed_b_b, c_b);
        } else if (mode == BENCH_MODE_PACKED_B) {
            (void)time_packed(&impl_a, shape, 1, a, &packed_b_a, c_a);
            (void)time_packed(&impl_b, shape, 1, a, &packed_b_b, c_b);
        } else {
            (void)time_direct(&impl_a, shape, 1, a, b, c_a);
            (void)time_direct(&impl_b, shape, 1, a, b, c_b);
        }
    }

    const double ops = 2.0 * (double)shape.m * (double)shape.n * (double)shape.k * (double)iterations;
    int repeats = 0;
    sample_stats a_stats = {0.0, 0.0, 0.0, 0.0};
    sample_stats b_stats = {0.0, 0.0, 0.0, 0.0};
    sample_stats speedup_stats = {0.0, 0.0, 0.0, 0.0};
    while (repeats < max_repeats) {
        int target_repeats = repeats == 0 ? min_repeats : repeats + repeat_batch;
        if (target_repeats > max_repeats) {
            target_repeats = max_repeats;
        }
        for (int r = repeats; r < target_repeats; ++r) {
            if ((r & 1) == 0) {
                times_a[r] =
                    time_impl_sample(&impl_a, shape, iterations, mode, a, b, &packed_a_a, &packed_b_a, c_a);
                times_b[r] =
                    time_impl_sample(&impl_b, shape, iterations, mode, a, b, &packed_a_b, &packed_b_b, c_b);
            } else {
                times_b[r] =
                    time_impl_sample(&impl_b, shape, iterations, mode, a, b, &packed_a_b, &packed_b_b, c_b);
                times_a[r] =
                    time_impl_sample(&impl_a, shape, iterations, mode, a, b, &packed_a_a, &packed_b_a, c_a);
            }

            gf_a[r] = ops / times_a[r] * 1.0e-9;
            gf_b[r] = ops / times_b[r] * 1.0e-9;
            speedups[r] = times_a[r] / times_b[r];
            log_speedups[r] = log(speedups[r]);
            sleep_microseconds(cooldown_us);
        }

        repeats = target_repeats;
        a_stats = summarize_samples(gf_a, repeats);
        b_stats = summarize_samples(gf_b, repeats);
        speedup_stats = summarize_samples(speedups, repeats);
        if (repeats >= min_repeats &&
            ((a_stats.cv_percent <= cv_target_percent && b_stats.cv_percent <= cv_target_percent) ||
                speedup_stats.cv_percent <= cv_target_percent)) {
            break;
        }
    }

    const paired_stats all_stats = summarize_paired(speedups, log_speedups, repeats, bootstrap_draws);
    const float checksum_a = checksum(c_a, c_count);
    const float checksum_b = checksum(c_b, c_count);
    const int checksum_ok = checksums_match(checksum_a, checksum_b);

    printf(
        "%dx%dx%d mode=%s repeats=%d iterations=%d\n",
        shape.m,
        shape.n,
        shape.k,
        bench_mode_name(mode),
        repeats,
        iterations);
    if (cooldown_us > 0) {
        printf("  cooldown: %d us after each paired sample\n", cooldown_us);
    }
    if (repeats > min_repeats) {
        printf(
            "  auto-repeats: requested %d, max %d, batch %d, cv target %.2f%%\n",
            min_repeats,
            max_repeats,
            repeat_batch,
            cv_target_percent);
    }
    printf(
        "  %s median %8.2f GF/s best %8.2f GF/s cv %5.2f%% checksum=% .6e\n",
        impl_a.name,
        a_stats.median,
        a_stats.best,
        a_stats.cv_percent,
        checksum_a);
    printf(
        "  %s median %8.2f GF/s best %8.2f GF/s cv %5.2f%% checksum=% .6e\n",
        impl_b.name,
        b_stats.median,
        b_stats.best,
        b_stats.cv_percent,
        checksum_b);
    printf(
        "  paired %s/%s median %.4fx mean-log %.4fx cv %5.2f%%",
        impl_b.name,
        impl_a.name,
        all_stats.speedup.median,
        exp(all_stats.mean_log_speedup),
        all_stats.speedup.cv_percent);
    if (all_stats.ci_low > 0.0 && all_stats.ci_high > 0.0) {
        printf(" bootstrap95 [%.4fx, %.4fx]", all_stats.ci_low, all_stats.ci_high);
    }
    printf(
        " %s-faster %d/%d sign-p %.3g\n",
        impl_b.name,
        all_stats.b_faster,
        repeats,
        all_stats.sign_pvalue);
    if (holdout_reporting && repeats >= 8) {
        const int split = repeats / 2;
        const int holdout_count = repeats - split;
        const paired_stats holdout_stats = summarize_paired(
            speedups + split,
            log_speedups + split,
            holdout_count,
            bootstrap_draws);
        printf(
            "  split-half holdout %s/%s median %.4fx mean-log %.4fx cv %5.2f%%",
            impl_b.name,
            impl_a.name,
            holdout_stats.speedup.median,
            exp(holdout_stats.mean_log_speedup),
            holdout_stats.speedup.cv_percent);
        if (holdout_stats.ci_low > 0.0 && holdout_stats.ci_high > 0.0) {
            printf(
                " bootstrap95 [%.4fx, %.4fx]",
                holdout_stats.ci_low,
                holdout_stats.ci_high);
        }
        printf(
            " %s-faster %d/%d sign-p %.3g\n",
            impl_b.name,
            holdout_stats.b_faster,
            holdout_count,
            holdout_stats.sign_pvalue);
    }
    if ((a_stats.cv_percent > cv_target_percent || b_stats.cv_percent > cv_target_percent) &&
        all_stats.speedup.cv_percent > cv_target_percent) {
        printf(
            "  warning: sample CV above %.2f%%; treat this shape as noisy\n",
            cv_target_percent);
    } else if (a_stats.cv_percent > cv_target_percent || b_stats.cv_percent > cv_target_percent) {
        printf(
            "  note: implementation CV above %.2f%%, but paired speedup CV met target\n",
            cv_target_percent);
    }
    if (!checksum_ok) {
        fprintf(
            stderr,
            "checksum mismatch for %dx%dx%d: A=% .6e B=% .6e\n",
            shape.m,
            shape.n,
            shape.k,
            checksum_a,
            checksum_b);
    }

    impl_a.free_packed_a(&packed_a_a);
    impl_b.free_packed_a(&packed_a_b);
    impl_a.free_packed_b(&packed_b_a);
    impl_b.free_packed_b(&packed_b_b);
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
    return checksum_ok ? 0 : 2;
}

int main(int argc, char** argv)
{
    configure_benchmark_thread();

    const int repeats = env_int_clamped("COB_AB_REPEATS", 31, 1, COB_AB_MAX_REPEATS);
    int max_repeats = env_int_clamped("COB_AB_MAX_REPEATS", repeats, 1, COB_AB_MAX_REPEATS);
    if (max_repeats < repeats) {
        max_repeats = repeats;
    }
    int repeat_batch = env_int_clamped("COB_AB_REPEAT_BATCH", repeats, 1, COB_AB_MAX_REPEATS);
    if (repeat_batch > max_repeats) {
        repeat_batch = max_repeats;
    }
    const double cv_target_percent = env_double_clamped("COB_AB_CV_TARGET", 2.0, 0.1, 100.0);
    const int warmups = env_int_clamped("COB_AB_WARMUPS", 2, 0, 100);
    const int bootstrap_draws = env_int_clamped("COB_AB_BOOTSTRAP", 2000, 0, 100000);
    const int forced_iterations = env_int_clamped("COB_AB_ITERS", 0, 0, 100000000);
    const int cooldown_us = env_int_clamped("COB_AB_COOLDOWN_US", 0, 0, 10000000);
    const int holdout_reporting = env_int_clamped("COB_AB_HOLDOUT", 1, 0, 1);
    const char* mode = getenv("COB_AB_MODE");
    bench_mode mode_kind = BENCH_MODE_DIRECT;
    if (mode != NULL) {
        if (strcmp(mode, "packed") == 0 || strcmp(mode, "packed-B") == 0 ||
            strcmp(mode, "packed_b") == 0) {
            mode_kind = BENCH_MODE_PACKED_B;
        } else if (strcmp(mode, "packed-AB") == 0 || strcmp(mode, "packed_ab") == 0) {
            mode_kind = BENCH_MODE_PACKED_AB;
        }
    }
#if defined(COB_AB_B_ACCELERATE)
    if (mode_kind != BENCH_MODE_DIRECT) {
        fprintf(stderr, "Accelerate paired benchmark supports only direct one-shot mode\n");
        return 2;
    }
#endif

    bench_shape defaults[] = {
        {384, 384, 384},
        {768, 768, 768},
        {1024, 1024, 1024},
        {1536, 1536, 1536},
        {64, 4096, 7168},
        {128, 4096, 1024},
    };

    int status = 0;
    if (argc <= 1) {
        const int count = (int)(sizeof(defaults) / sizeof(defaults[0]));
        for (int i = 0; i < count; ++i) {
            const int iterations = forced_iterations > 0 ? forced_iterations : default_iterations(defaults[i]);
            status |= bench_one_shape(
                defaults[i],
                repeats,
                max_repeats,
                repeat_batch,
                cv_target_percent,
                iterations,
                warmups,
                mode_kind,
                bootstrap_draws,
                cooldown_us,
                holdout_reporting);
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
        status |= bench_one_shape(
            shape,
            repeats,
            max_repeats,
            repeat_batch,
            cv_target_percent,
            iterations,
            warmups,
            mode_kind,
            bootstrap_draws,
            cooldown_us,
            holdout_reporting);
    }
    return status;
}
