#define _DARWIN_C_SOURCE
#include "mtmm.h"
#include <dlfcn.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Same CLI and CSV schema as the exploratory harness. A and B may be read up
   to 64 floats beyond the logical allocation. All C padding must survive. */
enum { guard_floats = 64 };
static const float padding = -12345.0f;
static const double tolerance = 0.002;
typedef void (*gemm_fn)(int, int, int, const float *, int, const float *, int,
                        float *, int, const mtmm_config *);

static void fail(const char *reason)
{
    fprintf(stderr, "%s\n", reason);
    exit(EXIT_FAILURE);
}

static long number(const char *s, long min, long max)
{
    char *end;
    errno = 0;
    long v = strtol(s, &end, 10);
    if (errno || s == end || *end || v < min || v > max) fail("invalid numeric argument");
    return v;
}

static gemm_fn load(const char *path)
{
    void *lib = dlopen(path, RTLD_NOW | RTLD_LOCAL | RTLD_FIRST);
    if (!lib) fail(dlerror());
    void *sym = dlsym(lib, "mtmm_sgemm_rowmajor");
    if (!sym) fail(dlerror());
    gemm_fn fn;
    _Static_assert(sizeof(fn) == sizeof(sym), "function pointer size");
    memcpy(&fn, &sym, sizeof(fn));
    return fn;
}

static float *matrix(int rows, int ld)
{
    if ((size_t)rows > (SIZE_MAX / sizeof(float) - guard_floats) / (size_t)ld)
        fail("matrix too large");
    size_t count = (size_t)rows * ld;
    float *p = NULL;
    if (posix_memalign((void **)&p, 128, (count + guard_floats) * sizeof(*p)))
        fail("matrix allocation failed");
    memset(p + count, 0, guard_floats * sizeof(*p));
    return p;
}

static void fill(float *p, int rows, int cols, int ld, uint32_t *state)
{
    for (int i = 0; i < rows; ++i) {
        float *row = p + (size_t)i * ld;
        for (int j = 0; j < cols; ++j) {
            *state = 1664525u * *state + 1013904223u;
            row[j] = (float)((int)((*state >> 8) % 2001u) - 1000) / 997.0f;
        }
        for (int j = cols; j < ld; ++j) row[j] = padding;
    }
}

static void prepare_output(float *p, int rows, int cols, int ld)
{
    for (int i = 0; i < rows; ++i) {
        float *row = p + (size_t)i * ld;
        for (int j = 0; j < cols; ++j) row[j] = NAN;
        for (int j = cols; j < ld; ++j) row[j] = padding;
    }
}

static void check_guard(const float *p, size_t count)
{
    static const float zeros[guard_floats] = {0};
    if (memcmp(p + count, zeros, sizeof(zeros))) fail("allocation guard overwritten");
}

static void check_output_padding(const float *p, int rows, int cols, int ld)
{
    if (ld > cols) {
        for (int i = 0; i < rows; ++i)
            for (int j = cols; j < ld; ++j)
                if (p[(size_t)i * ld + j] != padding) fail("output row padding overwritten");
    }
    check_guard(p, (size_t)rows * ld);
}

static uint64_t now_ns(void)
{
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &t)) fail("clock_gettime failed");
    return (uint64_t)t.tv_sec * UINT64_C(1000000000) + (uint64_t)t.tv_nsec;
}

int main(int argc, char **argv)
{
    if (argc != 9) fail("usage: compare baseline candidate M N K 1 pairs stride_pad");
    gemm_fn f[] = {load(argv[1]), load(argv[2])};
    int m = number(argv[3], 1, INT_MAX), n = number(argv[4], 1, INT_MAX);
    int k = number(argv[5], 1, INT_MAX), threads = number(argv[6], 1, 1);
    int pairs = number(argv[7], 1, INT_MAX), pad = number(argv[8], 0, INT_MAX);
    if (k > INT_MAX - pad || n > INT_MAX - pad) fail("stride overflow");
    int lda = k + pad, ldb = n + pad, ldc = n + pad;
    mtmm_config config = {.num_threads = threads};
    uint32_t seed = 233u;
    const char *seed_env = getenv("COB_STUDY_SEED");
    if (seed_env) seed = (uint32_t)number(seed_env, 1, UINT32_MAX);
    float *a = matrix(m, lda), *b = matrix(k, ldb), *c = matrix(m, ldc), *ref = matrix(m, ldc);
    fill(a, m, k, lda, &seed); fill(b, k, n, ldb, &seed);
    prepare_output(c, m, n, ldc); prepare_output(ref, m, n, ldc);
    f[0](m, n, k, a, lda, b, ldb, ref, ldc, &config);
    f[1](m, n, k, a, lda, b, ldb, c, ldc, &config);
    check_output_padding(ref, m, n, ldc); check_output_padding(c, m, n, ldc);
    double max_diff = 0.0;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            size_t off = (size_t)i * ldc + j;
            if (!isfinite(ref[off]) || !isfinite(c[off])) fail("nonfinite or unwritten output");
            max_diff = fmax(max_diff, fabs((double)ref[off] - c[off]));
        }
    }
    if (max_diff > tolerance) {
        fprintf(stderr, "max absolute difference %.9g above %.9g\n", max_diff, tolerance);
        return 4;
    }
    const double operations = 2.0 * m * n * k;
    const unsigned calls = operations < 2e8 ? (unsigned)fmin(64.0, ceil(2e8 / operations)) : 1;
    for (int pair = 0; pair < pairs; ++pair) {
        uint64_t dt[2];
        for (int j = 0; j < 2; ++j) {
            int which = (pair & 1) ^ j;
            uint64_t start = now_ns();
            for (unsigned rep = 0; rep < calls; ++rep)
                f[which](m, n, k, a, lda, b, ldb, c, ldc, &config);
            dt[which] = now_ns() - start;
            if (!dt[which]) fail("clock resolution too low for this case");
            check_output_padding(c, m, n, ldc);
        }
        printf("%s,%d,%d,%d,%d,%d,%d,%.9f,%.9f,%.8f,%.9g\n", argv[2], m, n, k,
               threads, pad, pair + 1, (double)dt[0] * 1e-9 / calls, (double)dt[1] * 1e-9 / calls,
               (double)dt[0] / (double)dt[1], max_diff);
        fflush(stdout);
    }
    check_guard(a, (size_t)m * lda); check_guard(b, (size_t)k * ldb);
    free(ref); free(c); free(b); free(a);
    return 0;
}
