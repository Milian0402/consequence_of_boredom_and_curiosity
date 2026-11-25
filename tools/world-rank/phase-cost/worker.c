#if defined(__linux__)
#define _GNU_SOURCE
#endif
#define _POSIX_C_SOURCE 200809L
#include <dlfcn.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if defined(__linux__)
#include <dirent.h>
#include <errno.h>
#include <sched.h>
#endif

typedef void (*gemm_fn)(int, int, int, const float*, int, const float*, int, float*, int);
typedef void (*reset_fn)(void);
typedef void (*get_fn)(uint64_t[8]);
typedef struct { int m, n, k, pad; } shape;
enum { guard_size = 64, rounds = 6, calls = 3 };

static const uint32_t input_guard = UINT32_C(0x7fc12345);
static const uint32_t output_guard = UINT32_C(0xc7428000);

static void fail(const char* message)
{
    fprintf(stderr, "phase worker: %s\n", message);
    exit(EXIT_FAILURE);
}

#if defined(__linux__)
static void check_single_thread_and_cpu(void)
{
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) || CPU_COUNT(&allowed) != 1)
        fail("a singleton affinity mask is required");
    DIR* tasks = opendir("/proc/self/task");
    if (!tasks) fail("cannot inspect process threads");
    int count = 0;
    errno = 0;
    struct dirent* entry;
    while ((entry = readdir(tasks))) {
        if (entry->d_name[0] >= '0' && entry->d_name[0] <= '9') ++count;
    }
    const int read_error = errno;
    if (closedir(tasks) || read_error || count != 1) fail("expected exactly one thread");
    fprintf(stderr, "phase worker: affinity_cpus=1 tasks=%d\n", count);
}
#endif

static uint64_t now_ns(void)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts)) fail("monotonic clock unavailable");
    return (uint64_t)ts.tv_sec * UINT64_C(1000000000) + (uint64_t)ts.tv_nsec;
}

static float* matrix(int rows, int ld)
{
    void* p = NULL;
    if (posix_memalign(&p, 128, ((size_t)rows * ld + guard_size) * sizeof(float)))
        fail("matrix allocation failed");
    return p;
}

static void poison(float* data, int rows, int ld, uint32_t bits)
{
    for (size_t i = 0; i < (size_t)rows * ld + guard_size; ++i)
        memcpy(data + i, &bits, sizeof(bits));
}

static void check_guards(const float* data, int rows, int cols, int ld, uint32_t expected)
{
    for (int r = 0; r <= rows; ++r) {
        const int begin = r == rows ? 0 : cols;
        const int end = r == rows ? guard_size : ld;
        for (int j = begin; j < end; ++j) {
            uint32_t bits;
            memcpy(&bits, data + (size_t)r * ld + j, sizeof(bits));
            if (bits != expected) fail("matrix padding or trailing guard was overwritten");
        }
    }
}

static void fill(float* data, int rows, int cols, int ld, uint32_t* seed)
{
    poison(data, rows, ld, input_guard);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            *seed = *seed * UINT32_C(1664525) + UINT32_C(1013904223);
            data[(size_t)i * ld + j] = ((int)((*seed >> 8) % 2001u) - 1000) / 997.0f;
        }
    }
}

static void* open_library(const char* path)
{
    void* lib = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!lib) fail(dlerror());
    return lib;
}

static void load_function(void* lib, const char* name, void* target, size_t bytes)
{
    dlerror();
    void* symbol = dlsym(lib, name);
    const char* error = dlerror();
    if (error) fail(error);
    if (bytes != sizeof(symbol) || !symbol) fail("unsupported function pointer ABI");
    memcpy(target, &symbol, bytes);
}

static void run(shape s, const gemm_fn methods[2], reset_fn reset, get_fn get)
{
    const int lda = s.k + s.pad, ldb = s.n + s.pad, ldc = s.n + s.pad;
    float* a = matrix(s.m, lda);
    float* b = matrix(s.k, ldb);
    float* c[2] = {matrix(s.m, ldc), matrix(s.m, ldc)};
    uint32_t seed = UINT32_C(20271317);
    fill(a, s.m, s.k, lda, &seed);
    fill(b, s.k, s.n, ldb, &seed);
    for (int t = 0; t < 2; ++t) {
        poison(c[t], s.m, ldc, output_guard);
        for (int i = 0; i < s.m; ++i)
            for (int j = 0; j < s.n; ++j) c[t][(size_t)i * ldc + j] = NAN;
        methods[t](s.m, s.n, s.k, a, lda, b, ldb, c[t], ldc);
        check_guards(c[t], s.m, s.n, ldc, output_guard);
    }

    /* A contiguous double reference row lets compilers vectorize independent
       output columns while preserving increasing K order within each sum. */
    double* row = calloc((size_t)s.n, sizeof(*row));
    if (!row) fail("reference allocation failed");
    double maximum[2] = {0, 0};
    for (int i = 0; i < s.m; ++i) {
        memset(row, 0, (size_t)s.n * sizeof(*row));
        for (int p = 0; p < s.k; ++p) {
            const double left = a[(size_t)i * lda + p];
            const float* right = b + (size_t)p * ldb;
            for (int j = 0; j < s.n; ++j) row[j] += left * right[j];
        }
        for (int t = 0; t < 2; ++t) {
            for (int j = 0; j < s.n; ++j) {
                const double err = fabs((double)c[t][(size_t)i * ldc + j] - row[j]);
                if (!(err <= 0.002)) {
                    fprintf(stderr, "m=%d n=%d k=%d method=%d row=%d col=%d abs_error=%.9g\n",
                            s.m, s.n, s.k, t, i, j, err);
                    fail("FP64 comparison failed");
                }
                if (err > maximum[t]) maximum[t] = err;
            }
        }
    }
    free(row);
    check_guards(a, s.m, s.k, lda, input_guard);
    check_guards(b, s.k, s.n, ldb, input_guard);

    for (int round = 0; round < rounds; ++round) {
        uint64_t elapsed[2], phase[8] = {0};
        for (int order = 0; order < 2; ++order) {
            const int t = (round + order) % 2;
            /* Warm both separate output buffers equally. Exclude this call
               and the instrumentation reset/read from the measured batch. */
            methods[t](s.m, s.n, s.k, a, lda, b, ldb, c[t], ldc);
            if (t) reset();
            const uint64_t start = now_ns();
            for (int r = 0; r < calls; ++r)
                methods[t](s.m, s.n, s.k, a, lda, b, ldb, c[t], ldc);
            elapsed[t] = now_ns() - start;
            if (t) get(phase);
            if (elapsed[t] == 0) fail("zero elapsed time");
            check_guards(c[t], s.m, s.n, ldc, output_guard);
        }
        check_guards(a, s.m, s.k, lda, input_guard);
        check_guards(b, s.k, s.n, ldb, input_guard);
        if (!phase[0] || !phase[1] || !phase[2] || !phase[3] || !phase[4] ||
            !phase[5] || phase[6] != calls || !phase[7]) {
            fprintf(stderr, "m=%d n=%d k=%d round=%d route_calls=%" PRIu64
                    " nr=%" PRIu64 "\n", s.m, s.n, s.k, round + 1, phase[6], phase[7]);
            fail("empty phase or incomplete blocked route");
        }
        printf("%d,%d,%d,%d,%d,%s,%d,%" PRIu64 ",%" PRIu64,
               s.m, s.n, s.k, s.pad, round + 1, round % 2 ? "profile" : "control",
               calls, elapsed[0], elapsed[1]);
        for (int i = 0; i < 8; ++i) printf(",%" PRIu64, phase[i]);
        printf(",%.9g,%.9g\n", maximum[0], maximum[1]);
        if (fflush(stdout)) fail("raw CSV write failed");
    }
    free(c[1]); free(c[0]); free(b); free(a);
}

int main(int argc, char** argv)
{
    if (argc != 3) fail("usage: phase-worker CONTROL.so PROFILE.so");
    const shape cases[] = {
        {84,353,447,3}, {131,577,661,5}, {112,1744,357,0},
        {216,927,619,1}, {65,2161,529,7}, {128,4096,256,0}
    };
    void* libs[2] = {open_library(argv[1]), open_library(argv[2])};
    gemm_fn methods[2];
    reset_fn reset;
    get_fn get;
    for (int i = 0; i < 2; ++i)
        load_function(libs[i], "cob_sgemm_rowmajor", &methods[i], sizeof(methods[i]));
    load_function(libs[1], "phase_reset", &reset, sizeof(reset));
    load_function(libs[1], "phase_get", &get, sizeof(get));
#if defined(__linux__)
    check_single_thread_and_cpu();
#endif
    puts("m,n,k,pad,round,first,calls,control_ns,profile_ns,pack_a_ns,pack_b_ns,apply_ns,"
         "pack_a_calls,pack_b_calls,apply_calls,route_calls,nr,control_max_abs,profile_max_abs");
    for (size_t i = 0; i < sizeof(cases)/sizeof(cases[0]); ++i) run(cases[i], methods, reset, get);
    if (dlclose(libs[1]) || dlclose(libs[0])) fail("library close failed");
    return EXIT_SUCCESS;
}
