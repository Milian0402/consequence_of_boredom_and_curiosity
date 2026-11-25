/* Separate process for observing explicit oneMKL plan setup and teardown. */
#define _GNU_SOURCE
#include <mkl_cblas.h>
#include <mkl_service.h>
#include <ctype.h>
#include <dlfcn.h>
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

_Static_assert(sizeof(MKL_INT) == 4, "LP64 oneMKL headers required");

struct observation {
    mkl_jit_status_t status;
    int callable, destroy_status;
    int64_t create_ns, get_ns, destroy_ns, clock_pair_ns;
};

static void fail(const char *message)
{
    fprintf(stderr, "%s\n", message);
    exit(1);
}

static int positive(const char *s)
{
    if (!isdigit((unsigned char)*s)) fail("Expected a positive integer");
    char *end = NULL;
    errno = 0;
    long value = strtol(s, &end, 10);
    if (errno || *end || value < 1 || value > INT_MAX) fail("Invalid integer");
    return (int)value;
}

static int64_t now_ns(void)
{
    struct timespec stamp;
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &stamp)) fail("CLOCK_MONOTONIC_RAW failed");
    return (int64_t)stamp.tv_sec * 1000000000 + stamp.tv_nsec;
}

static void check_thread(void)
{
    FILE *proc = fopen("/proc/self/status", "r");
    if (!proc) fail("Linux process status unavailable");
    char line[1024], extra;
    int threads = 0, pinned = 0;
    unsigned cpu;
    while (fgets(line, sizeof line, proc)) {
        if (sscanf(line, "Threads: %d", &threads) == 1) continue;
        if (!strncmp(line, "Cpus_allowed_list:", 18))
            pinned = sscanf(line + 18, " %u %c", &cpu, &extra) == 1;
    }
    fclose(proc);
    if (threads != 1 || !pinned || mkl_get_max_threads() != 1)
        fail("A pinned single-thread process and sequential oneMKL are required");
}

static void check_owner(const void *fn, const char *operation)
{
    Dl_info info;
    if (!dladdr(fn, &info) || !info.dli_fname) fail("Unresolved oneMKL function");
    const char *slash = strrchr(info.dli_fname, '/');
    const char *name = slash ? slash + 1 : info.dli_fname;
    if (strcmp(name, "libmkl_intel_lp64.so.3") && strcmp(name, "libmkl_core.so.3") &&
        strcmp(name, "libmkl_sequential.so.3")) fail("Unrecognized oneMKL DSO");
    fprintf(stderr, "plan-cost %s provider=%s\n", operation, name);
}

static const char *label(mkl_jit_status_t status)
{
    if (status == MKL_JIT_SUCCESS) return "MKL_JIT_SUCCESS";
    if (status == MKL_NO_JIT) return "MKL_NO_JIT";
    return "MKL_JIT_ERROR";
}

int main(int argc, char **argv)
{
    if (argc != 8) fail("Expected m n k lda ldb ldc samples");
    int m=positive(argv[1]), n=positive(argv[2]), k=positive(argv[3]);
    int lda=positive(argv[4]), ldb=positive(argv[5]), ldc=positive(argv[6]);
    int samples=positive(argv[7]);
    if (lda < k || ldb < n || ldc < n || samples > 4096) fail("Invalid plan extent or sample count");
    check_owner((const void *)mkl_jit_create_sgemm, "create");
    check_owner((const void *)mkl_jit_get_sgemm_ptr, "get");
    check_owner((const void *)mkl_jit_destroy, "destroy");
    check_thread();
    struct observation *raw = calloc((size_t)samples, sizeof *raw);
    if (!raw) fail("Cannot allocate observation storage");
    int valid = 1;
    for (int i = 0; i < samples; ++i) {
        struct observation *out = raw + i;
        void *handle = NULL;
        out->get_ns = out->destroy_ns = out->destroy_status = -1;
        int64_t start = now_ns();
        out->clock_pair_ns = now_ns() - start;
        start = now_ns();
        mkl_jit_status_t status = mkl_jit_create_sgemm(&handle, MKL_ROW_MAJOR, MKL_NOTRANS,
            MKL_NOTRANS, m, n, k, 1.0f, lda, ldb, 0.0f, ldc);
        out->create_ns = now_ns() - start;
        out->status = status;
        if ((status != MKL_JIT_SUCCESS && status != MKL_NO_JIT) || !handle) {
            valid = 0;
            continue;
        }
        start = now_ns();
        sgemm_jit_kernel_t kernel = mkl_jit_get_sgemm_ptr(handle);
        out->get_ns = now_ns() - start;
        out->callable = kernel != NULL;
        start = now_ns();
        status = mkl_jit_destroy(handle);
        out->destroy_ns = now_ns() - start;
        out->destroy_status = (int)status;
        if (!out->callable || status != MKL_JIT_SUCCESS) valid = 0;
    }
    check_thread();
    for (int i = 0; i < samples; ++i) {
        const struct observation *r = raw + i;
        printf("COST,%d,%s,%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 ",%d,%d\n",
            i, label(r->status), r->create_ns, r->get_ns, r->destroy_ns,
            r->clock_pair_ns, r->callable, r->destroy_status);
    }
    free(raw);
    return valid ? 0 : 1;
}
