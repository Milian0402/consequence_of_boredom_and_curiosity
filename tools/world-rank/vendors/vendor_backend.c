#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>

#if defined(RANK_INTEL)
#include <mkl.h>
#elif defined(RANK_ARM)
#include <armpl.h>
#else
#error Select one vendor implementation.
#endif

int rank_backend_init(void)
{
    Dl_info info;
    if (!dladdr((void *)cblas_sgemm, &info) || !info.dli_fname) return 1;
    const char *soname = strrchr(info.dli_fname, '/');
    soname = soname ? soname + 1 : info.dli_fname;
#if defined(RANK_INTEL)
    if (!strstr(soname, "libmkl_intel_lp64") || mkl_get_max_threads() != 1) {
        fprintf(stderr, "unexpected oneMKL SGEMM provider or thread count: %s\n", info.dli_fname);
        return 1;
    }
    char version[256] = {0};
    mkl_get_version_string(version, sizeof(version));
    fprintf(stderr, "oneMKL LP64 sequential: %s (%s)\n", version, info.dli_fname);
#else
    /* Arm's unmodified LP64 shared object is the sequential variant. */
    if (strncmp(soname, "libarmpl", 8) || strstr(soname, "_mp")) {
        fprintf(stderr, "unexpected ArmPL SGEMM provider: %s\n", info.dli_fname);
        return 1;
    }
    fprintf(stderr, "ArmPL 26.07 LP64 sequential: %s\n", info.dli_fname);
#endif
    return 0;
}

void rank_backend_sgemm(int m, int n, int k, const float *a, int lda,
                        const float *b, int ldb, float *c, int ldc)
{
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, 1.0f, a, lda, b, ldb, 0.0f, c, ldc);
}
