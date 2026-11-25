#define _GNU_SOURCE
#include <dlfcn.h>
#include <link.h>
#include <stdio.h>

typedef void (*gemm_fn)(int, int, int, const float *, int,
                        const float *, int, float *, int);
static gemm_fn multiply;

int rank_vendor_init(void)
{
    if (multiply) return 0;

    /* All vendor dependencies enter the same private glibc namespace, so
       an already loaded BLAS used by a competitor or reference oracle cannot
       interpose on either the user-visible symbol or the vendor's subcalls. */
    void *handle = dlmopen(LM_ID_NEWLM, RANK_BACKEND_PATH, RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        fprintf(stderr, "vendor namespace: %s\n", dlerror());
        return 1;
    }
    int (*initialize)(void) = (int (*)(void))dlsym(handle, "rank_backend_init");
    gemm_fn backend = (gemm_fn)dlsym(handle, "rank_backend_sgemm");
    if (!initialize || !backend || initialize() != 0) {
        dlclose(handle);
        return 1;
    }
    multiply = backend;
    return 0;
}

void rank_vendor_sgemm(int m, int n, int k, const float *a, int lda,
                       const float *b, int ldb, float *c, int ldc)
{
    multiply(m, n, k, a, lda, b, ldb, c, ldc);
}
