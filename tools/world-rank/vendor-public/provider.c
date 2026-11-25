#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
#include "cob_gemm.h"

#if COB_ENABLE_MKL_SEQUENTIAL_LEAF
#include <mkl.h>
#define EXPECTED "libmkl_intel_lp64.so.3"
#else
#include <armpl.h>
#define EXPECTED "libarmpl_lp64.so"
#endif

/* Runs in the same private ELF namespace as the public GEMM and its provider. */
static const char *loaded_provider;
const char* rank_backend_provider(void) { return loaded_provider; }

int rank_backend_init(void)
{
    Dl_info provider;
    if (!dladdr(dlsym(RTLD_DEFAULT, "cblas_sgemm"), &provider) || !provider.dli_fname) return 1;
    const char *base = strrchr(provider.dli_fname, '/');
    base = base ? base + 1 : provider.dli_fname;
    fprintf(stderr, "CBLAS provider %s\n", provider.dli_fname);
    if (strcmp(base, EXPECTED)) return 1;
#if COB_ENABLE_MKL_SEQUENTIAL_LEAF
    fprintf(stderr, "oneMKL maximum threads %d\n", mkl_get_max_threads());
    if (mkl_get_max_threads() != 1) return 1;
#endif
    loaded_provider = provider.dli_fname;
    return 0;
}

void rank_backend_sgemm(int m,int n,int k,const float *a,int lda,
                        const float *b,int ldb,float *c,int ldc)
{
    cob_sgemm_rowmajor(m,n,k,a,lda,b,ldb,c,ldc);
}
