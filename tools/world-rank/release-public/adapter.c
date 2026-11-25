/* Extra entry points for untimed provider and parity checks. The ranking
   executable links the production archive directly, without this adapter. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
#include "cob_gemm.h"

#if COB_ENABLE_OPENBLAS_LEAF
#include <cblas.h>
/* The pinned serial archive keeps its symbols local to the parity DSO. */
extern const char local_f_gemm[] __asm__("sgemm_");
#if RANK_TRACE
extern const char local_c_gemm[] __asm__("__real_cblas_sgemm");
#else
extern const char local_c_gemm[] __asm__("cblas_sgemm");
#endif
#elif COB_ENABLE_MKL_SEQUENTIAL_LEAF
#include <mkl.h>
#define EXPECTED "libmkl_intel_lp64.so"
#else
#define EXPECTED "libarmpl_lp64.so"
#endif

static const char *identity;
const char *rank_vendor_provider(void) { return identity; }

int rank_vendor_init(void)
{
    Dl_info c, f, self, entry;
    if (!dladdr(&identity, &self) || !dladdr((void *)cob_sgemm_rowmajor, &entry) ||
        entry.dli_fbase != self.dli_fbase) return 1;
#if COB_ENABLE_OPENBLAS_LEAF
    if (!dladdr(local_c_gemm, &c) || !dladdr(local_f_gemm, &f) ||
        c.dli_fbase != self.dli_fbase || f.dli_fbase != self.dli_fbase ||
        openblas_get_parallel() != 0 || openblas_get_num_threads() != 1) return 1;
    identity = openblas_get_config();
#else
    if (!dladdr(dlsym(RTLD_DEFAULT, "cblas_sgemm"), &c) ||
        !dladdr(dlsym(RTLD_DEFAULT, "sgemm_"), &f) || !c.dli_fname ||
        c.dli_fbase != f.dli_fbase || c.dli_fbase == self.dli_fbase) return 1;
    const char *name = strrchr(c.dli_fname, '/');
    name = name ? name + 1 : c.dli_fname;
    size_t prefix = sizeof(EXPECTED)-1;
    if (strncmp(name, EXPECTED, prefix) || (name[prefix] != '\0' && name[prefix] != '.')) return 1;
#if COB_ENABLE_MKL_SEQUENTIAL_LEAF
    if (mkl_get_max_threads() != 1 || mkl_domain_get_max_threads(MKL_DOMAIN_BLAS) != 1) return 1;
#endif
    identity = c.dli_fname;
#endif
    fprintf(stderr, "public=%s cblas_sgemm=%s sgemm_=%s provider=%s\n",
            entry.dli_fname,c.dli_fname,f.dli_fname,identity);
    return 0;
}

void rank_vendor_sgemm(int m,int n,int k,const float* a,int lda,const float* b,int ldb,float* c,int ldc)
{
    cob_sgemm_rowmajor(m,n,k,a,lda,b,ldb,c,ldc);
}
