#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "cob_gemm.h"
#ifdef WIDE_MKL
#include <mkl.h>
#define OWNER "libmkl_intel_lp64.so.3"
_Static_assert(sizeof(MKL_INT)==sizeof(int),"oneMKL LP64 required");
#else
#include <cblas.h>
#define OWNER "libcob_rank_openblas.so"
#endif

static int same_owner(const char *symbol, void **base)
{
    Dl_info info;
    void *p=dlsym(RTLD_DEFAULT,symbol);
    if (!p || !dladdr(p,&info) || !info.dli_fname || !info.dli_fbase) return 0;
    const char *name=strrchr(info.dli_fname,'/');
    if (strcmp(name?name+1:info.dli_fname,OWNER)) return 0;
    if (*base && *base!=info.dli_fbase) return 0;
    *base=info.dli_fbase;
    return 1;
}
int wide_backend_init(void)
{
    void *base=NULL;
    if (!same_owner("cblas_sgemm",&base) || !same_owner("sgemm_",&base)) return 1;
#ifdef WIDE_MKL
    if (mkl_get_max_threads()!=1 || mkl_domain_get_max_threads(MKL_DOMAIN_BLAS)!=1) return 1;
#else
    if (!same_owner("openblas_get_config",&base) || !same_owner("openblas_get_num_threads",&base)
        || !same_owner("openblas_get_parallel",&base) || openblas_get_num_threads()!=1 ||
        openblas_get_parallel()!=0 || strncmp(openblas_get_config(),"OpenBLAS ",9) ||
        strstr(openblas_get_config(),"USE64BITINT")) return 1;
#endif
    fprintf(stderr,"qualified single-thread LP64 SGEMM binding %s\n",OWNER);
    return 0;
}

void wide_backend_gemm(int m,int n,int k,const float*a,int lda,const float*b,int ldb,float*c,int ldc)
{
#ifdef WIDE_DIRECT
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,n,k,1.0f,a,lda,b,ldb,0.0f,c,ldc);
#else
    cob_sgemm_rowmajor(m,n,k,a,lda,b,ldb,c,ldc);
#endif
}
