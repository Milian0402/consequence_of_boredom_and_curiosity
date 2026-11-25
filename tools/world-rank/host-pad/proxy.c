/* Untimed observer at the underlying provider's ABI boundary. */
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef void (*sgemm_fn)(int,int,int,const float *,int,const float *,int,float *,int);
static sgemm_fn real_gemm;
static int seen[7];
int rank_vendor_init(void)
{
    char path[4096];
    const char *dir=getenv("RANK_WITNESS_BLAS_DIR");
    if (!dir) return 1;
    int n=snprintf(path,sizeof path,"%s/%s.so",dir,HOST_PROVIDER);
    if (n<0 || (size_t)n>=sizeof path) return 1;
    void *h=dlopen(path,RTLD_NOW|RTLD_LOCAL);
    if (!h) return 1;
    int (*init)(void)=(int (*)(void))dlsym(h,"rank_vendor_init");
    real_gemm=(sgemm_fn)dlsym(h,"rank_vendor_sgemm");
    return !init || !real_gemm || init();
}
void rank_vendor_sgemm(int m,int n,int k,const float *a,int lda,const float *b,int ldb,float *c,int ldc)
{
    seen[0]++; seen[1]=m;seen[2]=n;seen[3]=k;seen[4]=lda;seen[5]=ldb;seen[6]=ldc;
    real_gemm(m,n,k,a,lda,b,ldb,c,ldc);
}
void rank_observed_call(int *out) { memcpy(out,seen,sizeof seen); }
