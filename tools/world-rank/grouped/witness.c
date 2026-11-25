/* Test-only wrappers forward to the exact generic-dispatch/ISA objects from
   the installed archive. The timed executable is linked without this file. */
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define PRODUCT_ARGS int m,int n,int k,const float *a,int lda,const float *b,int ldb,float *c,int ldc
#define PRODUCT_VARS m,n,k,a,lda,b,ldb,c,ldc
#define KERNEL_ARGS PRODUCT_ARGS,float *panel
static int public_calls,inside,refuse_once,refused,arg_ok=1,alignment_ok=1;
static int base2,base512,group2,group512;
static int first_align;
static size_t first_bytes;
static uintptr_t first_workspace;
struct product { int m,n,k,lda,ldb,ldc; const float *a,*b,*c; };
static struct product current;
void __real_cob_sgemm_rowmajor(PRODUCT_ARGS);
void __wrap_cob_sgemm_rowmajor(PRODUCT_ARGS)
{
    ++public_calls;
    current=(struct product){m,n,k,lda,ldb,ldc,a,b,c};
    inside=1;
    refuse_once=getenv("GROUPED_REFUSE_SCRATCH")!=NULL;
    __real_cob_sgemm_rowmajor(PRODUCT_VARS);
    inside=0;
}
int __real_posix_memalign(void **p,size_t a,size_t n);
int __wrap_posix_memalign(void **p,size_t a,size_t n)
{
    const int first = inside && !first_bytes;
    if (first) {first_bytes=n; first_align=(int)a;}
    if (inside && refuse_once) {refuse_once=0; ++refused; return ENOMEM;}
    int result=__real_posix_memalign(p,a,n);
    if (first && !result) first_workspace=(uintptr_t)*p;
    return result;
}
static void observe(KERNEL_ARGS)
{
    arg_ok &= inside && m==current.m && n==current.n && k==current.k &&
        lda==current.lda && ldb==current.ldb && ldc==current.ldc &&
        a==current.a && b==current.b && c==current.c;
    alignment_ok &= panel!=NULL && ((uintptr_t)panel & 63u)==0 && first_workspace==(uintptr_t)panel;
}
#define REDIRECT(symbol,counter) \
    void __real_##symbol(KERNEL_ARGS); \
    void __wrap_##symbol(KERNEL_ARGS) { \
        ++counter; observe(PRODUCT_VARS,panel); __real_##symbol(PRODUCT_VARS,panel); \
    }
REDIRECT(cob_stream_b_kernel_avx2,base2)
REDIRECT(cob_stream_b_kernel_avx512,base512)
#ifdef GROUPED_ENABLED
REDIRECT(cob_stream_b_grouped_avx2,group2)
REDIRECT(cob_stream_b_grouped_avx512,group512)
#endif
__attribute__((destructor)) static void report(void)
{
    printf("GROUPED,public_calls=%d,avx2=%d,avx512=%d,grouped_avx2=%d,grouped_avx512=%d,args_ok=%d,alignment_ok=%d,refused=%d,first_bytes=%zu,first_align=%d,m=%d,n=%d,k=%d,lda=%d,ldb=%d,ldc=%d,a_mod64=%u,b_mod64=%u,c_mod64=%u\n",
        public_calls,base2,base512,group2,group512,arg_ok,alignment_ok,refused,first_bytes,first_align,current.m,current.n,current.k,current.lda,current.ldb,current.ldc,
        (unsigned)((uintptr_t)current.a%64),(unsigned)((uintptr_t)current.b%64),(unsigned)((uintptr_t)current.c%64));
}
