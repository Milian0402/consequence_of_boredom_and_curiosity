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
static int base2,base512,packed2,packed512;
struct product { int m,n,k,lda,ldb,ldc; const float *a,*b,*c; };
static struct product current;
void __real_cob_sgemm_rowmajor(PRODUCT_ARGS);
void __wrap_cob_sgemm_rowmajor(PRODUCT_ARGS)
{
    ++public_calls;
    current=(struct product){m,n,k,lda,ldb,ldc,a,b,c};
    inside=1;
    refuse_once=getenv("FRONTIER_REFUSE_SCRATCH")!=NULL;
    __real_cob_sgemm_rowmajor(PRODUCT_VARS);
    inside=0;
}
int __real_posix_memalign(void **p,size_t a,size_t n);
int __wrap_posix_memalign(void **p,size_t a,size_t n)
{
    if (inside && refuse_once) {refuse_once=0; ++refused; return ENOMEM;}
    return __real_posix_memalign(p,a,n);
}
static void observe(KERNEL_ARGS)
{
    arg_ok &= inside && m==current.m && n==current.n && k==current.k &&
        lda==current.lda && ldb==current.ldb && ldc==current.ldc &&
        a==current.a && b==current.b && c==current.c;
    alignment_ok &= panel!=NULL && ((uintptr_t)panel & 63u)==0;
}
#define REDIRECT(symbol,counter) \
    void __real_##symbol(KERNEL_ARGS); \
    void __wrap_##symbol(KERNEL_ARGS) { \
        ++counter; observe(PRODUCT_VARS,panel); __real_##symbol(PRODUCT_VARS,panel); \
    }
REDIRECT(cob_stream_b_kernel_avx2,base2)
REDIRECT(cob_stream_b_kernel_avx512,base512)
#ifdef FRONTIER_PACKED
REDIRECT(cob_stream_b_packed_avx2,packed2)
REDIRECT(cob_stream_b_packed_avx512,packed512)
#endif
__attribute__((destructor)) static void report(void)
{
    printf("FRONTIER,public_calls=%d,avx2=%d,avx512=%d,packed_avx2=%d,packed_avx512=%d,args_ok=%d,alignment_ok=%d,refused=%d\n",
        public_calls,base2,base512,packed2,packed512,arg_ok,alignment_ok,refused);
}
