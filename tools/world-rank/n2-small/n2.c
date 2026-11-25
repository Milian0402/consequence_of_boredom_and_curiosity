/* Direct FP32 GEMM for small Arm matrices. Define N2_NO_PLUGIN to use the
   guarded n2_try_sgemm entry as a route in a general-purpose multiplier. */
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>

#include "../../../src/cob_n2_small.inc"

int n2_try_sgemm(int m,int n,int k,const float *a,int lda,const float *b,int ldb,float *c,int ldc)
{
    return cob_n2_try_sgemm(m,n,k,a,lda,b,ldb,c,ldc);
}

#else
int n2_try_sgemm(int m,int n,int k,const float *a,int lda,const float *b,int ldb,float *c,int ldc)
{
    (void)m; (void)n; (void)k; (void)a; (void)lda; (void)b; (void)ldb; (void)c; (void)ldc;
    return 0;
}
#endif

#ifndef N2_NO_PLUGIN
int rank_vendor_init(void) {
#if defined(__aarch64__) && defined(__ARM_NEON)
    return 0;
#else
    return -1;
#endif
}
void rank_vendor_sgemm(int m,int n,int k,const float *a,int lda,const float *b,int ldb,float *c,int ldc)
{
    if (n2_try_sgemm(m,n,k,a,lda,b,ldb,c,ldc)) return;
    if (m<1 || n<1 || k<1 || !a || !b || !c || lda<k || ldb<n || ldc<n) return;
    for (int i=0; i<m; ++i)
        for (int j=0; j<n; ++j) {
            float sum=0;
            for (int p=0;p<k;++p) sum+=a[(size_t)i*lda+p]*b[(size_t)p*ldb+j];
            c[(size_t)i*ldc+j]=sum;
        }
}
#endif
