#include "cob_hk15_ragged.h"
#include <math.h>
#include <stddef.h>

/* Match the target's ordinary multiply-add convention and visit K in order. */
static inline float cob_hk15_madd(float a, float b, float c)
{
#if defined(__FMA__) || defined(__ARM_FEATURE_FMA)
    return fmaf(a, b, c);
#else
    return c + a*b;
#endif
}

/* The validated parent has already computed C[0:mc,0:nc] through kc.
   Its HK scratch is no longer live and has room for the k-by-(n%3) strip. */
void cob_hk15_ragged_finish(int m, int n, int k, const float* a, int lda,
    const float* b, int ldb, float* c, int ldc, float* workspace, cob_hk15_base_fn base)
{
    const int mc = m-m%2, nc = n-n%3, kc = k-k%3;
    if (kc < k) {
        const float* b0 = b+(size_t)kc*ldb;
        for (int i = 0; i < mc; ++i) {
            const float* ai = a+(size_t)i*lda+kc;
            float* ci = c+(size_t)i*ldc;
            if (k-kc == 1) {
                for (int j = 0; j < nc; ++j)
                    ci[j] = cob_hk15_madd(ai[0],b0[j],ci[j]);
            } else {
                const float* b1 = b0+ldb;
                for (int j = 0; j < nc; ++j) {
                    float first = cob_hk15_madd(ai[0],b0[j],ci[j]);
                    ci[j] = cob_hk15_madd(ai[1],b1[j],first);
                }
            }
        }
    }
    if (nc < n) {
        /* Keep one or two adjacent columns compact throughout the full-K
           right-strip product. A single pass reads them from the caller. */
        const int width = n-nc;
        for (int p = 0; p < k; ++p) {
            float* packed = workspace+(size_t)p*width;
            const float* row = b+(size_t)p*ldb+nc;
            packed[0] = row[0];
            if (width == 2) packed[1] = row[1];
        }
        base(mc,width,k,a,lda,workspace,width,c+nc,ldc);
    }
    if (mc < m)
        base(1,n,k,a+(size_t)mc*lda,lda,b,ldb,c+(size_t)mc*ldc,ldc);
}
