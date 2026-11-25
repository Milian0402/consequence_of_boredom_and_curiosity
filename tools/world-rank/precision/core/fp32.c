#include "precision.h"
#include <stddef.h>

extern void cob_sgemm_rowmajor(int,int,int,const float*,int,const float*,int,float*,int);

void precision_fp32_control(int m,int n,int k,const float *a,int lda,const float *b,int ldb,float *c,int ldc)
{ cob_sgemm_rowmajor(m,n,k,a,lda,b,ldb,c,ldc); }

/* IEEE binary32 products are exact in binary64. For valid int-sized K, even
   |a|=|b|=FLT_MAX cannot overflow a finite-input binary64 partial sum.
   Summation is still rounded to binary64 and the final cast is to binary32. */
void precision_range_rescue(int m,int n,int k,const float *a,int lda,const float *b,int ldb,float *c,int ldc)
{
    for(int i=0;i<m;++i) for(int j=0;j<n;++j){
        double v=0.0;
        for(int p=0;p<k;++p) v+=(double)a[(size_t)i*lda+p]*(double)b[(size_t)p*ldb+j];
        c[(size_t)i*ldc+j]=(float)v;
    }
}
