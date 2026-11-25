#include "cob_gemm.h"
#include <stdlib.h>
void mtmm_sgemm_rowmajor(int m,int n,int k,const float *a,int lda,const float *b,int ldb,float *c,int ldc,const int *threads)
{
    if (!threads || *threads!=1) abort();
    cob_sgemm_rowmajor(m,n,k,a,lda,b,ldb,c,ldc);
}
