#include "vendor.h"
#include "cob_gemm.h"
int rank_vendor_init(void) { return 0; }
void rank_vendor_sgemm(int m,int n,int k,const float* a,int lda,const float* b,int ldb,float* c,int ldc)
{
    cob_sgemm_rowmajor(m,n,k,a,lda,b,ldb,c,ldc);
}
