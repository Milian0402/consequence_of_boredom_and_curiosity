#include <cblas.h>
#include <stdio.h>
#include "vendor.h"

int rank_vendor_init(void)
{
    openblas_set_num_threads(1);
    fprintf(stderr,"openblas_core=%s openblas_configuration=%s threads=%d\n",
            openblas_get_corename(),openblas_get_config(),openblas_get_num_threads());
    return openblas_get_num_threads() == 1 ? 0 : 1;
}

void rank_vendor_sgemm(int m,int n,int k,const float* a,int lda,const float* b,int ldb,float* c,int ldc)
{
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,n,k,1.0f,a,lda,b,ldb,0.0f,c,ldc);
}

void rank_vendor_dgemm(int m,int n,int k,const double* a,int lda,const double* b,int ldb,double* c,int ldc)
{
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,n,k,1.0,a,lda,b,ldb,0.0,c,ldc);
}
