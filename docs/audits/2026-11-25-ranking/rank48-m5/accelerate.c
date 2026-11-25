#define ACCELERATE_NEW_LAPACK 1
#include <Accelerate/Accelerate.h>
int rank_vendor_init(void) { return BLASSetThreading(BLAS_THREADING_SINGLE_THREADED) || BLASGetThreading()!=BLAS_THREADING_SINGLE_THREADED; }
void rank_vendor_sgemm(int m,int n,int k,const float* a,int lda,const float* b,int ldb,float* c,int ldc) {
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,n,k,1,a,lda,b,ldb,0,c,ldc);
}
void rank_vendor_dgemm(int m,int n,int k,const double* a,int lda,const double* b,int ldb,double* c,int ldc) {
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,n,k,1,a,lda,b,ldb,0,c,ldc);
}
