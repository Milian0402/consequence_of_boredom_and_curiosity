#define ACCELERATE_NEW_LAPACK 1
#include <Accelerate/Accelerate.h>
#include <stdlib.h>

__attribute__((constructor)) static void single_thread(void) {
    if (BLASSetThreading(BLAS_THREADING_SINGLE_THREADED) ||
        BLASGetThreading() != BLAS_THREADING_SINGLE_THREADED) abort();
}

void cob_sgemm_rowmajor(int m, int n, int k, const float* a, int lda,
                        const float* b, int ldb, float* c, int ldc) {
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,n,k,1.f,a,lda,b,ldb,0.f,c,ldc);
}
