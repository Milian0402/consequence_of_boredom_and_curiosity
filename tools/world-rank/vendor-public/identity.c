#include "cob_gemm.h"
#include <cblas.h>
#include <stdio.h>

int rank_backend_init(void)
{
    fprintf(stderr, "private archive %s, parallel=%d, threads=%d\n",
        openblas_get_config(),openblas_get_parallel(),openblas_get_num_threads());
    return openblas_get_parallel() != 0 || openblas_get_num_threads() != 1;
}

const char* rank_backend_provider(void) { return openblas_get_config(); }

void rank_backend_sgemm(int m,int n,int k,const float* a,int lda,const float* b,int ldb,float* c,int ldc)
{
    cob_sgemm_rowmajor(m,n,k,a,lda,b,ldb,c,ldc);
}
