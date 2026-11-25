#define _POSIX_C_SOURCE 200112L
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fail_workspace;
static int test_memalign(void** ptr, size_t align, size_t size)
{
    return fail_workspace ? 12 : posix_memalign(ptr, align, size);
}
#define posix_memalign test_memalign
#include "../src/gemm.c"
#undef posix_memalign

int main(void)
{
    enum { m = 71, n = 74, k = 73, lda = k+5, ldb = n+3, ldc = n+7 };
    float a[m*lda], b[k*ldb], c[m*ldc], original_a[m*lda], original_b[k*ldb];
    for (int i=0; i<m; ++i) {
        for (int p=0; p<lda; ++p)
            a[i*lda+p] = p<k ? (float)(((i*3+p*5)%29)-14)/13 : NAN;
    }
    for (int p=0; p<k; ++p) {
        for (int j=0; j<ldb; ++j)
            b[p*ldb+j] = j<n ? (float)(((p*7+j*2)%31)-15)/17 : NAN;
    }
    memcpy(original_a, a, sizeof(a));
    memcpy(original_b, b, sizeof(b));
    for (int i=0; i<m*ldc; ++i) c[i] = 531.0f;
    openblas_set_num_threads(1);
    if (cob_openblas_winograd_try(m,n,k,a,lda,b,ldb,c,ldc,2048) ||
        cob_openblas_winograd_try(m,n,k,a,k-1,b,ldb,c,ldc,64) ||
        cob_openblas_winograd_try(m,n,k,a,lda,b,ldb,a,ldc,64) ||
        cob_openblas_winograd_try(64,64,64,a,lda,b,ldb,c,ldc,64)) return 1;
    fail_workspace = 1;
    if (cob_openblas_winograd_try(m,n,k,a,lda,b,ldb,c,ldc,64)) return 1;
    fail_workspace = 0;
    openblas_set_num_threads(2);
    if (openblas_get_num_threads() > 1 &&
        cob_openblas_winograd_try(m,n,k,a,lda,b,ldb,c,ldc,64)) return 1;
    openblas_set_num_threads(1);
    for (int i=0; i<m*ldc; ++i) if (c[i] != 531.0f) return 1;
    if (!cob_openblas_winograd_try(m,n,k,a,lda,b,ldb,c,ldc,64)) return 1;
    if (memcmp(original_a,a,sizeof(a)) || memcmp(original_b,b,sizeof(b))) return 1;
    for (int i=0; i<m; ++i) {
        for (int j=0; j<n; ++j) {
            double ref = 0;
            for (int p=0; p<k; ++p) ref += (double)a[i*lda+p]*b[p*ldb+j];
            if (!isfinite(c[i*ldc+j]) || fabs(ref-c[i*ldc+j]) > 0.002) return 1;
        }
        for (int j=n; j<ldc; ++j) if (c[i*ldc+j] != 531.0f) return 1;
    }
    puts("Rectangular OpenBLAS Winograd passed full FP64, stride, alias and allocation checks.");
    return 0;
}
