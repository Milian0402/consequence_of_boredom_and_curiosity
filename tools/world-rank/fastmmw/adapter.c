#include <cblas.h>
#include <mat-operands.h>
#include <stdio.h>
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
int s_wm(S_Matrix, S_Matrix, S_Matrix);

#ifdef FASTMMW_TRACE
static _Thread_local int leaves, padded;
int rank_fastmmw_padded(void) { return padded; }
void __real_sgemm_(const char *, const char *, const int *, const int *, const int *,
                   const float *, const float *, const int *, const float *, const int *,
                   const float *, float *, const int *);
void __wrap_sgemm_(const char *ta, const char *tb, const int *m, const int *n,
                   const int *k, const float *alpha, const float *a, const int *lda,
                   const float *b, const int *ldb, const float *beta, float *c, const int *ldc)
{
    ++leaves;
    __real_sgemm_(ta,tb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc);
}
int rank_fastmmw_leaves(void) { return leaves; }
#endif

int rank_vendor_init(void)
{
    openblas_set_num_threads(1);
    fprintf(stderr, "FastMMW s_wm serial, OpenBLAS %s threads=%d parallel=%d cutoff=%d\n",
            openblas_get_config(), openblas_get_num_threads(), openblas_get_parallel(), LEAF);
    return openblas_get_num_threads() != 1 || openblas_get_parallel() != OPENBLAS_SEQUENTIAL;
}

static void upstream_gemm(int m, int n, int k, const float *a, int lda,
                       const float *b, int ldb, float *c, int ldc)
{
    /* C^T = B^T A^T lets the column-major upstream descriptor use caller
       leading dimensions directly. The upstream routine allocates its scratch. */
    S_Matrix left = {(float *)b,n,k,ldb,k,'n',1.0f};
    S_Matrix right = {(float *)a,k,m,lda,m,'n',1.0f};
    S_Matrix result = {c,n,m,ldc,m,'n',1.0f};
#ifdef FASTMMW_TRACE
    leaves = 0;
#endif
    s_wm(result, left, right);
}

/* Repeated exact halving prevents the unequal-quadrant fringe bug in the
   upstream column-major addition kernels. The BLAS leaves accept odd sizes. */
static int round_dimension(int dim, int64_t multiple)
{
    int64_t rounded = ((dim + multiple - 1) / multiple) * multiple;
    if (rounded > INT_MAX) abort();
    return (int)rounded;
}
static int64_t split_multiple(int m, int n, int k)
{
    int smallest = m < n ? m : n;
    if (k < smallest) smallest = k;
    int64_t multiple = 1;
    while ((smallest + multiple - 1) / multiple > LEAF) multiple *= 2;
    return multiple;
}
#ifdef FASTMMW_PAD
static float *zeros(int rows, int cols)
{
    if ((size_t)rows > SIZE_MAX / sizeof(float) / (size_t)cols) abort();
    float *buffer = calloc((size_t)rows * cols, sizeof(float));
    if (!buffer) abort();
    return buffer;
}
static void copy_rows(int rows, int cols, const float *src, int lds, float *dst, int ldd)
{
    for (int i = 0; i < rows; ++i)
        memcpy(dst + (size_t)i*ldd, src + (size_t)i*lds, (size_t)cols*sizeof(*dst));
}
#endif

void rank_vendor_sgemm(int m, int n, int k, const float *a, int lda,
                       const float *b, int ldb, float *c, int ldc)
{
    if (m < 1 || n < 1 || k < 1 || lda < k || ldb < n || ldc < n) abort();
    int64_t multiple = split_multiple(m, n, k);
    int mp=round_dimension(m,multiple), np=round_dimension(n,multiple), kp=round_dimension(k,multiple);
#ifdef FASTMMW_TRACE
    padded = mp != m || np != n || kp != k;
#endif
    if (mp == m && np == n && kp == k) {
        upstream_gemm(m,n,k,a,lda,b,ldb,c,ldc);
        return;
    }
#ifdef FASTMMW_PAD
    float *ap=zeros(mp,kp), *bp=zeros(kp,np), *cp=zeros(mp,np);
    copy_rows(m,k,a,lda,ap,kp);
    copy_rows(k,n,b,ldb,bp,np);
    upstream_gemm(mp,np,kp,ap,kp,bp,np,cp,np);
    copy_rows(m,n,cp,np,c,ldc);
    free(cp);
    free(bp);
    free(ap);
#else
    fprintf(stderr,"FastMMW stock adapter does not support this recursive fringe\n");
    abort();
#endif
}
