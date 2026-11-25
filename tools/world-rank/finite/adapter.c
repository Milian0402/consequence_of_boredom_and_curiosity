/* Experimental single-level Hopcroft-Kerr rank-15 row-major SGEMM.
   Shape notation here is M,N,K. The split is (M/2, N/3, K/3).
   All matrix products use the ordinary single-thread COB entry point. */
#define _POSIX_C_SOURCE 200112L
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

void cob_sgemm_rowmajor(int, int, int, const float *, int, const float *, int, float *, int);
static _Thread_local int last_products;
int rank_vendor_init(void) { return 0; }
int rank_hk233_last_products(void) { return last_products; }

static int overlaps(const float *a, size_t as, const float *b, size_t bs)
{
    uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
    return x <= y ? y - x < as * sizeof(float) : x - y < bs * sizeof(float);
}

void rank_vendor_sgemm(int m, int n, int k, const float *aa, int lda,
                       const float *bb, int ldb, float *cc, int ldc)
{
    last_products = 0;
    if (m <= 0 || n <= 0 || k <= 0 || m % 2 || n % 3 || k % 3 ||
        !aa || !bb || !cc || lda < k || ldb < n || ldc < n ||
        overlaps(cc, (size_t)(m - 1) * ldc + n, aa, (size_t)(m - 1) * lda + k) ||
        overlaps(cc, (size_t)(m - 1) * ldc + n, bb, (size_t)(k - 1) * ldb + n))
        goto fallback;

    const int mh = m / 2, nh = n / 3, kh = k / 3;
    const size_t ac = (size_t)mh * kh, bc = (size_t)kh * nh, pc = (size_t)mh * nh;
    /* The dimension divisors bound the sum below SIZE_MAX/4 on 64-bit CPUs,
       but keep this explicit for reuse in smaller memory models. */
    if (ac > SIZE_MAX / sizeof(float) || bc > SIZE_MAX / sizeof(float) - ac ||
        pc > SIZE_MAX / sizeof(float) - ac - bc) goto fallback;
    float *workspace = NULL;
    if (posix_memalign((void **)&workspace, 128, (ac + bc + pc) * sizeof(float))) goto fallback;
    float *x = workspace, *y = workspace + ac, *product = y + bc;
    const float *a[6], *b[9];
    float *c[6];
    for (int i = 0; i < 6; ++i) {
        a[i] = aa + (size_t)(i / 3) * mh * lda + (i % 3) * kh;
        c[i] = cc + (size_t)(i / 3) * mh * ldc + (i % 3) * nh;
    }
    for (int i = 0; i < 9; ++i)
        b[i] = bb + (size_t)(i / 3) * kh * ldb + (i % 3) * nh;
#include "hk233.inc"
    free(workspace);
    last_products = 15;
    return;

fallback:
    cob_sgemm_rowmajor(m, n, k, aa, lda, bb, ldb, cc, ldc);
}

void mtmm_sgemm_rowmajor(int m, int n, int k, const float *a, int lda,
                         const float *b, int ldb, float *c, int ldc, const int *threads)
{
    if (!threads || *threads != 1) abort();
    rank_vendor_sgemm(m, n, k, a, lda, b, ldb, c, ldc);
}
