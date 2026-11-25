#include <inttypes.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef N2_TEST_PUBLIC
#include "cob_gemm.h"
#else
int n2_try_sgemm(int, int, int, const float *, int, const float *, int, float *, int);
#endif

static uint32_t state = 3141592U;
static float draw(void)
{
    state = state * 1664525U + 1013904223U;
    return ((int)(state >> 16) - 32768) * 0x1p-15f;
}

static uint64_t outputs;
static int cases;
static double worst;

static int check(int m, int n, int k, int apad, int bpad, int cpad)
{
    const int lda = k + apad, ldb = n + bpad, ldc = n + cpad;
    float *a = malloc((size_t)m * lda * sizeof(*a));
    float *b = malloc((size_t)k * ldb * sizeof(*b));
    float *c = malloc((size_t)m * ldc * sizeof(*c));
    if (!a || !b || !c) exit(2);
    for (int i = 0; i < m; ++i)
        for (int p = 0; p < lda; ++p)
            a[(size_t)i * lda + p] = p < k ? draw() : NAN;
    for (int p = 0; p < k; ++p)
        for (int j = 0; j < ldb; ++j)
            b[(size_t)p * ldb + j] = j < n ? draw() : NAN;
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < ldc; ++j)
            c[(size_t)i * ldc + j] = j < n ? NAN : 486123.0f;

#ifdef N2_TEST_PUBLIC
    cob_sgemm_rowmajor(m, n, k, a, lda, b, ldb, c, ldc);
    int bad = 0;
#else
    int bad = !n2_try_sgemm(m, n, k, a, lda, b, ldb, c, ldc);
#endif
    for (int i = 0; i < m && !bad; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0;
            for (int p = 0; p < k; ++p)
                sum += (double)a[(size_t)i * lda + p] * b[(size_t)p * ldb + j];
            const double diff = fabs(sum - c[(size_t)i * ldc + j]);
            if (!isfinite(diff) || diff > 0.002) ++bad;
            if (diff > worst) worst = diff;
        }
        for (int j = n; j < ldc; ++j)
            if (c[(size_t)i * ldc + j] != 486123.0f) ++bad;
    }
    if (bad) fprintf(stderr, "failure in %dx%dx%d pads %d,%d,%d\n", m,n,k,apad,bpad,cpad);
    outputs += (uint64_t)m*n;
    ++cases;
    free(c);
    free(b);
    free(a);
    return bad != 0;
}

int main(void)
{
    for (int n = 16; n <= 128; n += 16) {
        if (check(n,n,n,0,0,0) || check(n,n,n,2,1,3)) return 1;
    }
    for (int t = 0; t < 48; ++t) {
        int m = 8 + 7*t%121, n = 8 + 11*t%121, k = 8 + 17*t%121;
        if (check(m,n,k,t%5,t%3,t%7)) return 1;
    }
    for (int r=0;r<6;++r)
        for (int col=0;col<16;++col)
            if (check(24+r,48+col,47+(r+col)%4,2,3,5)) return 1;
    const int cases_extra[][4]={{40,40,40,0},{88,88,88,0},{37,113,59,5},
        {23,77,43,3},{73,61,89,7},{32,96,48,0},{56,56,56,0},{104,104,104,0},
        {29,83,53,5},{79,67,97,3},{40,112,56,0}};
    for (size_t t=0;t<sizeof(cases_extra)/sizeof(cases_extra[0]);++t) {
        const int* v=cases_extra[t];
        if (check(v[0],v[1],v[2],v[3],v[3],v[3])) return 1;
    }
#ifdef N2_TEST_PUBLIC
    /* Valid products just outside each dispatch boundary use ordinary COB. */
    const int fallback[][3]={{7,37,25},{23,7,29},{23,17,7},{129,17,29},
                             {23,129,29},{23,17,129},{1,1,1}};
    for (size_t t=0; t<sizeof(fallback)/sizeof(fallback[0]); ++t) {
        const int *v=fallback[t];
        if (check(v[0],v[1],v[2],2,3,5)) return 1;
    }
#else
    float alias[4096] = {0};
    if (n2_try_sgemm(32,32,32,alias,32,alias+2000,32,alias+10,32) != 0 ||
        n2_try_sgemm(32,32,32,alias,32,alias+1040,32,alias+1080,32) != 0 ||
        n2_try_sgemm(129,128,8,alias,8,alias,128,alias,128) != 0) return 1;
#endif
    printf("%d layouts, %" PRIu64 " FP64 outputs, max error %.9g, padding preserved\n",
           cases, outputs, worst);
    return 0;
}
