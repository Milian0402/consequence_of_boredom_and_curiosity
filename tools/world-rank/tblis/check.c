#include <math.h>
#include <stdio.h>
#include <string.h>

int rank_vendor_init(void);
void rank_vendor_sgemm(int, int, int, const float *, int, const float *, int, float *, int);

int main(void)
{
    static const int cases[][3] = {
        {32, 64, 48}, {33, 37, 23}, {34, 37, 23}, {33, 36, 23},
        {33, 37, 24}, {16, 32, 4}, {3, 1, 1}, {1, 3, 0}
    };
    if (rank_vendor_init()) return 1;
    for (size_t t = 0; t < sizeof(cases)/sizeof(cases[0]); ++t) {
        const int m = cases[t][0], n = cases[t][1], k = cases[t][2];
        const int lda = k + 1, ldb = n + 3, ldc = n + 5;
        float a[m*lda], b[(k+1)*ldb], c[m*ldc], acopy[m*lda], bcopy[(k+1)*ldb];
        for (int i = 0; i < m*lda; ++i) a[i] = i % lda < k ? (i%7)-3 : NAN;
        for (int i = 0; i < (k+1)*ldb; ++i) b[i] = i % ldb < n ? (i%5)-2 : NAN;
        for (int i = 0; i < m*ldc; ++i) c[i] = i % ldc < n ? NAN : 1543;
        memcpy(acopy, a, sizeof(a)); memcpy(bcopy, b, sizeof(b));
        rank_vendor_sgemm(m, n, k, a, lda, b, ldb, c, ldc);
        if (memcmp(acopy, a, sizeof(a)) || memcmp(bcopy, b, sizeof(b))) return 2;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < ldc; ++j) {
                double expected = j < n ? 0 : 1543;
                if (j < n)
                    for (int p = 0; p < k; ++p) expected += (double)a[i*lda+p] * b[p*ldb+j];
                if (expected != c[i*ldc+j]) {
                    fprintf(stderr, "%dx%dx%d (%d,%d) got %.9g expected %.9g\n",
                            m,n,k,i,j,c[i*ldc+j],expected);
                    return 3;
                }
            }
        }
        printf("%dx%dx%d, padding and operands passed\n", m,n,k);
    }
}
