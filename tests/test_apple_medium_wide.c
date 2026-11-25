#include "cob_gemm.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int run_case(int m, int n, int k, int pad)
{
    const int lda = k + pad, ldb = n + pad, ldc = n + pad + 1;
    const float guard = -8765.0f;
    float *a = malloc((size_t)m * lda * sizeof(*a));
    float *b = malloc((size_t)k * ldb * sizeof(*b));
    float *c = malloc(((size_t)m * ldc + 1) * sizeof(*c));
    if (!a || !b || !c) exit(EXIT_FAILURE);
    for (int i = 0; i < m; ++i)
        for (int p = 0; p < lda; ++p)
            a[(size_t)i * lda + p] = p < k ? (float)((i * 23 + p * 13) % 31 - 15) / 16.0f : NAN;
    for (int p = 0; p < k; ++p)
        for (int j = 0; j < ldb; ++j)
            b[(size_t)p * ldb + j] = j < n ? (float)((p * 17 + j * 11) % 29 - 14) / 16.0f : NAN;
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < ldc; ++j) c[(size_t)i * ldc + j] = j < n ? NAN : guard;
    c[(size_t)m * ldc] = guard;
    cob_sgemm_rowmajor(m, n, k, a, lda, b, ldb, c, ldc);
    int ok = c[(size_t)m * ldc] == guard;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            double expected = 0;
            for (int p = 0; p < k; ++p)
                expected += (double)a[(size_t)i * lda + p] * b[(size_t)p * ldb + j];
            if ((double)c[(size_t)i * ldc + j] != expected) ok = 0;
        }
        for (int j = n; j < ldc; ++j)
            if (c[(size_t)i * ldc + j] != guard) ok = 0;
    }
    free(a); free(b); free(c);
    if (!ok) fprintf(stderr, "medium-wide mismatch for %d x %d x %d with pad %d\n", m, n, k, pad);
    return ok;
}

int main(void)
{
    const int cases[][4] = {{24, 1024, 96, 0}, {31, 1087, 127, 3}, {48, 1153, 193, 7},
                            {55, 4096, 384, 1}, {23, 1089, 131, 5}, {56, 1217, 147, 0}};
    for (size_t i = 0; i < sizeof(cases) / sizeof(*cases); ++i)
        if (!run_case(cases[i][0], cases[i][1], cases[i][2], cases[i][3])) return EXIT_FAILURE;
    return EXIT_SUCCESS;
}
