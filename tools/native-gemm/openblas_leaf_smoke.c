#include "cob_gemm.h"
#include <cblas.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void baseline_sgemm_rowmajor(int m, int n, int k, const float *a, int lda,
    const float *b, int ldb, float *c, int ldc);

static int calls;
void __real_cblas_sgemm(enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE ta,
    enum CBLAS_TRANSPOSE tb, blasint m, blasint n, blasint k, float alpha,
    const float *a, blasint lda, const float *b, blasint ldb, float beta,
    float *c, blasint ldc);

void __wrap_cblas_sgemm(enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE ta,
    enum CBLAS_TRANSPOSE tb, blasint m, blasint n, blasint k, float alpha,
    const float *a, blasint lda, const float *b, blasint ldb, float beta,
    float *c, blasint ldc)
{
    ++calls;
    __real_cblas_sgemm(order, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

static float sample(uint32_t *state)
{
    *state = *state * 1664525u + 1013904223u;
    return ((int)(*state >> 16) - 32768) / 32768.0f;
}

int main(int argc, char **argv)
{
    enum { M = 193, N = 255, K = 257, LDA = K+5, LDB = N+3, LDC = N+7 };
    static float a[M*LDA], b[K*LDB], c[M*LDC+2], native[M*LDC+2];
    const float guard = 87654.0f;
    if (argc != 2 || (strcmp(argv[1], "ON") && strcmp(argv[1], "OFF"))) return 1;
    const int hybrid = !strcmp(argv[1], "ON");
    uint32_t seed = 963513;
    for (int i = 0; i < M; ++i)
        for (int p = 0; p < LDA; ++p) a[i*LDA+p] = p < K ? sample(&seed) : NAN;
    for (int p = 0; p < K; ++p)
        for (int j = 0; j < LDB; ++j) b[p*LDB+j] = j < N ? sample(&seed) : NAN;
    for (size_t i = 0; i < sizeof(native)/sizeof(*native); ++i) native[i] = guard;
    baseline_sgemm_rowmajor(M,N,K,a,LDA,b,LDB,native+1,LDC);
    for (int threads = 1; threads <= 2; ++threads) {
        openblas_set_num_threads(threads);
        if (openblas_get_num_threads() != threads) return 2;
        for (size_t i = 0; i < sizeof(c)/sizeof(*c); ++i) c[i] = guard;
        calls = 0;
        cob_sgemm_rowmajor(M,N,K,a,LDA,b,LDB,c+1,LDC);
        const int expected_calls = hybrid && threads == 1 ? 7 : 0;
        if (calls != expected_calls || (!calls && memcmp(c,native,sizeof(c)))) {
            fprintf(stderr, "expected %d BLAS calls and the selected route, got %d\n", expected_calls, calls);
            return 3;
        }
        if (c[0] != guard || c[M*LDC+1] != guard) return 4;
        double worst = 0;
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                double ref = 0;
                for (int p = 0; p < K; ++p) ref += (double)a[i*LDA+p]*b[p*LDB+j];
                double error = fabs(c[1+i*LDC+j]-ref);
                if (!isfinite(error) || error > 0.002) return 5;
                if (error > worst) worst = error;
            }
            for (int j = N; j < LDC; ++j) if (c[1+i*LDC+j] != guard) return 6;
        }
        printf("leaf=%s OpenBLAS_threads=%d calls=%d max_FP64_error=%.9g\n",argv[1],threads,calls,worst);
    }
    return 0;
}
