#include "../vendor.h"

#include <blasfeo_memory.h>
#include <blasfeo_s_blas_api.h>
#include <stdio.h>
#include <string.h>

#ifndef RANK_ROW_SLAB
#define RANK_ROW_SLAB 0
#endif

int rank_vendor_init(void)
{
    /* Each ranking worker is an isolated single-threaded process. BLASFEO's
       standard workspace holds scratch only, never a caller's packed weights. */
    if (!blasfeo_is_init())
        blasfeo_init();
    fprintf(stderr, "BLASFEO target=%s API=standard-SGEMM workspace=initialized inputs=unpacked timed_output_clear=%d\n",
            RANK_BLASFEO_TARGET, RANK_CLEAR_OUTPUT);
#if RANK_ROW_SLAB > 0
    fprintf(stderr, "BLASFEO row slab=%d, all library calls included in timing\n", RANK_ROW_SLAB);
#endif
    return !blasfeo_is_init() || !blasfeo_get_buffer();
}

void rank_vendor_sgemm(int m, int n, int k, const float *a, int lda,
                       const float *b, int ldb, float *c, int ldc)
{
    char no_trans = 'N';
    float alpha = 1.0f, beta = 0.0f;
#if RANK_CLEAR_OUTPUT
    /* On affected native targets BLASFEO's fringe microkernels read old C even
       for beta=0. Charge the overwrite to the call and preserve row padding. */
    if (ldc == n)
        memset(c, 0, (size_t)m * n * sizeof(*c));
    else
        for (int i = 0; i < m; ++i)
            memset(c + (size_t)i * ldc, 0, (size_t)n * sizeof(*c));
#endif
    /* The column-major view of these row-major buffers satisfies C^T=B^T A^T. */
#if RANK_ROW_SLAB > 0
    for (int first = 0; first < m;) {
        int rows = m - first < RANK_ROW_SLAB ? m - first : RANK_ROW_SLAB;
        blasfeo_blas_sgemm(&no_trans, &no_trans, &n, &rows, &k, &alpha, (float *)b, &ldb,
                           (float *)a + (size_t)first * lda, &lda, &beta, c + (size_t)first * ldc, &ldc);
        first += rows;
    }
#else
    blasfeo_blas_sgemm(&no_trans, &no_trans, &n, &m, &k, &alpha, (float *)b, &ldb,
                       (float *)a, &lda, &beta, c, &ldc);
#endif
}
