#include <libxsmm.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef RANK_LIBXSMM_MAX_MNK
#define RANK_LIBXSMM_MAX_MNK (64 * 64 * 64)
#endif

#if defined(RANK_LIBXSMM_CACHE_LAST) && RANK_LIBXSMM_CACHE_LAST
struct last_shape {
    int m, n, k, lda, ldb, ldc;
    libxsmm_gemmfunction kernel;
};
static _Thread_local struct last_shape last;
#endif

int rank_vendor_init(void)
{
    libxsmm_init();
    fprintf(stderr, "LIBXSMM version=%s target=%s mode=FP32-JIT inputs=unpacked beta=0\n",
            LIBXSMM_VERSION, libxsmm_get_target_arch());
    return 0;
}

static libxsmm_gemmfunction kernel_for(int m, int n, int k, int lda, int ldb, int ldc)
{
#if defined(RANK_LIBXSMM_CACHE_LAST) && RANK_LIBXSMM_CACHE_LAST
    if (last.kernel && last.m == m && last.n == n && last.k == k &&
        last.lda == lda && last.ldb == ldb && last.ldc == ldc) return last.kernel;
#endif
    /* Column-major C^T = B^T A^T aliases the row-major buffers directly. */
    const libxsmm_datatype fp = LIBXSMM_DATATYPE_F32;
    libxsmm_gemm_shape shape = libxsmm_create_gemm_shape(n, m, k, ldb, lda, ldc, fp, fp, fp, fp);
    libxsmm_gemmfunction kernel = libxsmm_dispatch_gemm(shape, LIBXSMM_GEMM_FLAG_BETA_0,
                                                       LIBXSMM_GEMM_PREFETCH_NONE);
#if defined(RANK_LIBXSMM_CACHE_LAST) && RANK_LIBXSMM_CACHE_LAST
    last = (struct last_shape){m, n, k, lda, ldb, ldc, kernel};
#endif
    return kernel;
}

void rank_vendor_sgemm(int m, int n, int k, const float *a, int lda,
                       const float *b, int ldb, float *c, int ldc)
{
    /* The study uses this independent JIT peer only on its frozen small panel. */
    if (m <= 0 || n <= 0 || k <= 0 || (int64_t)m * n > RANK_LIBXSMM_MAX_MNK / k) {
        fputs("LIBXSMM comparator received an ineligible shape\n", stderr);
        abort();
    }
    libxsmm_gemmfunction kernel = kernel_for(m,n,k,lda,ldb,ldc);
    if (!kernel) {
        fprintf(stderr, "LIBXSMM JIT unavailable for %d x %d x %d\n", m, n, k);
        abort();
    }
    libxsmm_gemm_param args = {0};
    args.a.primary = (void *)b;
    args.b.primary = (void *)a;
    args.c.primary = c;
    kernel(&args);
}
