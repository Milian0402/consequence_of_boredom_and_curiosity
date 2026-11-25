#include <libxsmm.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef RANK_LIBXSMM_MAX_MNK
#define RANK_LIBXSMM_MAX_MNK (64 * 64 * 64)
#endif

int rank_vendor_init(void)
{
    libxsmm_init();
    fprintf(stderr, "LIBXSMM version=%s target=%s mode=FP32-JIT inputs=unpacked beta=0\n",
            LIBXSMM_VERSION, libxsmm_get_target_arch());
    return 0;
}

void rank_vendor_sgemm(int m, int n, int k, const float *a, int lda,
                       const float *b, int ldb, float *c, int ldc)
{
    /* The study uses this independent JIT peer only on its frozen small panel. */
    if (m <= 0 || n <= 0 || k <= 0 || (int64_t)m * n > RANK_LIBXSMM_MAX_MNK / k) {
        fputs("LIBXSMM comparator received an ineligible shape\n", stderr);
        abort();
    }
    /* Column-major C^T = B^T A^T aliases the row-major buffers directly. */
    const libxsmm_datatype fp = LIBXSMM_DATATYPE_F32;
    libxsmm_gemm_shape shape = libxsmm_create_gemm_shape(n, m, k, ldb, lda, ldc, fp, fp, fp, fp);
    libxsmm_gemmfunction kernel = libxsmm_dispatch_gemm(shape, LIBXSMM_GEMM_FLAG_BETA_0,
                                                       LIBXSMM_GEMM_PREFETCH_NONE);
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
