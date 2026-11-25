/* A separate build of the frozen rank adapter removes the previous small-panel
   cap. Only verified, non-reference FP32 JIT products enter this study. */
#define RANK_LIBXSMM_MAX_MNK INT32_MAX
#include "../libxsmm/adapter.c"

size_t rank_xsmm_code_size(int m, int n, int k, int lda, int ldb, int ldc)
{
    libxsmm_xmmfunction code;
    libxsmm_kernel_info generated = {0};
    libxsmm_mmkernel_info gemm = {0};
    code.gemm = kernel_for(m, n, k, lda, ldb, ldc);
    if (!code.gemm || libxsmm_get_kernel_info(code.ptr_const, &generated) != 0 ||
        libxsmm_get_mmkernel_info(code, &gemm) != 0 || generated.is_reference_kernel ||
        generated.kind != LIBXSMM_KERNEL_KIND_MATMUL ||
        gemm.m != (unsigned)n || gemm.n != (unsigned)m || gemm.k != (unsigned)k ||
        gemm.lda != (unsigned)ldb || gemm.ldb != (unsigned)lda || gemm.ldc != (unsigned)ldc ||
        gemm.iprecision != LIBXSMM_DATATYPE_F32 || gemm.oprecision != LIBXSMM_DATATYPE_F32 ||
        !(gemm.flags & LIBXSMM_GEMM_FLAG_BETA_0))
        return 0;
    return generated.code_size;
}
