/* Four output rows share each 16-float B segment; no input packing. */
#if defined(COB_ENABLE_DIRECT_SMALL_GEMM) && COB_ENABLE_DIRECT_SMALL_GEMM && \
    (defined(COB_USE_X86_SIMD) || (defined(COB_USE_NEON) && defined(__aarch64__)))
#define COB_USE_DIRECT_SMALL_GEMM 1
#if defined(COB_USE_X86_SIMD)
#define COB_SMALL_FN __attribute__((target("avx2,fma")))
#define COB_SMALL_LANES 8
#else
#define COB_SMALL_FN
#define COB_SMALL_LANES 4
#endif
typedef float cob_small_vec __attribute__((vector_size(4 * COB_SMALL_LANES)));
COB_SMALL_FN static void cob_sgemm_small_direct(int m, int n, int k,
    const float* a, int lda, const float* b, int ldb, float* c, int ldc)
{
    enum { vectors = 16 / COB_SMALL_LANES };
    for (int i = 0; i < m; i += 4) {
        for (int j = 0; j < n; j += 16) {
            cob_small_vec acc[4][vectors] = {0};
            for (int p = 0; p < k; ++p) {
                const float* bp = b + (size_t)p * ldb + j;
                cob_small_vec right[vectors];
                memcpy(right, bp, sizeof(right));
                for (int r = 0; r < 4; ++r) {
                    float left = a[(size_t)(i + r) * lda + p];
                    for (int v = 0; v < vectors; ++v) acc[r][v] += right[v] * left;
                }
            }
            for (int r = 0; r < 4; ++r) {
                memcpy(c + (size_t)(i+r)*ldc + j, acc[r], sizeof(acc[r]));
            }
        }
    }
}
#undef COB_SMALL_FN
#undef COB_SMALL_LANES
#endif
