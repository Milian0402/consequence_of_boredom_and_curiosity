/* Only these two kernels have target-specific compiler flags. */
void cob_stream_b_kernel_avx2(int m, int n, int k, const float* a, int lda,
    const float* b, int ldb, float* c, int ldc, float* panel);
void cob_stream_b_kernel_avx512(int m, int n, int k, const float* a, int lda,
    const float* b, int ldb, float* c, int ldc, float* panel);

#ifndef COB_STREAM_B_SUPPORTS
#define COB_STREAM_B_SUPPORTS(feature) __builtin_cpu_supports(feature)
#endif
#ifndef COB_STREAM_B_OBSERVE
#define COB_STREAM_B_OBSERVE(width) ((void)0)
#endif

static int cob_stream_b_tile_width(void)
{
    if (!COB_STREAM_B_SUPPORTS("fma") || !COB_STREAM_B_SUPPORTS("avx2")) return 0;
#ifndef COB_DISABLE_X86_AVX512
    if (COB_STREAM_B_SUPPORTS("avx512f")) return 64;
#endif
    return 32;
}

static int cob_try_stream_b(int m, int n, int k, const float* a, int lda,
    const float* b, int ldb, float* c, int ldc)
{
    if (m < 24 || m > 160 || n < 1024 || n > 4096 || k < 96 || k > 384 ||
        n < 8*m || m*k > 49152 || !a || !b || !c || lda < k || ldb < n || ldc < n)
        return 0;
    const size_t output_size = (size_t)(m-1)*ldc+n;
    if (cob_f32_ranges_overlap(c, output_size, a, (size_t)(m-1)*lda+k) ||
        cob_f32_ranges_overlap(c, output_size, b, (size_t)(k-1)*ldb+n)) return 0;
    const int width = cob_stream_b_tile_width();
    if (!width) return 0;
    float* panel = cob_aligned_alloc(64, (size_t)k*width*sizeof(float));
    if (!panel) return 0;
    COB_STREAM_B_OBSERVE(width);
    if (width == 64)
        cob_stream_b_kernel_avx512(m, n, k, a, lda, b, ldb, c, ldc, panel);
    else
        cob_stream_b_kernel_avx2(m, n, k, a, lda, b, ldb, c, ldc, panel);
    free(panel);
    return 1;
}
