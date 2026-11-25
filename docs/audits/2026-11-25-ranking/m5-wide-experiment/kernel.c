/* Stream one 64-column B panel through the first row tile. Keeping its packed
   copy in the cache lets every subsequent A panel use the existing SME kernel. */
__attribute__((always_inline)) static inline void
cob_wide_first(int k, const float *ap, const float *b, int ldb, svcount_t cols,
               float *bp) __arm_streaming __arm_inout("za")
{
    const svbool_t pg = svptrue_b32();
    const svcount_t all = svptrue_c32();
    svzero_za();
    int p = 0;
#define COB_WIDE_STEP(lhs, depth)                                                      \
    do {                                                                               \
        svfloat32x4_t rhs = svld1_f32_x4(cols, b + (size_t)(depth) * ldb);               \
        svst1_f32_x4(all, bp + (size_t)(depth) * 64, rhs);                              \
        svmopa_za32_f32_m(0, pg, pg, lhs, svget4_f32(rhs, 0));                         \
        svmopa_za32_f32_m(1, pg, pg, lhs, svget4_f32(rhs, 1));                         \
        svmopa_za32_f32_m(2, pg, pg, lhs, svget4_f32(rhs, 2));                         \
        svmopa_za32_f32_m(3, pg, pg, lhs, svget4_f32(rhs, 3));                         \
    } while (0)
    for (; p + 4 <= k; p += 4) {
        svfloat32x4_t av = svld1_f32_x4(all, ap + (size_t)p * 16);
        COB_WIDE_STEP(svget4_f32(av, 0), p);
        COB_WIDE_STEP(svget4_f32(av, 1), p + 1);
        COB_WIDE_STEP(svget4_f32(av, 2), p + 2);
        COB_WIDE_STEP(svget4_f32(av, 3), p + 3);
    }
    for (; p < k; ++p) {
        svfloat32_t av = svld1_f32(pg, ap + (size_t)p * 16);
        COB_WIDE_STEP(av, p);
    }
#undef COB_WIDE_STEP
}

__arm_locally_streaming __arm_new("za") static int
cob_wide_run(int m, int n, int k, const float *a, int lda, const float *b, int ldb,
             float *c, int ldc, float *ap, size_t depth, float *bp)
{
    if (svcntsw() != 16) return 0;
    _Alignas(64) float stage[16 * 64];
    pack_A_block(a, lda, 0, m, 0, k, depth, ap, stage);
    for (int j = 0; j < n; j += 64) {
        const int width = n - j < 64 ? n - j : 64;
        if (width == 64)
            cob_wide_first(k, ap, b + j, ldb, svptrue_c32(), bp);
        else
            cob_wide_first(k, ap, b + j, ldb, svwhilelt_c32_u64(0, (unsigned)width, 4), bp);
        if (width == 64) store_za(c + j, ldc);
        else cob_sme_transfer_short_c(16, width, c + j, ldc, 1);
        for (int i = 16; i < m; i += 16) {
            const int height = m - i < 16 ? m - i : 16;
            float *dst = c + (size_t)i * ldc + j;
            svzero_za();
            micro_kernel_1x4(ap + i * depth, bp, k);
            if (width == 64 && height == 16) store_za(dst, ldc);
            else cob_sme_transfer_short_c(height, width, dst, ldc, 1);
        }
    }
    return 1;
}

int cob_sgemm_sme_medium_wide(int m, int n, int k, const float *a, int lda,
                             const float *b, int ldb, float *c, int ldc)
{
    const size_t depth = ((size_t)k + 63) & ~(size_t)63;
    const size_t rows = ((size_t)m + 15) & ~(size_t)15;
    const size_t a_size = rows * depth;
    float *packed = aligned_alloc(64, (a_size + (size_t)k * 64) * sizeof(*packed));
    if (!packed) return 0;
    const int done = cob_wide_run(m, n, k, a, lda, b, ldb, c, ldc, packed, depth, packed + a_size);
    free(packed);
    return done;
}
