/* One level of the 2-by-3-by-3 Hopcroft-Kerr schedule. Each leaf uses
   the existing dispatcher, including any enabled Strassen routes. */
#if defined(COB_USE_FUSED_STRASSEN)
#ifndef COB_HK15_CHILD_HARMONIC
#define COB_HK15_CHILD_HARMONIC 960
#endif
#if COB_HK15_CHILD_HARMONIC <= 0
#error "COB_HK15_CHILD_HARMONIC must be positive"
#endif

static void cob_hk15_fused_child(int m, int n, int k, const float* a, int lda,
    const float* b, int ldb, float* c, int ldc)
{
    if (
#if defined(COB_USE_RECURSIVE_STRASSEN)
        !(lda == k && ldb == n && cob_recursive_square_is_eligible(m, n, k)) &&
#endif
        cob_sgemm_strassen_fused_at(m, n, k, a, lda, b, ldb, c, ldc,
            COB_HK15_CHILD_HARMONIC)) return;
    cob_sgemm_rowmajor_base(m, n, k, a, lda, b, ldb, c, ldc);
}
#endif

static int cob_hk15_shape(int m, int n, int k)
{
#if !defined(COB_ENABLE_HK15_RAGGED) || !COB_ENABLE_HK15_RAGGED
    if (m%2 || n%3 || k%3) return 0;
#endif
    const int mh = m/2, nh = n/3, kh = k/3;
    if (mh < 768 || mh > 1280 || nh < 768 || nh > 1280 || kh < 768 || kh > 1280)
        return 0;
    const int lo = cob_min_i32(mh, cob_min_i32(nh, kh));
    return 4*mh <= 5*lo && 4*nh <= 5*lo && 4*kh <= 5*lo;
}

static int cob_hk15_try(int m, int n, int k, const float* aa, int lda,
    const float* bb, int ldb, float* cc, int ldc)
{
    if (!cob_hk15_shape(m,n,k) || !aa || !bb || !cc ||
        lda < k || ldb < n || ldc < n) return 0;

    const uint64_t as = (uint64_t)(m-1)*lda + k;
    const uint64_t bs = (uint64_t)(k-1)*ldb + n;
    const uint64_t cs = (uint64_t)(m-1)*ldc + n;
    if (as > SIZE_MAX/sizeof(float) || bs > SIZE_MAX/sizeof(float) ||
        cs > SIZE_MAX/sizeof(float) ||
        cob_f32_ranges_overlap(cc, (size_t)cs, aa, (size_t)as) ||
        cob_f32_ranges_overlap(cc, (size_t)cs, bb, (size_t)bs)) return 0;

#if defined(COB_ENABLE_HK15_RAGGED) && COB_ENABLE_HK15_RAGGED
    if (as*sizeof(float)-1 > UINTPTR_MAX-(uintptr_t)aa ||
        bs*sizeof(float)-1 > UINTPTR_MAX-(uintptr_t)bb ||
        cs*sizeof(float)-1 > UINTPTR_MAX-(uintptr_t)cc) return 0;
#endif
    const int mh = m/2, nh = n/3, kh = k/3;
    const size_t ac = (size_t)mh*kh, bc = (size_t)kh*nh, pc = (size_t)mh*nh;
    /* All three block dimensions have a validated upper bound of 1280. */
    float* workspace = cob_aligned_alloc(128, (ac+bc+pc)*sizeof(float));
    if (!workspace) return 0;
    float *x = workspace, *y = x+ac, *product = y+bc;
    const float *a[6], *b[9];
    float* c[6];
    for (int i=0; i<6; ++i) {
        a[i] = aa + (size_t)(i/3)*mh*lda + (i%3)*kh;
        c[i] = cc + (size_t)(i/3)*mh*ldc + (i%3)*nh;
    }
    for (int i=0; i<9; ++i)
        b[i] = bb + (size_t)(i/3)*kh*ldb + (i%3)*nh;

    /* All 15 products share one geometry and must bypass the outer HK dispatcher. */
    void (*cob_sgemm_rowmajor)(int,int,int,const float*,int,
        const float*,int,float*,int) = cob_sgemm_rowmajor_base;
#if defined(COB_USE_FUSED_STRASSEN)
    const double area = (double)mh*nh + (double)mh*kh + (double)nh*kh;
    const double volume = 3.0*mh*nh*kh;
    if (volume >= COB_HK15_CHILD_HARMONIC*area &&
        volume < COB_FUSED_STRASSEN_MIN_HARMONIC_DIM*area)
        cob_sgemm_rowmajor = cob_hk15_fused_child;
#endif
#include "../third_party/fast-matmul/hk233.inc"
#if defined(COB_ENABLE_HK15_RAGGED) && COB_ENABLE_HK15_RAGGED
    if (m%2 || n%3 || k%3)
        cob_hk15_ragged_finish(m,n,k,aa,lda,bb,ldb,cc,ldc,workspace,cob_sgemm_rowmajor_base);
#endif
    free(workspace);
    return 1;
}
