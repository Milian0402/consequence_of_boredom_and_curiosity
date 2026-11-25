/* One level of the 2-by-3-by-3 Hopcroft-Kerr schedule. Each leaf uses
   the existing dispatcher, including any enabled Strassen routes. */
static int cob_hk15_shape(int m, int n, int k)
{
    if (m < 2*768 || m > 2*1280 || n < 3*768 || n > 3*1280 ||
        k < 3*768 || k > 3*1280 || m%2 || n%3 || k%3) return 0;
    const int mh = m/2, nh = n/3, kh = k/3;
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

    /* The certified source spells its 15 calls cob_sgemm_rowmajor. Bind that
       local name to the original dispatcher so no leaf re-enters HK15. */
#ifdef COB_USE_HK15_CHILD
    void (*const cob_sgemm_rowmajor)(int,int,int,const float*,int,
        const float*,int,float*,int) = cob_hk15_child_product;
#else
    void (*const cob_sgemm_rowmajor)(int,int,int,const float*,int,
        const float*,int,float*,int) = cob_sgemm_rowmajor_base;
#endif
#include "../tools/world-rank/finite/hk233.inc"
    free(workspace);
    return 1;
}
