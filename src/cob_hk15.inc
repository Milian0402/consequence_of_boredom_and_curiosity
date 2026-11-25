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

/* Under binary32, unsigned magnitude bits preserve numerical order, including
   subnormals, independently of the current floating-point flush mode. */
enum cob_hk15_policy { COB_HK15_REGULAR, COB_HK15_CLASSICAL, COB_HK15_WIDE };

static enum cob_hk15_policy cob_hk15_axis_policy(int rows, int cols,
    const float* input, int ld, uint32_t* columns)
{
    const uint32_t low = UINT32_C(0x27800000);  /* 2^-48 */
    const uint32_t high = UINT32_C(0x57800000); /* 2^48 */
    uint32_t largest = 0, smallest_line = UINT32_MAX, has_tiny = 0;
    memset(columns, 0, (size_t)cols * sizeof(*columns));
    for (int i = 0; i < rows; ++i) {
        const float* row = input + (size_t)i * ld;
        uint32_t row_max = 0;
        for (int j = 0; j < cols; ++j) {
            uint32_t bits;
            memcpy(&bits, row + j, sizeof(bits));
            const uint32_t mag = bits & UINT32_C(0x7fffffff);
            has_tiny |= mag != 0 && mag < low;
            row_max = mag > row_max ? mag : row_max;
            columns[j] = mag > columns[j] ? mag : columns[j];
        }
        if (row_max > largest) largest = row_max;
        if (row_max < smallest_line) smallest_line = row_max;
    }
    if (has_tiny || largest > high) return COB_HK15_WIDE;
    for (int j = 0; j < cols; ++j)
        if (columns[j] < smallest_line) smallest_line = columns[j];
    /* Adding two to a normal binary32 exponent multiplies by four. */
    if (smallest_line == 0 || largest > smallest_line + (UINT32_C(2) << 23))
        return COB_HK15_CLASSICAL;
    return COB_HK15_REGULAR;
}

static enum cob_hk15_policy cob_hk15_policy_for(int m, int n, int k,
    const float* a, int lda, const float* b, int ldb)
{
    /* HK accepts at most 1280 columns per third plus two fringe columns.
       Keep the scan allocation-free even when the arithmetic workspace fails. */
    uint32_t columns[3 * 1280 + 2];
    if ((size_t)n > sizeof(columns)/sizeof(*columns) ||
        (size_t)k > sizeof(columns)/sizeof(*columns)) return COB_HK15_WIDE;
    enum cob_hk15_policy left = cob_hk15_axis_policy(m, k, a, lda, columns);
    if (left == COB_HK15_WIDE) return left;
    enum cob_hk15_policy right = cob_hk15_axis_policy(k, n, b, ldb, columns);
    return left > right ? left : right;
}

static void cob_hk15_classical(int m, int n, int k, const float* a, int lda,
    const float* b, int ldb, float* c, int ldc)
{
#if defined(COB_WITH_STRASSEN_SME_LEAF) && defined(COB_USE_APPLE_SME) && defined(COB_USE_APPLE_AMX)
    if (cob_sgemm_try_sme_classical(m, n, k, a, lda, b, ldb, c, ldc)) return;
#endif
    cob_sgemm_rowmajor_fallback(m, n, k, a, lda, b, ldb, c, ldc);
}

static void cob_hk15_wide(int m, int n, int k, const float* a, int lda,
    const float* b, int ldb, float* c, int ldc)
{
    /* Finite binary32 products, including subnormal products, fit exactly
       in binary64. Even k times the largest product fits its exponent range. */
    enum { tile = 32 };
    for (int i = 0; i < m; ++i) {
        const float* row = a + (size_t)i * lda;
        for (int j = 0; j < n; j += tile) {
            const int width = cob_min_i32(n-j, tile);
            double sum[tile] = {0};
            for (int p = 0; p < k; ++p) {
                const float* rhs = b + (size_t)p * ldb + j;
                const double lhs = row[p];
                for (int q = 0; q < width; ++q)
                    sum[q] += lhs * (double)rhs[q];
            }
            for (int q = 0; q < width; ++q)
                c[(size_t)i * ldc + j + q] = (float)sum[q];
        }
    }
}

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

    if (as*sizeof(float)-1 > UINTPTR_MAX-(uintptr_t)aa ||
        bs*sizeof(float)-1 > UINTPTR_MAX-(uintptr_t)bb ||
        cs*sizeof(float)-1 > UINTPTR_MAX-(uintptr_t)cc) return 0;

    const enum cob_hk15_policy policy = cob_hk15_policy_for(m,n,k,aa,lda,bb,ldb);
    if (policy != COB_HK15_REGULAR) {
        if (policy == COB_HK15_WIDE) cob_hk15_wide(m,n,k,aa,lda,bb,ldb,cc,ldc);
        else cob_hk15_classical(m,n,k,aa,lda,bb,ldb,cc,ldc);
        return 1;
    }
    const int mh = m/2, nh = n/3, kh = k/3;
    const size_t ac = (size_t)mh*kh, bc = (size_t)kh*nh, pc = (size_t)mh*nh;
    /* All three block dimensions have a validated upper bound of 1280. */
    float* workspace = cob_aligned_alloc(128, (ac+bc+pc)*sizeof(float));
    if (!workspace) {
        cob_hk15_classical(m,n,k,aa,lda,bb,ldb,cc,ldc);
        return 1;
    }
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
