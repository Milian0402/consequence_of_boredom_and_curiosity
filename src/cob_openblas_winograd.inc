/* Add omitted depth terms after the core is complete. A threaded BLAS
   installation can change its thread setting between subproducts. */
static void cob_cblas_add_depth(int m, int n, int depth, const float* a, int lda,
    const float* b, int ldb, float* c, int ldc)
{
#if defined(COB_ENABLE_ALIGNED_WINOGRAD) && COB_ENABLE_ALIGNED_WINOGRAD
    if (cob_cblas_leaf_is_serial()) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, depth,
            1.0f, a, lda, b, ldb, 1.0f, c, ldc);
        return;
    }
#endif
    for (int i = 0; i < m; ++i) {
        float* row = c + (size_t)i*ldc;
        for (int p = 0; p < depth; ++p) {
            const float left = a[(size_t)i*lda+p];
            const float* right = b + (size_t)p*ldb;
            for (int j = 0; j < n; ++j) row[j] += left*right[j];
        }
    }
}

/* Two Winograd levels for large rectangular products with serial CBLAS leaves.
   Each level shares three temporary blocks across its seven products. */
static int cob_cblas_winograd_try(int m, int n, int k, const float* a, int lda,
    const float* b, int ldb, float* c, int ldc, int min_harmonic)
{
    const int smallest = cob_min_i32(m, cob_min_i32(n, k));
    const int largest = m > n ? (m > k ? m : k) : (n > k ? n : k);
    if (smallest < min_harmonic/2 || (m == n && n == k) || (int64_t)largest > 8*(int64_t)smallest ||
        !a || !b || !c || lda < k || ldb < n || ldc < n) return 0;

    const double areas = (double)m*n + (double)m*k + (double)n*k;
    if (3.0*m*n*k < min_harmonic*areas || !cob_cblas_leaf_is_serial()) return 0;
    const size_t c_span = (size_t)(m-1)*ldc+n;
    if (cob_f32_ranges_overlap(c, c_span, a, (size_t)(m-1)*lda+k) ||
        cob_f32_ranges_overlap(c, c_span, b, (size_t)(k-1)*ldb+n)) return 0;

#if defined(COB_ENABLE_ALIGNED_WINOGRAD) && COB_ENABLE_ALIGNED_WINOGRAD
    /* With two splits, terminal products have dimensions divisible by 16, 16 and 8. */
    enum { align_output = 64, align_depth = 32 };
#else
    enum { align_output = 4, align_depth = 4 };
#endif
    const int core_m = m - m%align_output;
    const int core_n = n - n%align_output;
    const int core_k = k - k%align_depth;
    if (!core_m || !core_n || !core_k) return 0;
    const size_t rows = (size_t)core_m/2, cols = (size_t)core_n/2, depth = (size_t)core_k/2;
    const size_t parent = rows*depth + depth*cols + rows*cols;
    const size_t total = parent + parent/4;
    if (total > (size_t)256*1024*1024/sizeof(float)) return 0;
    float* work = cob_aligned_alloc(128, total*sizeof(float));
    if (!work) return 0;

    cob_winograd_product(core_m, core_n, core_k, a, lda, b, ldb, c, ldc,
        work, work+parent, cob_strassen_leaf);
    free(work);

    /* The omitted depth terms update the core. The two outer strips use full K. */
    if (k != core_k)
        cob_cblas_add_depth(core_m, core_n, k-core_k, a+core_k, lda,
            b+(size_t)core_k*ldb, ldb, c, ldc);
    if (n != core_n)
        cob_strassen_leaf(core_m, n-core_n, k, a, lda, b+core_n, ldb, c+core_n, ldc);
    if (m != core_m)
        cob_strassen_leaf(m-core_m, n, k, a+(size_t)core_m*lda, lda, b, ldb,
            c+(size_t)core_m*ldc, ldc);
    return 1;
}
