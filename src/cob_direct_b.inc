/* For medium rectangles, pack only A. Traverse B directly in 32-column
   strips, limiting the working set of strided rows with a short K panel. */
#ifndef COB_DIRECT_B_KC
#define COB_DIRECT_B_KC 96
#endif
__attribute__((target("avx512f,fma"), always_inline))
static inline void cob_direct_b_tile(int kb, const float *a, const float *b,
    int ldb, float *c, int ldc, int rows, int cols, int resume, int work_rows)
{
    const __mmask16 m0 = cols >= 16 ? 0xffffu : (1u << cols) - 1u;
    const __mmask16 m1 = cols <= 16 ? 0u : (1u << (cols-16)) - 1u;
    __m512 lo[8], hi[8];
    for (int r = 0; r < work_rows; ++r) {
        lo[r] = _mm512_setzero_ps();
        hi[r] = _mm512_setzero_ps();
        if (resume && r < rows) {
            lo[r] = _mm512_maskz_loadu_ps(m0, c + (size_t)r*ldc);
            if (m1) hi[r] = _mm512_maskz_loadu_ps(m1, c + (size_t)r*ldc + 16);
        }
    }
    for (int p = 0; p < kb; ++p) {
        const float *br = b + (size_t)p * ldb;
        const __m512 b0 = _mm512_maskz_loadu_ps(m0, br);
        const __m512 b1 = m1 ? _mm512_maskz_loadu_ps(m1, br+16) : _mm512_setzero_ps();
        for (int r = 0; r < work_rows; ++r) {
            const __m512 av = _mm512_set1_ps(a[(size_t)p*8+r]);
            lo[r] = _mm512_fmadd_ps(av,b0,lo[r]);
            hi[r] = _mm512_fmadd_ps(av,b1,hi[r]);
        }
    }
    for (int r = 0; r < rows; ++r) {
        _mm512_mask_storeu_ps(c + (size_t)r*ldc,m0,lo[r]);
        if (m1) _mm512_mask_storeu_ps(c + (size_t)r*ldc+16,m1,hi[r]);
    }
}

__attribute__((target("avx512f,fma")))
static int cob_sgemm_direct_b(int m, int n, int k, const float *a, int lda,
    const float *b, int ldb, float *c, int ldc)
{
    const int depth = cob_min_i32(COB_DIRECT_B_KC, k);
    float *pack = cob_aligned_alloc(64, (size_t)((m+7)/8*8)*depth*sizeof(float));
    if (!pack) return 0;
    for (int p = 0; p < k; p += depth) {
        const int kb = cob_min_i32(k-p,depth);
        cob_blocked_pack_a(pack, a+p, lda, m, kb, 8);
        for (int j = 0; j < n; j += 32) {
            const int nc = cob_min_i32(n-j,32);
            for (int i = 0; i < m; i += 8) {
                const int mr = cob_min_i32(m-i,8);
                const float *ap = pack + (size_t)i*kb;
                const float *bp = b + (size_t)p*ldb + j;
                float *cp = c + (size_t)i*ldc+j;
                if (mr == 8 && nc == 32)
                    cob_direct_b_tile(kb,ap,bp,ldb,cp,ldc,8,32,p!=0,8);
                else if (mr <= 4)
                    cob_direct_b_tile(kb,ap,bp,ldb,cp,ldc,mr,nc,p!=0,4);
                else
                    cob_direct_b_tile(kb,ap,bp,ldb,cp,ldc,mr,nc,p!=0,8);
            }
        }
    }
    free(pack);
    return 1;
}
