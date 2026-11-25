/* The right operand is stored in small K-major panels. The left operand
   stays in the caller's layout, with one broadcast stream per output row. */
#if defined(__AVX512F__) && !defined(COB_DISABLE_X86_AVX512)
#define COB_DA_MR 10
#define COB_DA_LANES 16
#define COB_DA_TARGET __attribute__((target("avx512f,fma")))
typedef __m512 cob_da_vec;
#define COB_DA_LOAD _mm512_load_ps
#define COB_DA_LOADU _mm512_loadu_ps
#define COB_DA_STOREU _mm512_storeu_ps
#define COB_DA_ZERO _mm512_setzero_ps
#define COB_DA_SPLAT _mm512_set1_ps
#define COB_DA_FMA _mm512_fmadd_ps
#else
#define COB_DA_MR 6
#define COB_DA_LANES 8
#define COB_DA_TARGET __attribute__((target("avx2,fma")))
typedef __m256 cob_da_vec;
#define COB_DA_LOAD _mm256_load_ps
#define COB_DA_LOADU _mm256_loadu_ps
#define COB_DA_STOREU _mm256_storeu_ps
#define COB_DA_ZERO _mm256_setzero_ps
#define COB_DA_SPLAT _mm256_set1_ps
#define COB_DA_FMA _mm256_fmadd_ps
#endif

#define COB_DA_NR (2 * COB_DA_LANES)
#define COB_DA_BASE(S) S(0) S(1) S(2) S(3) S(4) S(5)
#if COB_DA_MR == 10
#define COB_DA_ROWS(S) COB_DA_BASE(S) S(6) S(7) S(8) S(9)
#else
#define COB_DA_ROWS(S) COB_DA_BASE(S)
#endif

COB_DA_TARGET
static void cob_da_full(int k, const float* restrict a, int lda,
    const float* restrict pack, float* restrict c, int ldc, int resume)
{
#define INIT(r) \
    const float* a##r = a + (size_t)(r) * lda; \
    cob_da_vec l##r = resume ? COB_DA_LOADU(c + (size_t)(r)*ldc) : COB_DA_ZERO(); \
    cob_da_vec h##r = resume ? COB_DA_LOADU(c + (size_t)(r)*ldc + COB_DA_LANES) : COB_DA_ZERO();
    COB_DA_ROWS(INIT)
#undef INIT
    for (int p = 0; p < k; ++p) {
        const cob_da_vec lo = COB_DA_LOAD(pack);
        const cob_da_vec hi = COB_DA_LOAD(pack + COB_DA_LANES);
#define FMA(r) do { \
        const cob_da_vec v = COB_DA_SPLAT(a##r[p]); \
        l##r = COB_DA_FMA(v, lo, l##r); \
        h##r = COB_DA_FMA(v, hi, h##r); \
    } while (0);
        COB_DA_ROWS(FMA)
#undef FMA
        pack += COB_DA_NR;
    }
#define STORE(r) \
    COB_DA_STOREU(c + (size_t)(r)*ldc, l##r); \
    COB_DA_STOREU(c + (size_t)(r)*ldc + COB_DA_LANES, h##r);
    COB_DA_ROWS(STORE)
#undef STORE
}

/* B's ragged panel has explicit zero fill. A remains in the original
   allocation even on this edge. Read only real rows and write valid C. */
COB_DA_TARGET
static void cob_da_edge(int rows, int cols, int k, const float* a, int lda,
    const float* pack, float* c, int ldc, int resume)
{
    for (int row = 0; row < rows; ++row) {
        float* out = c + (size_t)row * ldc;
        _Alignas(64) float edge[COB_DA_NR] = {0};
        if (resume) memcpy(edge, out, (size_t)cols * sizeof(float));
        cob_da_vec lo = COB_DA_LOAD(edge), hi = COB_DA_LOAD(edge + COB_DA_LANES);
        const float* ap = a + (size_t)row * lda;
        const float* bp = pack;
        for (int p = 0; p < k; ++p) {
            const cob_da_vec av = COB_DA_SPLAT(ap[p]);
            lo = COB_DA_FMA(av, COB_DA_LOAD(bp), lo);
            hi = COB_DA_FMA(av, COB_DA_LOAD(bp + COB_DA_LANES), hi);
            bp += COB_DA_NR;
        }
        COB_DA_STOREU(edge, lo);
        COB_DA_STOREU(edge + COB_DA_LANES, hi);
        memcpy(out, edge, (size_t)cols * sizeof(float));
    }
}

static void cob_da_pack(float* dst, const float* src, int ldb, int n, int k)
{
    for (int p0 = 0; p0 < k; p0 += 16) {
        const int p_end = cob_min_i32(k, p0 + 16);
        int j = 0;
        for (; j + COB_DA_NR <= n; j += COB_DA_NR) {
            float* panel = dst + (size_t)j*k;
            for (int p = p0; p < p_end; ++p) {
                memcpy(panel + (size_t)p*COB_DA_NR, src + (size_t)p*ldb + j,
                    COB_DA_NR * sizeof(float));
            }
        }
        if (j < n) {
            const size_t cols = (size_t)(n - j);
            float* panel = dst + (size_t)j*k;
            for (int p = p0; p < p_end; ++p) {
                float* row = panel + (size_t)p*COB_DA_NR;
                memcpy(row, src + (size_t)p*ldb + j, cols*sizeof(float));
                memset(row + cols, 0, (COB_DA_NR-cols)*sizeof(float));
            }
        }
    }
}

static int cob_try_medium_direct_a(int m, int n, int k, const float* a, int lda,
    const float* b, int ldb, float* c, int ldc)
{
    if (m < 16 || n < 16 || k < 32 || (m > 1024 && n > 1024 && k > 1024) ||
        !a || !b || !c || lda < k || ldb < n || ldc < n ||
#if COB_DA_LANES == 16
        !__builtin_cpu_supports("avx512f") ||
#else
        !__builtin_cpu_supports("avx2") ||
#endif
        !__builtin_cpu_supports("fma")) return 0;
    const size_t c_span = (size_t)(m - 1) * ldc + n;
    if (cob_f32_ranges_overlap(c, c_span, a, (size_t)(m-1)*lda+k) ||
        cob_f32_ranges_overlap(c, c_span, b, (size_t)(k-1)*ldb+n)) return 0;
    const int kc = cob_min_i32(k, 256), nc = cob_min_i32(n, 256);
    const int panel_width = ((nc + COB_DA_NR - 1) / COB_DA_NR) * COB_DA_NR;
    float* pack = cob_aligned_alloc(64, (size_t)panel_width*kc*sizeof(float));
    if (!pack) return 0;
    for (int j = 0; j < n;) {
        const int nb = cob_min_i32(n-j, nc);
        for (int p = 0; p < k;) {
            const int kb = cob_min_i32(k-p, kc);
            cob_da_pack(pack, b + (size_t)p*ldb+j, ldb, nb, kb);
            for (int ib=0; ib<m;) {
                const int mb = cob_min_i32(12*COB_DA_MR,m-ib);
                const float* ap = a + (size_t)ib*lda+p;
                float* cp = c + (size_t)ib*ldc+j;
                for (int jj = 0; jj < nb; jj += COB_DA_NR) {
                    const int cols = cob_min_i32(COB_DA_NR,nb-jj);
                    const float* bp = pack + (size_t)jj*kb;
                    int i=0;
                    if (cols == COB_DA_NR) {
                        for (; i + COB_DA_MR <= mb; i += COB_DA_MR) {
                            cob_da_full(kb, ap + (size_t)i*lda, lda, bp,
                                cp + (size_t)i*ldc+jj, ldc, p != 0);
                        }
                    }
                    if (i < mb) {
                        cob_da_edge(mb-i, cols, kb, ap + (size_t)i*lda, lda, bp,
                            cp + (size_t)i*ldc+jj, ldc, p != 0);
                    }
                }
                ib+=mb;
            }
            p+=kb;
        }
        j+=nb;
    }
    free(pack);
    return 1;
}
#undef COB_DA_ROWS
#undef COB_DA_BASE
#undef COB_DA_NR
#undef COB_DA_MR
#undef COB_DA_LANES
#undef COB_DA_TARGET
#undef COB_DA_LOAD
#undef COB_DA_LOADU
#undef COB_DA_STOREU
#undef COB_DA_ZERO
#undef COB_DA_SPLAT
#undef COB_DA_FMA
