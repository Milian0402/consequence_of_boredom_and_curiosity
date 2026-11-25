/* No-pack row-major overwrite kernel. Caller supplies M,N,K >= 8 and
   disjoint output. Compile with the native target flags used by the study. */
#include <stddef.h>
#include <immintrin.h>

#if defined(__AVX512F__) && !defined(COB_DISABLE_X86_AVX512)
typedef __m512 small_x86_vec;
enum { small_x86_lanes = 16, small_x86_groups = 4 };
#define SMALL_X86_LOAD _mm512_loadu_ps
#define SMALL_X86_STORE _mm512_storeu_ps
#define SMALL_X86_SPLAT _mm512_set1_ps
#define SMALL_X86_ZERO _mm512_setzero_ps
#define SMALL_X86_FMA _mm512_fmadd_ps
#elif defined(__AVX2__) && defined(__FMA__)
typedef __m256 small_x86_vec;
enum { small_x86_lanes = 8, small_x86_groups = 2 };
#define SMALL_X86_LOAD _mm256_loadu_ps
#define SMALL_X86_STORE _mm256_storeu_ps
#define SMALL_X86_SPLAT _mm256_set1_ps
#define SMALL_X86_ZERO _mm256_setzero_ps
#define SMALL_X86_FMA _mm256_fmadd_ps
#else
typedef __m128 small_x86_vec;
enum { small_x86_lanes = 4, small_x86_groups = 2 };
#define SMALL_X86_LOAD _mm_loadu_ps
#define SMALL_X86_STORE _mm_storeu_ps
#define SMALL_X86_SPLAT _mm_set1_ps
#define SMALL_X86_ZERO _mm_setzero_ps
#define SMALL_X86_FMA(a, b, c) _mm_add_ps(_mm_mul_ps((a), (b)), (c))
#endif

/* Group count and masked/full loads are resolved in each inlined instance.
   Six rows keep 24 independent accumulators at AVX-512 width and 12 at
   AVX2 width. No lane reductions or horizontal output shuffles are needed. */
static inline __attribute__((always_inline)) void small_x86_tile(
    int groups, int valid, int k, const float *a, int lda,
    const float *b, int ldb, float *c, int ldc)
{
    const small_x86_vec zero = SMALL_X86_ZERO();
#define SMALL_X86_INIT(r) small_x86_vec r##a = zero, r##b = zero, \
                                         r##c = zero, r##d = zero
    SMALL_X86_INIT(q0);
    SMALL_X86_INIT(q1);
    SMALL_X86_INIT(q2);
    SMALL_X86_INIT(q3);
    SMALL_X86_INIT(q4);
    SMALL_X86_INIT(q5);
#undef SMALL_X86_INIT
#if defined(__AVX512F__) && !defined(COB_DISABLE_X86_AVX512)
    const __mmask16 mask = (__mmask16)((1u << valid) - 1u);
#else
    (void)valid;
#endif
    for (int p = 0; p < k; ++p) {
        const float *bp = b + (size_t)p * ldb;
        small_x86_vec b0;
#if defined(__AVX512F__) && !defined(COB_DISABLE_X86_AVX512)
        if (valid < small_x86_lanes) b0 = _mm512_maskz_loadu_ps(mask, bp);
        else
#endif
        b0 = SMALL_X86_LOAD(bp);
        const small_x86_vec b1 = groups >= 2 ? SMALL_X86_LOAD(bp + small_x86_lanes) : zero;
        const small_x86_vec b2 = groups >= 3 ? SMALL_X86_LOAD(bp + 2 * small_x86_lanes) : zero;
        const small_x86_vec b3 = groups >= 4 ? SMALL_X86_LOAD(bp + 3 * small_x86_lanes) : zero;
#define SMALL_X86_STEP(r, index) do { \
            const small_x86_vec av = SMALL_X86_SPLAT(a[(size_t)(index) * lda + p]); \
            r##a = SMALL_X86_FMA(av, b0, r##a); \
            if (groups >= 2) r##b = SMALL_X86_FMA(av, b1, r##b); \
            if (groups >= 3) r##c = SMALL_X86_FMA(av, b2, r##c); \
            if (groups >= 4) r##d = SMALL_X86_FMA(av, b3, r##d); \
        } while (0)
        SMALL_X86_STEP(q0, 0);
        SMALL_X86_STEP(q1, 1);
        SMALL_X86_STEP(q2, 2);
        SMALL_X86_STEP(q3, 3);
        SMALL_X86_STEP(q4, 4);
        SMALL_X86_STEP(q5, 5);
#undef SMALL_X86_STEP
    }
#if defined(__AVX512F__) && !defined(COB_DISABLE_X86_AVX512)
#define SMALL_X86_STORE_FIRST(cp, v) do { \
        if (valid < small_x86_lanes) _mm512_mask_storeu_ps((cp), mask, (v)); \
        else SMALL_X86_STORE((cp), (v)); \
    } while (0)
#else
#define SMALL_X86_STORE_FIRST(cp, v) SMALL_X86_STORE((cp), (v))
#endif
#define SMALL_X86_WRITE(r, index) do { \
        float *cp = c + (size_t)(index) * ldc; \
        SMALL_X86_STORE_FIRST(cp, r##a); \
        if (groups >= 2) SMALL_X86_STORE(cp + small_x86_lanes, r##b); \
        if (groups >= 3) SMALL_X86_STORE(cp + 2 * small_x86_lanes, r##c); \
        if (groups >= 4) SMALL_X86_STORE(cp + 3 * small_x86_lanes, r##d); \
    } while (0)
    SMALL_X86_WRITE(q0, 0);
    SMALL_X86_WRITE(q1, 1);
    SMALL_X86_WRITE(q2, 2);
    SMALL_X86_WRITE(q3, 3);
    SMALL_X86_WRITE(q4, 4);
    SMALL_X86_WRITE(q5, 5);
#undef SMALL_X86_WRITE
#undef SMALL_X86_STORE_FIRST
}

static void cob_small_x86_native(int m, int n, int k, const float *a, int lda,
                         const float *b, int ldb, float *c, int ldc)
{
    for (int end = 0; end < m;) {
        /* The last tile can overwrite already-computed rows, at most five. */
        const int i = end <= m - 6 ? end : m - 6;
        const float *ap = a + (size_t)i * lda;
        float *cp = c + (size_t)i * ldc;
        int j = 0;
        for (; j <= n - small_x86_groups * small_x86_lanes; j += small_x86_groups * small_x86_lanes) {
            small_x86_tile(small_x86_groups, small_x86_lanes, k, ap, lda, b + j, ldb, cp + j, ldc);
        }
        if (small_x86_groups == 4 && j <= n - 2 * small_x86_lanes) {
            small_x86_tile(2, small_x86_lanes, k, ap, lda, b + j, ldb, cp + j, ldc);
            j += 2 * small_x86_lanes;
        }
        if (j <= n - small_x86_lanes) {
            small_x86_tile(1, small_x86_lanes, k, ap, lda, b + j, ldb, cp + j, ldc);
            j += small_x86_lanes;
        }
        if (j < n) {
            if (n < small_x86_lanes) {
                small_x86_tile(1, n, k, ap, lda, b, ldb, cp, ldc);
            } else {
                const int last = n - small_x86_lanes;
                small_x86_tile(1, small_x86_lanes, k, ap, lda, b + last, ldb, cp + last, ldc);
            }
        }
        end = i + 6;
    }
}

#undef SMALL_X86_LOAD
#undef SMALL_X86_STORE
#undef SMALL_X86_SPLAT
#undef SMALL_X86_ZERO
#undef SMALL_X86_FMA

static int cob_try_small_x86(int m, int n, int k, const float *a, int lda,
                             const float *b, int ldb, float *c, int ldc)
{
    if (m < 8 || m > 128 || n < 8 || n > 128 || k < 8 || k > 128 ||
        a == NULL || b == NULL || c == NULL || lda < k || ldb < n || ldc < n)
        return 0;

    const size_t nc = (size_t)(m - 1) * ldc + n;
    if (cob_f32_ranges_overlap(c, nc, a, (size_t)(m - 1) * lda + k) ||
        cob_f32_ranges_overlap(c, nc, b, (size_t)(k - 1) * ldb + n))
        return 0;
    cob_small_x86_native(m, n, k, a, lda, b, ldb, c, ldc);
    return 1;
}
