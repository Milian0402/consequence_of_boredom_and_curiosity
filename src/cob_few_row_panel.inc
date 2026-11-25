/* One-shot FP32 row-major SGEMM call for 9..15 rows. Pack a small A panel
   once, then stream unmodified B across all of its output columns. */
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define SHORT_INLINE static inline __attribute__((always_inline))
#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
typedef float32x4_t short_vec;
enum { short_lanes = 4, short_rows = 12 };
#define SHORT_HAVE_SIMD 1
#define SHORT_LOAD vld1q_f32
#define SHORT_STORE vst1q_f32
#define SHORT_ZERO() vdupq_n_f32(0)
#define SHORT_FMA_ROW(r) do { \
    float32x4_t av = vld1q_f32(ap + ((r) & ~3)); \
    l##r = vfmaq_laneq_f32(l##r, b0, av, (r) & 3); \
    h##r = vfmaq_laneq_f32(h##r, b1, av, (r) & 3); \
} while (0)
#elif defined(__SSE2__)
#include <immintrin.h>
#define SHORT_HAVE_SIMD 1
#if defined(__AVX512F__) && !defined(COB_DISABLE_X86_AVX512)
typedef __m512 short_vec;
enum { short_lanes = 16, short_rows = 12 };
#define SHORT_LOAD _mm512_loadu_ps
#define SHORT_STORE _mm512_storeu_ps
#define SHORT_SPLAT _mm512_set1_ps
#define SHORT_ZERO _mm512_setzero_ps
#define SHORT_FMA _mm512_fmadd_ps
#elif defined(__AVX2__) && defined(__FMA__)
typedef __m256 short_vec;
enum { short_lanes = 8, short_rows = 6 };
#define SHORT_LOAD _mm256_loadu_ps
#define SHORT_STORE _mm256_storeu_ps
#define SHORT_SPLAT _mm256_set1_ps
#define SHORT_ZERO _mm256_setzero_ps
#define SHORT_FMA _mm256_fmadd_ps
#else
typedef __m128 short_vec;
enum { short_lanes = 4, short_rows = 6 };
#define SHORT_LOAD _mm_loadu_ps
#define SHORT_STORE _mm_storeu_ps
#define SHORT_SPLAT _mm_set1_ps
#define SHORT_ZERO _mm_setzero_ps
#define SHORT_FMA(a,b,c) _mm_add_ps(_mm_mul_ps((a),(b)),(c))
#endif
#define SHORT_FMA_ROW(r) do { \
    short_vec av = SHORT_SPLAT(ap[r]); \
    l##r = SHORT_FMA(b0, av, l##r); \
    h##r = SHORT_FMA(b1, av, h##r); \
} while (0)
#endif

#ifdef SHORT_HAVE_SIMD
enum { short_nr = short_lanes * 2, short_kc = 96, short_mr = 16 };
#define SHORT_ALL(F) F(0) F(1) F(2) F(3) F(4) F(5) F(6) F(7) F(8) F(9) F(10) F(11)

SHORT_INLINE void short_tile(int rows, int k, const float *packed, const float *b,
                             int ldb, float *c, int ldc, int resume, int skip)
{
#define SHORT_INIT(r) \
    short_vec l##r = rows > (r) && resume ? SHORT_LOAD(c + (size_t)(r) * ldc) : SHORT_ZERO(); \
    short_vec h##r = rows > (r) && resume ? SHORT_LOAD(c + (size_t)(r) * ldc + short_lanes) : SHORT_ZERO();
    SHORT_ALL(SHORT_INIT)
#undef SHORT_INIT
    for (int p = 0; p < k; ++p) {
        const float *ap = packed + (size_t)p * short_mr;
        const float *bp = b + (size_t)p * ldb;
        const short_vec b0 = SHORT_LOAD(bp), b1 = SHORT_LOAD(bp + short_lanes);
#define SHORT_STEP(r) if (rows > (r)) { SHORT_FMA_ROW(r); }
        SHORT_ALL(SHORT_STEP)
#undef SHORT_STEP
    }
#define SHORT_WRITE(r) if (rows > (r)) { \
    float *cp = c + (size_t)(r) * ldc; \
    if (skip == 0) { \
        SHORT_STORE(cp, l##r); SHORT_STORE(cp + short_lanes, h##r); \
    } else { \
        float tile[short_nr]; \
        SHORT_STORE(tile, l##r); SHORT_STORE(tile + short_lanes, h##r); \
        memcpy(cp + skip, tile + skip, (short_nr - (size_t)skip) * sizeof(float)); \
    } \
}
    SHORT_ALL(SHORT_WRITE)
#undef SHORT_WRITE
}
#undef SHORT_ALL

SHORT_INLINE void short_strip(int rows, int n, int k, const float *packed,
                              const float *b, int ldb, float *c, int ldc, int resume)
{
    int j = 0;
    for (; j <= n - short_nr; j += short_nr)
        short_tile(rows,k,packed,b + j,ldb,c + j,ldc,resume,0);
    if (j < n) {
        /* Replay the final in-bounds B vectors, but only write new columns.
           Previous full vectors in this panel have already accumulated it. */
        int last = n - short_nr;
        short_tile(rows,k,packed,b + last,ldb,c + last,ldc,resume,j-last);
    }
}

static void short_strips(int rows, int n, int k, const float *packed,
                         const float *b, int ldb, float *c, int ldc, int resume)
{
#define SHORT_CASE(r) case r: short_strip(r,n,k,packed,b,ldb,c,ldc,resume); break;
    switch (rows) {
        SHORT_CASE(1) SHORT_CASE(2) SHORT_CASE(3) SHORT_CASE(4)
        SHORT_CASE(5) SHORT_CASE(6) SHORT_CASE(7) SHORT_CASE(8)
        SHORT_CASE(9) SHORT_CASE(10) SHORT_CASE(11) SHORT_CASE(12)
    }
#undef SHORT_CASE
}

#endif

static int cob_try_panel_few_rows(int m, int n, int k, const float *a, int lda,
                    const float *b, int ldb, float *c, int ldc)
{
#ifdef SHORT_HAVE_SIMD
#if defined(__aarch64__)
    /* A single NEON row tile covers twelve rows. Split tiles lost to the packed kernel. */
    const int max_rows = 12;
#else
    const int max_rows = 15;
#endif
    if (m < 9 || m > max_rows || n < 128 || n >= 1024 || k < 64 || k >= 1024 ||
        (n <= 128 && k <= 128) ||
        !a || !b || !c || lda < k || ldb < n || ldc < n) return 0;
    size_t count = (size_t)(m - 1) * ldc + n;
    if (cob_f32_ranges_overlap(c,count,a,(size_t)(m - 1)*lda+k) ||
        cob_f32_ranges_overlap(c,count,b,(size_t)(k - 1)*ldb+n)) return 0;
    /* Initialize the last four-row group for the NEON lane loads. The packed
       pitch stays 16 on every ISA and the entire buffer uses only 6 KiB. */
    _Alignas(64) float packed[short_kc * short_mr] = {0};
    for (int p = 0; p < k; p += short_kc) {
        int kb = k - p < short_kc ? k - p : short_kc;
        for (int q = 0; q < kb; ++q)
            for (int r = 0; r < m; ++r)
                packed[(size_t)q * short_mr + r] = a[(size_t)r * lda + p + q];
        for (int i = 0; i < m;) {
            int rows = m - i;
            if (rows > short_rows) rows = short_rows == 12 ? 8 : short_rows;
            short_strips(rows,n,kb,packed + i,b + (size_t)p * ldb,ldb,
                         c + (size_t)i * ldc,ldc,p != 0);
            i += rows;
        }
    }
    return 1;
#else
    (void)m; (void)n; (void)k; (void)a; (void)lda; (void)b; (void)ldb;
    (void)c; (void)ldc;
    return 0;
#endif
}
