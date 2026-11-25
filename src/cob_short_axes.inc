/* Native row-major kernels for up to four output columns or eight inner products. */
#if defined(__GNUC__) || defined(__clang__)
#define AXES_INLINE static inline __attribute__((always_inline))
#else
#define AXES_INLINE static inline
#endif

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#define AXES_HAVE_SIMD 1
#define AXES_LANES 4
#define AXES_WIDE_REGS 1
typedef float32x4_t axes_vec;
AXES_INLINE axes_vec axes_zero(void) { return vdupq_n_f32(0.0f); }
AXES_INLINE axes_vec axes_splat(float x) { return vdupq_n_f32(x); }
AXES_INLINE axes_vec axes_load(const float *p) { return vld1q_f32(p); }
AXES_INLINE void axes_store(float *p, axes_vec x) { vst1q_f32(p, x); }
AXES_INLINE axes_vec axes_mul(axes_vec a, axes_vec b) { return vmulq_f32(a, b); }
AXES_INLINE axes_vec axes_fma(axes_vec a, axes_vec b, axes_vec c) { return vfmaq_f32(c, a, b); }
AXES_INLINE float axes_sum(axes_vec x) { return vaddvq_f32(x); }
#elif defined(__SSE2__)
#include <immintrin.h>
#define AXES_HAVE_SIMD 1
#if defined(__AVX512F__)
#define AXES_LANES 16
#define AXES_WIDE_REGS 1
typedef __m512 axes_vec;
AXES_INLINE axes_vec axes_zero(void) { return _mm512_setzero_ps(); }
AXES_INLINE axes_vec axes_splat(float x) { return _mm512_set1_ps(x); }
AXES_INLINE axes_vec axes_load(const float *p) { return _mm512_loadu_ps(p); }
AXES_INLINE void axes_store(float *p, axes_vec x) { _mm512_storeu_ps(p, x); }
AXES_INLINE axes_vec axes_mul(axes_vec a, axes_vec b) { return _mm512_mul_ps(a, b); }
AXES_INLINE axes_vec axes_fma(axes_vec a, axes_vec b, axes_vec c) { return _mm512_fmadd_ps(a, b, c); }
AXES_INLINE float axes_sum(axes_vec x) { return _mm512_reduce_add_ps(x); }
#else
#define AXES_WIDE_REGS 0
AXES_INLINE float axes_sum4(__m128 x)
{
    __m128 pairs = _mm_add_ps(x, _mm_movehl_ps(x, x));
    return _mm_cvtss_f32(_mm_add_ss(pairs, _mm_shuffle_ps(pairs, pairs, 1)));
}
#if defined(__AVX2__)
#define AXES_LANES 8
typedef __m256 axes_vec;
AXES_INLINE axes_vec axes_zero(void) { return _mm256_setzero_ps(); }
AXES_INLINE axes_vec axes_splat(float x) { return _mm256_set1_ps(x); }
AXES_INLINE axes_vec axes_load(const float *p) { return _mm256_loadu_ps(p); }
AXES_INLINE void axes_store(float *p, axes_vec x) { _mm256_storeu_ps(p, x); }
AXES_INLINE axes_vec axes_mul(axes_vec a, axes_vec b) { return _mm256_mul_ps(a, b); }
AXES_INLINE axes_vec axes_fma(axes_vec a, axes_vec b, axes_vec c)
{
#ifdef __FMA__
    return _mm256_fmadd_ps(a, b, c);
#else
    return _mm256_add_ps(_mm256_mul_ps(a, b), c);
#endif
}
AXES_INLINE float axes_sum(axes_vec x)
{
    return axes_sum4(_mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1)));
}
#else
#define AXES_LANES 4
typedef __m128 axes_vec;
AXES_INLINE axes_vec axes_zero(void) { return _mm_setzero_ps(); }
AXES_INLINE axes_vec axes_splat(float x) { return _mm_set1_ps(x); }
AXES_INLINE axes_vec axes_load(const float *p) { return _mm_loadu_ps(p); }
AXES_INLINE void axes_store(float *p, axes_vec x) { _mm_storeu_ps(p, x); }
AXES_INLINE axes_vec axes_mul(axes_vec a, axes_vec b) { return _mm_mul_ps(a, b); }
AXES_INLINE axes_vec axes_fma(axes_vec a, axes_vec b, axes_vec c) { return _mm_add_ps(_mm_mul_ps(a, b), c); }
AXES_INLINE float axes_sum(axes_vec x) { return axes_sum4(x); }
#endif
#endif
#endif

#ifdef AXES_HAVE_SIMD
/* Each column of our transposed B is contiguous across K. One vector load per
   column serves up to four A rows. Tile dimensions are compile-time constants. */
typedef struct {
    axes_vec col0, col1, col2, col3;
} axes_columns;

AXES_INLINE axes_columns axes_load_columns(int cols, int k, const float *b)
{
    axes_columns v = {0};
    v.col0 = axes_load(b);
    if (cols >= 2) v.col1 = axes_load(b + k);
    if (cols >= 3) v.col2 = axes_load(b + (size_t)2 * k);
    if (cols >= 4) v.col3 = axes_load(b + (size_t)3 * k);
    return v;
}

AXES_INLINE void axes_dot_step(int cols, axes_columns *acc, const float *a, axes_columns b)
{
    const axes_vec x = axes_load(a);
    acc->col0 = axes_fma(x, b.col0, acc->col0);
    if (cols >= 2) acc->col1 = axes_fma(x, b.col1, acc->col1);
    if (cols >= 3) acc->col2 = axes_fma(x, b.col2, acc->col2);
    if (cols >= 4) acc->col3 = axes_fma(x, b.col3, acc->col3);
}

AXES_INLINE void axes_dot_finish(int cols, int k, int tail, const float *a,
                                  const float *b, axes_columns acc, float *c)
{
    float sum[4];
    sum[0] = axes_sum(acc.col0);
    if (cols >= 2) sum[1] = axes_sum(acc.col1);
    if (cols >= 3) sum[2] = axes_sum(acc.col2);
    if (cols >= 4) sum[3] = axes_sum(acc.col3);
    for (int j = 0; j < cols; ++j) {
        for (int p = tail; p < k; ++p) sum[j] += a[p] * b[(size_t)j * k + p];
        c[j] = sum[j];
    }
}

AXES_INLINE void axes_dot_tile(int rows, int cols, int k, const float *a, int lda,
                               const float *b, float *c, int ldc)
{
    axes_columns s0 = {0}, s1 = {0}, s2 = {0}, s3 = {0};
    int p = 0;
    for (; k - p >= AXES_LANES; p += AXES_LANES) {
        const axes_columns v = axes_load_columns(cols, k, b + p);
        axes_dot_step(cols, &s0, a + p, v);
        if (rows >= 2) axes_dot_step(cols, &s1, a + (size_t)lda + p, v);
        if (rows >= 3) axes_dot_step(cols, &s2, a + (size_t)2 * lda + p, v);
        if (rows >= 4) axes_dot_step(cols, &s3, a + (size_t)3 * lda + p, v);
    }
    axes_dot_finish(cols, k, p, a, b, s0, c);
    if (rows >= 2) axes_dot_finish(cols, k, p, a + lda, b, s1, c + ldc);
    if (rows >= 3) axes_dot_finish(cols, k, p, a + (size_t)2 * lda, b, s2, c + (size_t)2 * ldc);
    if (rows >= 4) axes_dot_finish(cols, k, p, a + (size_t)3 * lda, b, s3, c + (size_t)3 * ldc);
}

AXES_INLINE void axes_dot_rows(int cols, int m, int k, const float *a, int lda,
                               const float *b, float *c, int ldc)
{
    const int rows = AXES_WIDE_REGS || cols <= 2 ? 4 : 2;
    int i = 0;
    for (; m - i >= rows; i += rows)
        axes_dot_tile(rows, cols, k, a + (size_t)i * lda, lda, b, c + (size_t)i * ldc, ldc);
    for (; i < m; ++i)
        axes_dot_tile(1, cols, k, a + (size_t)i * lda, lda, b, c + (size_t)i * ldc, ldc);
}

static int axes_short_n(int m, int n, int k, const float *a, int lda,
                        const float *b, int ldb, float *c, int ldc)
{
    float *packed = NULL;
    if (n != 1 || ldb != 1) {
        packed = malloc((size_t)n * k * sizeof(float));
        if (!packed) return 0;
        for (int j = 0; j < n; ++j) {
            float *col = packed + (size_t)j * k;
            for (int p = 0; p < k; ++p) col[p] = b[(size_t)p * ldb + j];
        }
        b = packed;
    }
    switch (n) {
        case 1: axes_dot_rows(1, m, k, a, lda, b, c, ldc); break;
        case 2: axes_dot_rows(2, m, k, a, lda, b, c, ldc); break;
        case 3: axes_dot_rows(3, m, k, a, lda, b, c, ldc); break;
        case 4: axes_dot_rows(4, m, k, a, lda, b, c, ldc); break;
    }
    free(packed);
    return 1;
}

/* Direct row-major product for 1 <= K <= 8 and N >= AXES_LANES. */
#if defined(__GNUC__) || defined(__clang__)
#define AXES_K_INLINE static inline __attribute__((always_inline))
#else
#define AXES_K_INLINE static inline
#endif

/* The dispatch below supplies constant depth, row count and vector count.
   Reuse each vector of B for two output rows and each A value for two vectors. */
AXES_K_INLINE void axes_short_k_tile(
    int depth, int rows, int vectors, const float *a0, const float *a1,
    const float *b, int ldb, float *c0, float *c1)
{
    axes_vec x0 = axes_splat(a0[0]);
    axes_vec x1 = rows == 2 ? axes_splat(a1[0]) : axes_zero();
    axes_vec y0 = axes_load(b);
    axes_vec y1 = vectors == 2 ? axes_load(b + AXES_LANES) : axes_zero();
    axes_vec c00 = axes_mul(x0, y0);
    axes_vec c01 = axes_mul(x0, y1);
    axes_vec c10 = axes_mul(x1, y0);
    axes_vec c11 = axes_mul(x1, y1);
    for (int p = 1; p < depth; ++p) {
        const float *bp = b + (size_t)p * ldb;
        x0 = axes_splat(a0[p]);
        y0 = axes_load(bp);
        c00 = axes_fma(x0, y0, c00);
        if (rows == 2) {
            x1 = axes_splat(a1[p]);
            c10 = axes_fma(x1, y0, c10);
        }
        if (vectors == 2) {
            y1 = axes_load(bp + AXES_LANES);
            c01 = axes_fma(x0, y1, c01);
            if (rows == 2) c11 = axes_fma(x1, y1, c11);
        }
    }
    axes_store(c0, c00);
    if (vectors == 2) axes_store(c0 + AXES_LANES, c01);
    if (rows == 2) {
        axes_store(c1, c10);
        if (vectors == 2) axes_store(c1 + AXES_LANES, c11);
    }
}

AXES_K_INLINE void axes_short_k_rows(
    int depth, int rows, int n, const float *a0, const float *a1,
    const float *b, int ldb, float *restrict c0, float *restrict c1)
{
    int col = 0;
    for (; col <= n - 2 * AXES_LANES; col += 2 * AXES_LANES) {
        axes_short_k_tile(depth, rows, 2, a0, a1, b + col, ldb, c0 + col,
                          rows == 2 ? c1 + col : NULL);
    }
    if (col <= n - AXES_LANES) {
        axes_short_k_tile(depth, rows, 1, a0, a1, b + col, ldb, c0 + col,
                          rows == 2 ? c1 + col : NULL);
        col += AXES_LANES;
    }
    if (col < n) {
        /* Replay within the row to cover a partial last vector without
           touching the next row or padding. C is an overwrite-only output. */
        int last = n - AXES_LANES;
        axes_short_k_tile(depth, rows, 1, a0, a1, b + last, ldb, c0 + last,
                          rows == 2 ? c1 + last : NULL);
    }
}

AXES_K_INLINE void axes_short_k_fixed(
    int m, int n, int depth, const float *a, int lda, const float *b, int ldb,
    float *restrict c, int ldc)
{
    int row = 0;
    for (; row < m - 1; row += 2) {
        const float *ap = a + (size_t)row * lda;
        float *cp = c + (size_t)row * ldc;
        axes_short_k_rows(depth, 2, n, ap, ap + lda, b, ldb, cp, cp + ldc);
    }
    if (row < m) {
        axes_short_k_rows(depth, 1, n, a + (size_t)row * lda, NULL, b, ldb,
                          c + (size_t)row * ldc, NULL);
    }
}

static void axes_short_k(
    int m, int n, int k, const float *a, int lda, const float *b, int ldb,
    float *c, int ldc)
{
#define AXES_K_CASE(depth) case depth: \
        axes_short_k_fixed(m, n, depth, a, lda, b, ldb, c, ldc); return
    switch (k) {
        AXES_K_CASE(1);
        AXES_K_CASE(2);
        AXES_K_CASE(3);
        AXES_K_CASE(4);
        AXES_K_CASE(5);
        AXES_K_CASE(6);
        AXES_K_CASE(7);
        AXES_K_CASE(8);
    }
#undef AXES_K_CASE
}

#undef AXES_K_INLINE

#endif

/* Return zero without writing C on any unsupported input or allocation failure.
   A COB caller can then use its established general path. */
static int cob_try_short_axes(int m, int n, int k, const float *a, int lda,
                   const float *b, int ldb, float *c, int ldc)
{
#ifdef AXES_HAVE_SIMD
    if (m < 128 || n < 1 || k < 1 || !a || !b || !c ||
        lda < k || ldb < n || ldc < n) return 0;
    const int small_n = n <= 4 && k >= 128;
    const int small_k = n >= 128 && k <= 8;
    if (!small_n && !small_k) return 0;
    const size_t c_count = (size_t)(m - 1) * ldc + n;
    if (cob_f32_ranges_overlap(c, c_count, a, (size_t)(m - 1) * lda + k) ||
        cob_f32_ranges_overlap(c, c_count, b, (size_t)(k - 1) * ldb + n)) return 0;
    if (small_n) return axes_short_n(m, n, k, a, lda, b, ldb, c, ldc);
    axes_short_k(m, n, k, a, lda, b, ldb, c, ldc);
    return 1;
#else
    (void)m; (void)n; (void)k; (void)a; (void)lda;
    (void)b; (void)ldb; (void)c; (void)ldc;
    return 0;
#endif
}
