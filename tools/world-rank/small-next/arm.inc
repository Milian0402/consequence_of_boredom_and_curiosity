#include <arm_neon.h>

/* Load four consecutive reduction values from each of eight A rows.
   Each FMA uses a single A lane and a B vector shared by all eight rows. */
#define SMALL_ROW8(F) F(0) F(1) F(2) F(3) F(4) F(5) F(6) F(7)
#define SMALL_DECLARE(r) float32x4_t c##r##a = vdupq_n_f32(0), c##r##b = vdupq_n_f32(0);
#define SMALL_LOAD_A(r) const float32x4_t x##r = vld1q_f32(a + (size_t)(r) * lda + p);
#define SMALL_FMA(r, lane) \
    c##r##a = vfmaq_laneq_f32(c##r##a, ba, x##r, lane); \
    c##r##b = vfmaq_laneq_f32(c##r##b, bb, x##r, lane);
#define SMALL_REDUCE(lane) do { \
    const float *bp = b + (size_t)(p + lane) * ldb; \
    const float32x4_t ba = vld1q_f32(bp), bb = vld1q_f32(bp + 4); \
    SMALL_FMA(0,lane) SMALL_FMA(1,lane) SMALL_FMA(2,lane) SMALL_FMA(3,lane) \
    SMALL_FMA(4,lane) SMALL_FMA(5,lane) SMALL_FMA(6,lane) SMALL_FMA(7,lane) \
} while (0)
#define SMALL_TAIL(r) do { \
    float ax = a[(size_t)(r) * lda + p]; \
    c##r##a = vfmaq_n_f32(c##r##a, ba, ax); \
    c##r##b = vfmaq_n_f32(c##r##b, bb, ax); \
} while (0);
#define SMALL_STORE(r) \
    vst1q_f32(c + (size_t)(r) * ldc, c##r##a); \
    vst1q_f32(c + (size_t)(r) * ldc + 4, c##r##b);

SMALL_INLINE void small_arm_tile(int k, const float *a, int lda, const float *b, int ldb,
                                 float *c, int ldc)
{
    SMALL_ROW8(SMALL_DECLARE)
    int p = 0;
    for (; k - p >= 4; p += 4) {
        SMALL_ROW8(SMALL_LOAD_A)
        SMALL_REDUCE(0);
        SMALL_REDUCE(1);
        SMALL_REDUCE(2);
        SMALL_REDUCE(3);
    }
    for (; p < k; ++p) {
        const float *bp = b + (size_t)p * ldb;
        const float32x4_t ba = vld1q_f32(bp), bb = vld1q_f32(bp + 4);
        SMALL_ROW8(SMALL_TAIL)
    }
    SMALL_ROW8(SMALL_STORE)
}
#undef SMALL_STORE
#undef SMALL_TAIL
#undef SMALL_REDUCE
#undef SMALL_FMA
#undef SMALL_LOAD_A
#undef SMALL_DECLARE
#undef SMALL_ROW8

SMALL_INLINE void small_arm_product(int m, int n, int k, const float *a, int lda,
                                    const float *b, int ldb, float *c, int ldc)
{
    for (int i = 0; i < m; i += 8) {
        /* Replaying an in-bounds final tile preserves padding and avoids
           heap workspace and a second scalar implementation for odd sizes. */
        int row = i <= m - 8 ? i : m - 8;
        for (int j = 0; j < n; j += 8) {
            int col = j <= n - 8 ? j : n - 8;
            small_arm_tile(k, a + (size_t)row * lda, lda, b + col, ldb,
                           c + (size_t)row * ldc + col, ldc);
        }
    }
}

static void small_native(int m, int n, int k, const float *a, int lda,
                         const float *b, int ldb, float *c, int ldc)
{
    /* Regular square family exposes row-stride immediates to the compiler.
       All other shapes, including independently padded squares, share the
       general row-stride implementation above. */
    if (m == n && n == k && lda == k && ldb == n && ldc == n) {
#define SMALL_SQUARE(d) case d: small_arm_product(d,d,d,a,d,b,d,c,d); return;
        switch (n) {
            SMALL_SQUARE(16) SMALL_SQUARE(32) SMALL_SQUARE(48) SMALL_SQUARE(64)
            SMALL_SQUARE(80) SMALL_SQUARE(96) SMALL_SQUARE(112) SMALL_SQUARE(128)
        }
#undef SMALL_SQUARE
    }
    small_arm_product(m,n,k,a,lda,b,ldb,c,ldc);
}
