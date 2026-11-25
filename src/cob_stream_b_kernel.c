/* This translation unit uses the ISA flags of its private CMake/Make object.
   The public entry checks CPU and OS support before calling either variant. */
#include <immintrin.h>
#include <stddef.h>
#include <string.h>

#if COB_STREAM_B_AVX512
#define COB_STREAM_B_KERNEL cob_stream_b_kernel_avx512
#else
#define COB_STREAM_B_KERNEL cob_stream_b_kernel_avx2
#endif

/* A single K-by-NR panel of B serves every row tile. Four vectors across C
   share each A broadcast. Accumulators span all of K, including the fringe. */
#if COB_STREAM_B_AVX512
typedef __m512 cob_sb_vec;
enum { cob_sb_rows = 6, cob_sb_lanes = 16 };
#define COB_SB_ZERO _mm512_setzero_ps
#define COB_SB_SPLAT _mm512_set1_ps
#define COB_SB_FMA _mm512_fmadd_ps
#define COB_SB_LOAD _mm512_load_ps
#define COB_SB_STORE _mm512_storeu_ps
#else
typedef __m256 cob_sb_vec;
enum { cob_sb_rows = 3, cob_sb_lanes = 8 };
#define COB_SB_ZERO _mm256_setzero_ps
#define COB_SB_SPLAT _mm256_set1_ps
#define COB_SB_FMA _mm256_fmadd_ps
#define COB_SB_LOAD _mm256_load_ps
#define COB_SB_STORE _mm256_storeu_ps
#endif

enum { cob_sb_cols = 4 * cob_sb_lanes };

/* Each call below specializes rows to a constant, including partial tiles. */
__attribute__((always_inline)) static inline void cob_sb_tile(int rows, int k,
    const float* a, int lda, const float* panel, float* c, int ldc)
{
    cob_sb_vec accum[4][cob_sb_rows];
    for (int q = 0; q < 4; ++q)
        for (int i = 0; i < rows; ++i) accum[q][i] = COB_SB_ZERO();
    for (int p = 0; p < k; ++p) {
        cob_sb_vec left[cob_sb_rows];
        for (int i = 0; i < rows; ++i) left[i] = COB_SB_SPLAT(a[(size_t)i*lda+p]);
        for (int q = 0; q < 4; ++q) {
            const cob_sb_vec right = COB_SB_LOAD(panel + q*cob_sb_lanes);
            for (int i = 0; i < rows; ++i)
                accum[q][i] = COB_SB_FMA(left[i], right, accum[q][i]);
        }
        panel += cob_sb_cols;
    }
    for (int i = 0; i < rows; ++i)
        for (int q = 0; q < 4; ++q)
            COB_SB_STORE(c + (size_t)i*ldc+q*cob_sb_lanes, accum[q][i]);
}

/* Name each accumulator separately. GCC otherwise materializes the entire
   array on the stack around the inline instruction groups. */
__attribute__((noinline)) static void cob_sb_full(int k, const float* a, int lda,
    const float* panel, float* c, int ldc)
{
#if COB_STREAM_B_AVX512
#define COB_SB_EACH_ROW(F) F(0) F(1) F(2) F(3) F(4) F(5)
#define COB_SB_INIT(i) \
    cob_sb_vec r##i##a = COB_SB_ZERO(), r##i##b = COB_SB_ZERO(); \
    cob_sb_vec r##i##c = COB_SB_ZERO(), r##i##d = COB_SB_ZERO();
    COB_SB_EACH_ROW(COB_SB_INIT)
#undef COB_SB_INIT
    for (int p = 0; p < k; ++p) {
        const cob_sb_vec b0 = COB_SB_LOAD(panel), b1 = COB_SB_LOAD(panel+cob_sb_lanes);
        const cob_sb_vec b2 = COB_SB_LOAD(panel+2*cob_sb_lanes), b3 = COB_SB_LOAD(panel+3*cob_sb_lanes);
#define COB_SB_ROW(i) do { \
        cob_sb_vec av; \
        __asm__("vbroadcastss {%5, %4|%4, %5}\n\t" \
            "vfmadd231ps {%6, %4, %0|%0, %4, %6}\n\t" "vfmadd231ps {%7, %4, %1|%1, %4, %7}\n\t" \
            "vfmadd231ps {%8, %4, %2|%2, %4, %8}\n\t" "vfmadd231ps {%9, %4, %3|%3, %4, %9}" \
            : "+&v"(r##i##a), "+&v"(r##i##b), "+&v"(r##i##c), "+&v"(r##i##d), "=&v"(av) \
            : "m"(a[(size_t)(i)*lda+p]), "v"(b0), "v"(b1), "v"(b2), "v"(b3)); \
        } while (0);
        COB_SB_EACH_ROW(COB_SB_ROW)
#undef COB_SB_ROW
        panel += cob_sb_cols;
    }
#define COB_SB_WRITE(i) do { \
        float* row = c+(size_t)(i)*ldc; \
        COB_SB_STORE(row, r##i##a); COB_SB_STORE(row+cob_sb_lanes, r##i##b); \
        COB_SB_STORE(row+2*cob_sb_lanes, r##i##c); COB_SB_STORE(row+3*cob_sb_lanes, r##i##d); \
    } while (0);
    COB_SB_EACH_ROW(COB_SB_WRITE)
#undef COB_SB_WRITE
#undef COB_SB_EACH_ROW
#else
#define COB_SB_EACH_COL(F) F(0) F(1) F(2) F(3)
#define COB_SB_INIT(q) cob_sb_vec c##q##a = COB_SB_ZERO(), c##q##b = COB_SB_ZERO(), c##q##c = COB_SB_ZERO();
    COB_SB_EACH_COL(COB_SB_INIT)
#undef COB_SB_INIT
    for (int p = 0; p < k; ++p) {
        const cob_sb_vec a0 = COB_SB_SPLAT(a[p]), a1 = COB_SB_SPLAT(a[(size_t)lda+p]);
        const cob_sb_vec a2 = COB_SB_SPLAT(a[2*(size_t)lda+p]);
#define COB_SB_COL(q) do { \
        cob_sb_vec bv; \
        __asm__("vmovaps {%4, %3|%3, %4}\n\t" "vfmadd231ps {%5, %3, %0|%0, %3, %5}\n\t" \
            "vfmadd231ps {%6, %3, %1|%1, %3, %6}\n\t" "vfmadd231ps {%7, %3, %2|%2, %3, %7}" \
            : "+&x"(c##q##a), "+&x"(c##q##b), "+&x"(c##q##c), "=&x"(bv) \
            : "m"(*(const __m256*)(panel+(q)*cob_sb_lanes)), "x"(a0), "x"(a1), "x"(a2)); \
        } while (0);
        COB_SB_EACH_COL(COB_SB_COL)
#undef COB_SB_COL
        panel += cob_sb_cols;
    }
#define COB_SB_WRITE(q) do { \
        float* col = c+(q)*cob_sb_lanes; \
        COB_SB_STORE(col, c##q##a); COB_SB_STORE(col+ldc, c##q##b); \
        COB_SB_STORE(col+2*(size_t)ldc, c##q##c); \
    } while (0);
    COB_SB_EACH_COL(COB_SB_WRITE)
#undef COB_SB_WRITE
#undef COB_SB_EACH_COL
#endif
}

void COB_STREAM_B_KERNEL(int m, int n, int k, const float* a, int lda,
    const float* b, int ldb, float* c, int ldc, float* panel)
{
    for (int j = 0; j < n; j += cob_sb_cols) {
        /* An overlapping last full-width tile recomputes at most NR-1 columns.
           Earlier C is never used and the entire tile stays inside each row. */
        const int col = (j < n-cob_sb_cols) ? j : n-cob_sb_cols;
        for (int p = 0; p < k; ++p)
            memcpy(panel+(size_t)p*cob_sb_cols, b+(size_t)p*ldb+col, cob_sb_cols*sizeof(float));
        int i = 0;
        for (; i + cob_sb_rows <= m; i += cob_sb_rows) {
            cob_sb_full(k, a+(size_t)i*lda, lda, panel, c+(size_t)i*ldc+col, ldc);
        }
        if (i < m) {
            const float* tail_a = a+(size_t)i*lda;
            float* tail_c = c+(size_t)i*ldc+col;
#define COB_SB_TAIL(rows) case rows: cob_sb_tile(rows, k, tail_a, lda, panel, tail_c, ldc); break
            switch (m-i) {
                COB_SB_TAIL(1);
                COB_SB_TAIL(2);
#if COB_STREAM_B_AVX512
                COB_SB_TAIL(3);
                COB_SB_TAIL(4);
                COB_SB_TAIL(5);
#endif
            }
#undef COB_SB_TAIL
        }
    }
}
#undef COB_SB_ZERO
#undef COB_SB_SPLAT
#undef COB_SB_FMA
#undef COB_SB_LOAD
#undef COB_SB_STORE
