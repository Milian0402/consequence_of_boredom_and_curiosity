/* A four-row strip combines the last full 16-column panel and its 8-column
   neighbor. Each row holds six vector accumulators throughout K. */
#define N2_24_FMA(acc, bv, av, lane) "fmla v" #acc ".4s,v" #bv ".4s,v" #av ".s[" #lane "]\n"
#define N2_24_PAIR(off, c00,c01,c10,c11,c20,c21,c30,c31, lane) \
    "ldp q4,q5,[%[bp],#" #off "]\n" \
    N2_24_FMA(c00,4,0,lane) N2_24_FMA(c01,5,0,lane) \
    N2_24_FMA(c10,4,1,lane) N2_24_FMA(c11,5,1,lane) \
    N2_24_FMA(c20,4,2,lane) N2_24_FMA(c21,5,2,lane) \
    N2_24_FMA(c30,4,3,lane) N2_24_FMA(c31,5,3,lane)
#define N2_24_STEP(lane) \
    N2_24_PAIR(0,  8, 9,14,15,20,21,26,27,lane) \
    N2_24_PAIR(32,10,11,16,17,22,23,28,29,lane) \
    N2_24_PAIR(64,12,13,18,19,24,25,30,31,lane) \
    "add %[bp],%[bp],%[bstep]\n"
#define N2_24_A(kind,step) \
    "ldr " #kind "0,[%[a0]],#" #step "\n" "ldr " #kind "1,[%[a1]],#" #step "\n" \
    "ldr " #kind "2,[%[a2]],#" #step "\n" "ldr " #kind "3,[%[a3]],#" #step "\n"
#define N2_24_ZERO(q) "movi v" #q ".4s,#0\n"
#define N2_24_CLEAR(a,b,c,d,e,f) \
    N2_24_ZERO(a) N2_24_ZERO(b) N2_24_ZERO(c) N2_24_ZERO(d) N2_24_ZERO(e) N2_24_ZERO(f)
#define N2_24_STORE(a,b,c,d,e,f) \
    "stp q" #a ",q" #b ",[%[cp]]\n" "stp q" #c ",q" #d ",[%[cp],#32]\n" \
    "stp q" #e ",q" #f ",[%[cp],#64]\n" "add %[cp],%[cp],%[cstep]\n"

N2_INLINE void n2_full_4x24(int k, const float *a, int lda, const float *b,
                            int ldb, float *c, int ldc)
{
    const float *a0 = a, *a1 = a + lda;
    const float *a2 = a + (size_t)2 * lda, *a3 = a + (size_t)3 * lda;
    int blocks = k / 4, rest = k % 4;
    uintptr_t bstep = (size_t)ldb * sizeof(float), cstep = (size_t)ldc * sizeof(float);
    __asm__ volatile (
        N2_24_CLEAR(8,9,10,11,12,13) N2_24_CLEAR(14,15,16,17,18,19)
        N2_24_CLEAR(20,21,22,23,24,25) N2_24_CLEAR(26,27,28,29,30,31)
        "1:\n" N2_24_A(q,16)
        N2_24_STEP(0) N2_24_STEP(1) N2_24_STEP(2) N2_24_STEP(3)
        "subs %w[blocks],%w[blocks],#1\n" "b.ne 1b\n" "cbz %w[rest],3f\n"
        "2:\n" N2_24_A(s,4) N2_24_STEP(0)
        "subs %w[rest],%w[rest],#1\n" "b.ne 2b\n"
        "3:\n" N2_24_STORE(8,9,10,11,12,13) N2_24_STORE(14,15,16,17,18,19)
        N2_24_STORE(20,21,22,23,24,25) N2_24_STORE(26,27,28,29,30,31)
        : [a0] "+&r"(a0), [a1] "+&r"(a1), [a2] "+&r"(a2), [a3] "+&r"(a3),
          [bp] "+&r"(b), [cp] "+&r"(c), [blocks] "+&r"(blocks), [rest] "+&r"(rest)
        : [bstep] "r"(bstep), [cstep] "r"(cstep)
        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5",
          "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
          "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
          "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
}
#undef N2_24_STORE
#undef N2_24_CLEAR
#undef N2_24_ZERO
#undef N2_24_A
#undef N2_24_STEP
#undef N2_24_PAIR
#undef N2_24_FMA

N2_INLINE void n2_tail24_rows(int rows, int extra, int k, const float *a, int lda,
                              const float *b, int ldb, float *c, int ldc)
{
    if (rows == 4) {
        n2_full_4x24(k, a, lda, b, ldb, c, ldc);
    } else {
        n2_edge(rows, 4, 4, k, a, lda, b, ldb, c, ldc);
        n2_edge(rows, 2, 4, k, a, lda, b + 16, ldb, c + 16, ldc);
    }
    if (extra) {
        /* Load the final four in-bounds columns and store only the suffix. */
        const int last = 24 + extra - 4;
        n2_edge(rows, 1, extra, k, a, lda, b + last, ldb, c + last, ldc);
    }
}

static void n2_tail24_strip(int m, int extra, int k, const float *a, int lda,
                            const float *b, int ldb, float *c, int ldc)
{
    int i = 0;
    for (; i <= m - 4; i += 4) {
        n2_tail24_rows(4, extra, k, a + (size_t)i * lda, lda,
                       b, ldb, c + (size_t)i * ldc, ldc);
    }
    if (i == m) return;
    const float *ap = a + (size_t)i * lda;
    float *cp = c + (size_t)i * ldc;
    switch (m - i) {
        case 1: n2_tail24_rows(1, extra, k, ap, lda, b, ldb, cp, ldc); break;
        case 2: n2_tail24_rows(2, extra, k, ap, lda, b, ldb, cp, ldc); break;
        case 3: n2_tail24_rows(3, extra, k, ap, lda, b, ldb, cp, ldc); break;
    }
}
