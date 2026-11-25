/* Direct FP32 GEMM for small Arm matrices. Define N2_NO_PLUGIN to use the
   guarded n2_try_sgemm entry as a route in a general-purpose multiplier. */
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>

/* 24 output vectors, 6 A vectors, 2 B vectors. Keep A in the lane-encoding
   registers v0..v5 and load each A row once per four reduction steps. */
#define N2_FMA(c,b,a,l) "fmla v" #c ".4s, v" #b ".4s, v" #a ".s[" #l "]\n"
#define N2_PAIR(l) \
    "ldp q6,q7,[%[b]]\n" \
    N2_FMA(8,6,0,l)  N2_FMA(9,7,0,l)   N2_FMA(12,6,1,l) N2_FMA(13,7,1,l) \
    N2_FMA(16,6,2,l) N2_FMA(17,7,2,l)  N2_FMA(20,6,3,l) N2_FMA(21,7,3,l) \
    N2_FMA(24,6,4,l) N2_FMA(25,7,4,l)  N2_FMA(28,6,5,l) N2_FMA(29,7,5,l) \
    "ldp q6,q7,[%[b],#32]\n" \
    N2_FMA(10,6,0,l) N2_FMA(11,7,0,l)  N2_FMA(14,6,1,l) N2_FMA(15,7,1,l) \
    N2_FMA(18,6,2,l) N2_FMA(19,7,2,l)  N2_FMA(22,6,3,l) N2_FMA(23,7,3,l) \
    N2_FMA(26,6,4,l) N2_FMA(27,7,4,l)  N2_FMA(30,6,5,l) N2_FMA(31,7,5,l) \
    "add %[b],%[b],%[bstep]\n"
#define N2_CLR(v) "movi v" #v ".4s,#0\n"
#define N2_LOAD_A(r,kind,inc) "ldr " #kind #r ",[%[a" #r "]],#" #inc "\n"
#define N2_EACH_LOAD(kind,inc) N2_LOAD_A(0,kind,inc) N2_LOAD_A(1,kind,inc) \
    N2_LOAD_A(2,kind,inc) N2_LOAD_A(3,kind,inc) N2_LOAD_A(4,kind,inc) N2_LOAD_A(5,kind,inc)
#define N2_STORE(a,b,c,d) "stp q" #a ",q" #b ",[%[cp]]\n" \
    "stp q" #c ",q" #d ",[%[cp],#32]\n" "add %[cp],%[cp],%[cstep]\n"

static inline __attribute__((always_inline)) void n2_full(
    int k, const float *a, int lda, const float *b, int ldb, float *c, int ldc)
{
    const float *a0 = a, *a1 = a + lda, *a2 = a + (size_t)2*lda;
    const float *a3 = a + (size_t)3*lda, *a4 = a + (size_t)4*lda, *a5 = a + (size_t)5*lda;
    int chunks = k/4, tail = k%4;
    uintptr_t bstep = (size_t)ldb * sizeof(float), cstep = (size_t)ldc * sizeof(float);
    __asm__ volatile (
        N2_CLR(8) N2_CLR(9) N2_CLR(10) N2_CLR(11)  N2_CLR(12) N2_CLR(13) N2_CLR(14) N2_CLR(15)
        N2_CLR(16) N2_CLR(17) N2_CLR(18) N2_CLR(19) N2_CLR(20) N2_CLR(21) N2_CLR(22) N2_CLR(23)
        N2_CLR(24) N2_CLR(25) N2_CLR(26) N2_CLR(27) N2_CLR(28) N2_CLR(29) N2_CLR(30) N2_CLR(31)
        "1:\n" N2_EACH_LOAD(q,16)
        N2_PAIR(0) N2_PAIR(1) N2_PAIR(2) N2_PAIR(3)
        "subs %w[chunks],%w[chunks],#1\n" "b.ne 1b\n"
        "cbz %w[tail],3f\n"
        "2:\n" N2_EACH_LOAD(s,4) N2_PAIR(0)
        "subs %w[tail],%w[tail],#1\n" "b.ne 2b\n"
        "3:\n"
        N2_STORE(8,9,10,11) N2_STORE(12,13,14,15) N2_STORE(16,17,18,19)
        N2_STORE(20,21,22,23) N2_STORE(24,25,26,27) N2_STORE(28,29,30,31)
        : [a0] "+&r"(a0), [a1] "+&r"(a1), [a2] "+&r"(a2),
          [a3] "+&r"(a3), [a4] "+&r"(a4), [a5] "+&r"(a5),
          [b] "+&r"(b), [cp] "+&r"(c), [chunks] "+&r"(chunks), [tail] "+&r"(tail)
        : [bstep] "r"(bstep), [cstep] "r"(cstep)
        : "cc","memory","v0","v1","v2","v3","v4","v5","v6","v7","v8","v9","v10","v11",
          "v12","v13","v14","v15","v16","v17","v18","v19","v20","v21","v22","v23",
          "v24","v25","v26","v27","v28","v29","v30","v31");
}
#undef N2_STORE
#undef N2_EACH_LOAD
#undef N2_LOAD_A
#undef N2_CLR
#undef N2_PAIR
#undef N2_FMA

#define N2_INLINE static inline __attribute__((always_inline))
#define N2_ROWS(F) F(0) F(1) F(2) F(3) F(4) F(5)

/* Each complete group loads ordinary four-column vectors. For the last
   one-to-three columns, read the final four in-bounds columns and store only
   the valid suffix. The K loop stays identical to a full vector. */
N2_INLINE void n2_partial_store(float *p, float32x4_t value, int valid)
{
    if (valid == 4) { vst1q_f32(p,value); return; }
    if (valid >= 3) vst1q_lane_f32(p+1,value,1);
    if (valid >= 2) vst1q_lane_f32(p+2,value,2);
    vst1q_lane_f32(p+3,value,3);
}
N2_INLINE void n2_edge(int rows, int vecs, int valid, int k, const float *a, int lda,
                       const float *b, int ldb, float *c, int ldc)
{
#define N2_INIT(r) float32x4_t c##r##a=vdupq_n_f32(0), c##r##b=vdupq_n_f32(0), \
                               c##r##c=vdupq_n_f32(0), c##r##d=vdupq_n_f32(0);
    N2_ROWS(N2_INIT)
#undef N2_INIT
    for (int p=0; p<k; ++p) {
        const float *bp = b + (size_t)p*ldb;
        const float32x4_t z = vdupq_n_f32(0);
        float32x4_t b0=vld1q_f32(bp);
        float32x4_t b1=vecs>=2 ? vld1q_f32(bp+4) : z;
        float32x4_t b2=vecs>=3 ? vld1q_f32(bp+8) : z;
        float32x4_t b3=vecs>=4 ? vld1q_f32(bp+12) : z;
#define N2_STEP(r) if (rows > r) { \
        float av=a[(size_t)(r)*lda+p]; \
        c##r##a=vfmaq_n_f32(c##r##a,b0,av); \
        if (vecs>=2) c##r##b=vfmaq_n_f32(c##r##b,b1,av); \
        if (vecs>=3) c##r##c=vfmaq_n_f32(c##r##c,b2,av); \
        if (vecs>=4) c##r##d=vfmaq_n_f32(c##r##d,b3,av); \
    }
        N2_ROWS(N2_STEP)
#undef N2_STEP
    }
#define N2_WRITE(r) if (rows>r) { \
    float *cp=c+(size_t)(r)*ldc; \
    n2_partial_store(cp,c##r##a,vecs==1 ? valid : 4); \
    if (vecs>=2) n2_partial_store(cp+4,c##r##b,vecs==2 ? valid : 4); \
    if (vecs>=3) n2_partial_store(cp+8,c##r##c,vecs==3 ? valid : 4); \
    if (vecs>=4) n2_partial_store(cp+12,c##r##d,valid); \
}
    N2_ROWS(N2_WRITE)
#undef N2_WRITE
}

N2_INLINE void n2_strip(int rows, int n, int k, const float *a, int lda,
                        const float *b, int ldb, float *c, int ldc)
{
    int j=0;
    for (; j<=n-16; j+=16) {
        if (rows == 6) n2_full(k,a,lda,b+j,ldb,c+j,ldc);
        else n2_edge(rows,4,4,k,a,lda,b+j,ldb,c+j,ldc);
    }
    /* Remaining complete four-float vectors can use ordinary vector loads.
       A separate narrow group handles one to three leftover columns. */
    const int groups=(n-j)/4;
#define N2_EDGE(g) case g: n2_edge(rows,g,4,k,a,lda,b+j,ldb,c+j,ldc); break;
    switch (groups) { N2_EDGE(1) N2_EDGE(2) N2_EDGE(3) }
#undef N2_EDGE
    j+=groups*4;
    if (j<n) n2_edge(rows,1,n-j,k,a,lda,b+n-4,ldb,c+n-4,ldc);
}

static int n2_overlap(const float *x, size_t nx, const float *y, size_t ny)
{
    uintptr_t px=(uintptr_t)x, py=(uintptr_t)y;
    return px <= py ? py-px < nx*sizeof(float) : px-py < ny*sizeof(float);
}

int n2_try_sgemm(int m,int n,int k,const float *a,int lda,const float *b,int ldb,float *c,int ldc)
{
    if (m<8 || m>128 || n<8 || n>128 || k<8 || k>128 || !a || !b || !c ||
        lda<k || ldb<n || ldc<n) return 0;
    const size_t count=(size_t)(m-1)*ldc+n;
    if (n2_overlap(c,count,a,(size_t)(m-1)*lda+k) || n2_overlap(c,count,b,(size_t)(k-1)*ldb+n)) return 0;
    int i=0;
    for (; i<=m-6; i+=6) n2_strip(6,n,k,a+(size_t)i*lda,lda,b,ldb,c+(size_t)i*ldc,ldc);
    if (i==m) return 1;
    a+=(size_t)i*lda;
    c+=(size_t)i*ldc;
#define N2_REST(r) case r: n2_strip(r,n,k,a,lda,b,ldb,c,ldc); break;
    switch (m-i) { N2_REST(1) N2_REST(2) N2_REST(3) N2_REST(4) N2_REST(5) }
#undef N2_REST
    return 1;
}
#else
int n2_try_sgemm(int m,int n,int k,const float *a,int lda,const float *b,int ldb,float *c,int ldc)
{
    (void)m; (void)n; (void)k; (void)a; (void)lda; (void)b; (void)ldb; (void)c; (void)ldc;
    return 0;
}
#endif

#ifndef N2_NO_PLUGIN
int rank_vendor_init(void) {
#if defined(__aarch64__) && defined(__ARM_NEON)
    return 0;
#else
    return -1;
#endif
}
void rank_vendor_sgemm(int m,int n,int k,const float *a,int lda,const float *b,int ldb,float *c,int ldc)
{
    if (n2_try_sgemm(m,n,k,a,lda,b,ldb,c,ldc)) return;
    if (m<1 || n<1 || k<1 || !a || !b || !c || lda<k || ldb<n || ldc<n) return;
    for (int i=0; i<m; ++i)
        for (int j=0; j<n; ++j) {
            float sum=0;
            for (int p=0;p<k;++p) sum+=a[(size_t)i*lda+p]*b[(size_t)p*ldb+j];
            c[(size_t)i*ldc+j]=sum;
        }
}
#endif
