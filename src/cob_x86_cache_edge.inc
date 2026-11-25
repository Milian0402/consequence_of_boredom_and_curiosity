/* Complete vectors use ordinary C loads and stores. The final partial
   vector uses fault-suppressing masked operations. No inactive row
   performs an FMA and no inactive output lane is read or written. */
#define COB_CACHE_EDGE_CORE(FUNC, TARGET, VEC, MASK, WIDTH, MR, LOAD, STORE, MLOAD, MSTORE, ZERO, SPLAT, FMA) \
TARGET __attribute__((always_inline)) static inline void FUNC( \
    int k, const float* a, const float* b, float* c, int ldc, int rows, \
    int cols, int resume, MASK mask, int two) \
{ \
    VEC lo[MR], hi[MR]; \
    const int low_full = cols >= WIDTH, high_full = cols == 2*WIDTH; \
    for (int r=0; r<rows; ++r) { \
        const float* dst=c+(size_t)r*ldc; \
        lo[r]=!resume ? ZERO() : low_full ? LOAD(dst) : MLOAD(dst,mask); \
        if (two) hi[r]=!resume ? ZERO() : high_full ? LOAD(dst+WIDTH) : MLOAD(dst+WIDTH,mask); \
    } \
    for (int p=0; p<k; ++p) { \
        /* The unmodified packers pad B to 2*WIDTH and A to MR. */ \
        const VEC left=LOAD(b); \
        VEC right; \
        if (two) right=LOAD(b+WIDTH); \
        for (int r=0; r<rows; ++r) { \
            const VEC av=SPLAT(a[r]); \
            lo[r]=FMA(av,left,lo[r]); \
            if (two) hi[r]=FMA(av,right,hi[r]); \
        } \
        a+=MR; b+=2*WIDTH; \
    } \
    for (int r=0; r<rows; ++r) { \
        float* dst=c+(size_t)r*ldc; \
        if (low_full) STORE(dst,lo[r]); else MSTORE(dst,mask,lo[r]); \
        if (two) { \
            if (high_full) STORE(dst+WIDTH,hi[r]); else MSTORE(dst+WIDTH,mask,hi[r]); \
        } \
    } \
}

#define COB_EDGE_AVX2 __attribute__((target("avx2,fma")))
#define COB_EDGE_AVX512 __attribute__((target("avx512f,fma")))
#define COB_EDGE_512_MLOAD(dst,mask) _mm512_maskz_loadu_ps(mask,dst)
#define COB_EDGE_512_MSTORE(dst,mask,value) _mm512_mask_storeu_ps(dst,mask,value)

COB_CACHE_EDGE_CORE(cob_cache_edge_512_rows, COB_EDGE_AVX512, __m512, __mmask16, 16, 12,
    _mm512_loadu_ps, _mm512_storeu_ps, COB_EDGE_512_MLOAD, COB_EDGE_512_MSTORE,
    _mm512_setzero_ps, _mm512_set1_ps, _mm512_fmadd_ps)
COB_CACHE_EDGE_CORE(cob_cache_edge_256_rows, COB_EDGE_AVX2, __m256, __m256i, 8, 6,
    _mm256_loadu_ps, _mm256_storeu_ps, _mm256_maskload_ps, _mm256_maskstore_ps,
    _mm256_setzero_ps, _mm256_set1_ps, _mm256_fmadd_ps)

#undef COB_EDGE_512_MSTORE
#undef COB_EDGE_512_MLOAD
#undef COB_CACHE_EDGE_CORE

/* Make the accumulator count and the number of B vectors compile-time
   constants. This avoids a runtime row loop and per-K accumulator spills. */
#define COB_EDGE_ROWS(R,F,W) case R: \
    if (cols > W) F(k,a,b,c,ldc,R,cols,resume,mask,1); \
    else F(k,a,b,c,ldc,R,cols,resume,mask,0); \
    return

COB_EDGE_AVX512 static void cob_cache_edge_512(int k,const float* a,const float* b,
    float* c,int ldc,int rows,int cols,int resume)
{
    const int active=cols>16 ? cols-16 : cols;
    const __mmask16 mask=(__mmask16)((1u<<active)-1);
#define COB_EDGE_R(r) COB_EDGE_ROWS(r,cob_cache_edge_512_rows,16)
    switch (rows) {
        COB_EDGE_R(1); COB_EDGE_R(2); COB_EDGE_R(3); COB_EDGE_R(4);
        COB_EDGE_R(5); COB_EDGE_R(6); COB_EDGE_R(7); COB_EDGE_R(8);
        COB_EDGE_R(9); COB_EDGE_R(10); COB_EDGE_R(11); COB_EDGE_R(12);
    }
#undef COB_EDGE_R
}

COB_EDGE_AVX2 static void cob_cache_edge_256(int k,const float* a,const float* b,
    float* c,int ldc,int rows,int cols,int resume)
{
    const int active=cols>8 ? cols-8 : cols;
    const __m256i lanes=_mm256_setr_epi32(0,1,2,3,4,5,6,7);
    const __m256i mask=_mm256_cmpgt_epi32(_mm256_set1_epi32(active),lanes);
#define COB_EDGE_R(r) COB_EDGE_ROWS(r,cob_cache_edge_256_rows,8)
    switch (rows) {
        COB_EDGE_R(1); COB_EDGE_R(2); COB_EDGE_R(3);
        COB_EDGE_R(4); COB_EDGE_R(5); COB_EDGE_R(6);
    }
#undef COB_EDGE_R
}
#undef COB_EDGE_ROWS
#undef COB_EDGE_AVX512
#undef COB_EDGE_AVX2

static int cob_x86_cache_edge(cob_blocked_kernel kernel,int k,
    const float* a,const float* b,float* c,int ldc,int rows,int cols,int resume)
{
    if (kernel.mr==12 && kernel.nr==32) {
        cob_cache_edge_512(k,a,b,c,ldc,rows,cols,resume);
        return 1;
    }
    if (kernel.mr==6 && kernel.nr==16) {
        cob_cache_edge_256(k,a,b,c,ldc,rows,cols,resume);
        return 1;
    }
    return 0;
}
