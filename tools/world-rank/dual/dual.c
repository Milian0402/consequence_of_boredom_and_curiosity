/* One-shot row-major FP32. Compile for the host to select its SIMD width.
   A fixed B slab is shared by all row microtiles. A keeps its caller layout. */
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#if defined(__AVX512F__) && defined(__FMA__)
#include <immintrin.h>
typedef __m512 dual_vec;
#define D_WIDTH 16
#define D_ROWS 8
#define D_VECS 3
#define D_ZERO _mm512_setzero_ps()
#define D_LOAD(p) _mm512_loadu_ps(p)
#define D_STORE(p,x) _mm512_storeu_ps(p,x)
#define D_SPLAT(x) _mm512_set1_ps(x)
#define D_FMA(s,a,b) _mm512_fmadd_ps(a,b,s)
#elif defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
typedef __m256 dual_vec;
#define D_WIDTH 8
#define D_ROWS 4
#define D_VECS 3
#define D_ZERO _mm256_setzero_ps()
#define D_LOAD(p) _mm256_loadu_ps(p)
#define D_STORE(p,x) _mm256_storeu_ps(p,x)
#define D_SPLAT(x) _mm256_set1_ps(x)
#define D_FMA(s,a,b) _mm256_fmadd_ps(a,b,s)
#elif defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
typedef float32x4_t dual_vec;
#define D_WIDTH 4
#define D_ROWS 6
#define D_VECS 4
#define D_ZERO vdupq_n_f32(0)
#define D_LOAD(p) vld1q_f32(p)
#define D_STORE(p,x) vst1q_f32(p,x)
#define D_SPLAT(x) vdupq_n_f32(x)
#define D_FMA(s,a,b) vfmaq_f32(s,a,b)
#endif

static void remaining(int m,int n,int k,const float* a,int lda,
    const float* b,int ldb,float* c,int ldc)
{
    for(int i=0;i<m;++i) {
        float* dst=c+(size_t)i*ldc;
        memset(dst,0,(size_t)n*sizeof(float));
        for(int p=0;p<k;++p) {
            const float av=a[(size_t)i*lda+p];
            const float* bp=b+(size_t)p*ldb;
            for(int j=0;j<n;++j) dst[j]=fmaf(av,bp[j],dst[j]);
        }
    }
}

#ifdef D_WIDTH
enum { D_COLS=D_WIDTH*D_VECS, D_SLAB_TILES=8 };

static inline void tile(int k,const float* a,int lda,const float* packed,float* c,int ldc)
{
    dual_vec accum[D_ROWS][D_VECS];
    for(int r=0;r<D_ROWS;++r)
        for(int v=0;v<D_VECS;++v) accum[r][v]=D_ZERO;
    for(int p=0;p<k;++p) {
        dual_vec bv[D_VECS];
        for(int v=0;v<D_VECS;++v) bv[v]=D_LOAD(packed+v*D_WIDTH);
        for(int r=0;r<D_ROWS;++r) {
            const dual_vec av=D_SPLAT(a[(size_t)r*lda+p]);
            for(int v=0;v<D_VECS;++v)
                accum[r][v]=D_FMA(accum[r][v],av,bv[v]);
        }
        packed+=D_COLS;
    }
    for(int r=0;r<D_ROWS;++r)
        for(int v=0;v<D_VECS;++v) D_STORE(c+(size_t)r*ldc+v*D_WIDTH,accum[r][v]);
}
#endif

int rank_vendor_init(void) { return 0; }

void rank_vendor_sgemm(int m,int n,int k,const float* a,int lda,
    const float* b,int ldb,float* c,int ldc)
{
    if(m<=0||n<=0||k<0) return;
#ifdef D_WIDTH
    const int full_rows=m-m%D_ROWS, full_cols=n-n%D_COLS;
    if(k>0 && full_rows && full_cols) {
        const int cap=full_cols/D_COLS<D_SLAB_TILES?full_cols/D_COLS:D_SLAB_TILES;
        float* packed=malloc((size_t)cap*k*D_COLS*sizeof(float));
        if(packed) {
            for(int j=0;j<full_cols;) {
                int tiles=(full_cols-j)/D_COLS;
                if(tiles>cap) tiles=cap;
                /* Read contiguous spans of every original B row, and arrange
                   each microtile contiguously along K for repeated reads. */
                for(int p=0;p<k;++p)
                    for(int t=0;t<tiles;++t)
                        memcpy(packed+((size_t)t*k+p)*D_COLS,
                            b+(size_t)p*ldb+j+t*D_COLS,D_COLS*sizeof(float));
                for(int i=0;i<full_rows;i+=D_ROWS)
                    for(int t=0;t<tiles;++t)
                        tile(k,a+(size_t)i*lda,lda,packed+(size_t)t*k*D_COLS,
                            c+(size_t)i*ldc+j+t*D_COLS,ldc);
                j+=tiles*D_COLS;
            }
            free(packed);
            if(full_cols<n)
                remaining(full_rows,n-full_cols,k,a,lda,b+full_cols,ldb,c+full_cols,ldc);
            if(full_rows<m)
                remaining(m-full_rows,n,k,a+(size_t)full_rows*lda,lda,b,ldb,
                    c+(size_t)full_rows*ldc,ldc);
            return;
        }
    }
#endif
    remaining(m,n,k,a,lda,b,ldb,c,ldc);
}
