/* The top-level schedule exposes the original destination values needed by
   M1, M5, M6 and M7. Each final-panel tile recombines only its own elements. */
enum cob_final_operation { COB_FINAL_M1, COB_FINAL_M5, COB_FINAL_M6, COB_FINAL_M7 };
typedef struct {
    float* c11;
    float* c12;
    float* c21;
    float* c22;
    int ldc;
    enum cob_final_operation operation;
} cob_final_output;

typedef void (*cob_final_tile)(int, const float*, const float*, const float*,
    int, int, const cob_final_output*, size_t);

#if defined(COB_USE_X86_SIMD)
#define COB_FINAL_TARGET __attribute__((target("avx2,fma")))
#define COB_FINAL_VEC __m256
#define COB_FINAL_LANES 8
#define COB_FINAL_LOAD _mm256_loadu_ps
#define COB_FINAL_STORE _mm256_storeu_ps
#define COB_FINAL_FMA _mm256_fmadd_ps
#define COB_FINAL_SPLAT _mm256_set1_ps
#define COB_FINAL_ZERO _mm256_setzero_ps
#if defined(COB_ENABLE_WIDE_X86) && COB_ENABLE_WIDE_X86
#define COB_FINAL_MR 4
#define COB_FINAL_NR 24
#else
#define COB_FINAL_MR 6
#define COB_FINAL_NR 16
#endif
#define COB_FINAL_NAME cob_final_avx2
#include "cob_final_product_tile.inc"
#undef COB_FINAL_TARGET
#undef COB_FINAL_VEC
#undef COB_FINAL_LANES
#undef COB_FINAL_LOAD
#undef COB_FINAL_STORE
#undef COB_FINAL_FMA
#undef COB_FINAL_SPLAT
#undef COB_FINAL_ZERO

#define COB_FINAL_TARGET __attribute__((target("avx512f,fma")))
#define COB_FINAL_VEC __m512
#define COB_FINAL_LANES 16
#define COB_FINAL_LOAD _mm512_loadu_ps
#define COB_FINAL_STORE _mm512_storeu_ps
#define COB_FINAL_FMA _mm512_fmadd_ps
#define COB_FINAL_SPLAT _mm512_set1_ps
#define COB_FINAL_ZERO _mm512_setzero_ps
#if defined(COB_ENABLE_WIDE_X86) && COB_ENABLE_WIDE_X86
#define COB_FINAL_MR 8
#define COB_FINAL_NR 48
#else
#define COB_FINAL_MR 12
#define COB_FINAL_NR 32
#endif
#define COB_FINAL_NAME cob_final_avx512
#else
#define COB_FINAL_TARGET
#define COB_FINAL_VEC float32x4_t
#define COB_FINAL_LANES 4
#define COB_FINAL_LOAD vld1q_f32
#define COB_FINAL_STORE vst1q_f32
#define COB_FINAL_FMA(a,b,c) vfmaq_f32(c,b,a)
#define COB_FINAL_SPLAT vdupq_n_f32
#define COB_FINAL_ZERO() vdupq_n_f32(0.0f)
#define COB_FINAL_MR 8
#define COB_FINAL_NR 12
#define COB_FINAL_NAME cob_final_neon
#endif
#include "cob_final_product_tile.inc"
#undef COB_FINAL_TARGET
#undef COB_FINAL_VEC
#undef COB_FINAL_LANES
#undef COB_FINAL_LOAD
#undef COB_FINAL_STORE
#undef COB_FINAL_FMA
#undef COB_FINAL_SPLAT
#undef COB_FINAL_ZERO

/* Fringe tiles keep the exact stock kernel. Valid partial sums are copied
   to a padded local tile so no source load or destination store crosses a row. */
static void cob_final_edge(const cob_blocked_kernel* kernel, int k,
    const float* a, const float* b, const float* prior, int ldp, int resume,
    const cob_final_output* q, size_t offset, int rows, int cols)
{
    _Alignas(64) float edge[12*48] = {0};
    if (resume) {
        for (int r=0; r<rows; ++r)
            memcpy(edge+(size_t)r*kernel->nr,prior+(size_t)r*ldp,(size_t)cols*sizeof(float));
    }
    kernel->tile(k,a,b,edge,kernel->nr,resume);
    for (int r=0; r<rows; ++r) {
        for (int j=0; j<cols; ++j) {
            const size_t off=offset+(size_t)r*q->ldc+j;
            const float product=edge[(size_t)r*kernel->nr+j];
            if (q->operation==COB_FINAL_M1) {
                const float m4=q->c11[off], m3=q->c12[off], m2=q->c21[off];
                q->c11[off]=product+m4;
                q->c21[off]=m2+m4;
                q->c22[off]=(product-m2)+m3;
            } else if (q->operation==COB_FINAL_M5) {
                q->c11[off]-=product;
                q->c12[off]+=product;
            } else {
                float* dst=q->operation==COB_FINAL_M6 ? q->c22 : q->c11;
                dst[off]+=product;
            }
        }
    }
}

static void cob_final_panel(const cob_blocked_kernel* kernel, int m, int n, int k,
    const float* pack_a,const float* pack_b,const float* prior,int ldp,
    int resume,const cob_final_output* q,int base_row)
{
#if defined(COB_USE_X86_SIMD)
    const cob_final_tile full = kernel->nr >= 32 ? cob_final_avx512 : cob_final_avx2;
#else
    const cob_final_tile full = cob_final_neon;
#endif
    for (int j=0; j<n; j+=kernel->nr) {
        const int cols=cob_min_i32(kernel->nr,n-j);
        const float* b=pack_b+(size_t)j*k;
        for (int i=0; i<m; i+=kernel->mr) {
            const int rows=cob_min_i32(kernel->mr,m-i);
            const float* a=pack_a+(size_t)i*k;
            const float* p=prior+(size_t)i*ldp+j;
            const size_t offset=(size_t)(base_row+i)*q->ldc+j;
            if (rows==kernel->mr && cols==kernel->nr)
                full(k,a,b,p,ldp,resume,q,offset);
            else
                cob_final_edge(kernel,k,a,b,p,ldp,resume,q,offset,rows,cols);
        }
    }
}
