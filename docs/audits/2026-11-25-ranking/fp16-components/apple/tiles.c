#include <arm_sme.h>
#include <arm_sve.h>
#include <stdint.h>
#include <stdatomic.h>
#include <stddef.h>
#include <sys/sysctl.h>

static int apple_feature(const char *key)
{
    int result=0; size_t size=sizeof(result);
    return sysctlbyname(key,&result,&size,NULL,0)==0 && size==sizeof(result) && result;
}

int cob_half_svl(void)
{
    static atomic_int known;
    int enabled=atomic_load_explicit(&known,memory_order_relaxed);
    if (!enabled) {
        enabled=apple_feature("hw.optional.arm.FEAT_SME2") &&
                apple_feature("hw.optional.arm.SME_F16F32") ? 1 : -1;
        atomic_store_explicit(&known,enabled,memory_order_relaxed);
    }
    /* The one-shot packed formats and local partial tile use M5's 512-bit SVL. */
    return enabled>0 && svcntsw()==16 ? 16 : 0;
}

static inline __attribute__((always_inline)) void
half_tile_dot(const _Float16 *a,const _Float16 *b,int pairs)
    __arm_streaming __arm_inout("za")
{
    const svbool_t all=svptrue_b16();
    svzero_za();
#pragma clang loop unroll_count(4)
    for (int p=0;p<pairs;++p) {
        const float16_t *aa=(const float16_t*)(a+(size_t)p*64);
        const float16_t *bb=(const float16_t*)(b+(size_t)p*64);
        const svfloat16_t a0=svld1_f16(all,aa), a1=svld1_f16(all,aa+32);
        const svfloat16_t b0=svld1_f16(all,bb), b1=svld1_f16(all,bb+32);
        svmopa_za32_f16_m(0,all,all,a0,b0);
        svmopa_za32_f16_m(1,all,all,a0,b1);
        svmopa_za32_f16_m(2,all,all,a1,b0);
        svmopa_za32_f16_m(3,all,all,a1,b1);
    }
}

#define STORE_QUADRANT(T,ROW,COL) do {                                            \
    if (height > (ROW) && width > (COL)) {                                         \
        const svbool_t pg=svwhilelt_b32_s32(0,width-(COL));                         \
        const int stop=height-(ROW)<16 ? height-(ROW) : 16;                        \
        for (int r=0;r<stop;++r) {                                                 \
            float *dest=out+(size_t)(r+(ROW))*ld+(COL);                            \
            if (operation==0) svst1_hor_za32(T,r,pg,dest);                        \
            else {                                                                  \
                svfloat32_t za=svread_hor_za32_f32_m(svdup_f32(0),pg,T,r);         \
                const float *low=partial+(r+(ROW))*32+(COL);                      \
                svfloat32_t v=svld1_f32(pg,low);                                  \
                if (operation==2) v=svmul_n_f32_x(pg,v,0x1p-11f);                \
                svst1_f32(pg,dest,svadd_f32_x(pg,za,v));                          \
            }                                                                       \
        }                                                                           \
    }                                                                               \
} while (0)

static inline __attribute__((always_inline)) void
half_tile_store(float *out,int ld,int height,int width,int operation,const float *partial)
    __arm_streaming __arm_in("za")
{
    STORE_QUADRANT(0,0,0); STORE_QUADRANT(1,0,16);
    STORE_QUADRANT(2,16,0); STORE_QUADRANT(3,16,16);
}
#undef STORE_QUADRANT

/* Packed dimension 32, two depth values adjacent for each output lane. */
__arm_new("za") __arm_locally_streaming
void apple_half_tiles(int mode,int m,int n,int k,const _Float16 *ah,const _Float16 *al,
                      const _Float16 *bh,const _Float16 *bl,float *c,int ldc)
{
    const int pairs=k/2+(k&1);
    const size_t stride=(size_t)pairs*64;
    _Alignas(64) float cross[32*32];
    for (int ib=0;ib<m;) {
        const int height=m-ib<32 ? m-ib : 32;
        const size_t ai=(size_t)(ib/32)*stride;
        for (int jb=0;jb<n;) {
            const int width=n-jb<32 ? n-jb : 32;
            const size_t bi=(size_t)(jb/32)*stride;
            float *out=c+(size_t)ib*ldc+jb;
            if (mode) {
                half_tile_dot(ah+ai,bl+bi,pairs);
                half_tile_store(cross,32,height,width,0,NULL);
                half_tile_dot(al+ai,bh+bi,pairs);
                half_tile_store(cross,32,height,width,1,cross);
            }
            half_tile_dot(ah+ai,bh+bi,pairs);
            half_tile_store(out,ldc,height,width,mode ? 2 : 0,cross);
            jb += width;
        }
        ib += height;
    }
}
