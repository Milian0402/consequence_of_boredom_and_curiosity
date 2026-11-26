#define _POSIX_C_SOURCE 200112L
#define ACCELERATE_NEW_LAPACK 1
#include <Accelerate/Accelerate.h>
#include <arm_neon.h>
#include <arm_sme.h>
#include <arm_sve.h>
#include <stdint.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <sys/sysctl.h>

static int feature(const char *name) {
    int value=0; size_t bytes=sizeof(value);
    return !sysctlbyname(name,&value,&bytes,NULL,0) && bytes==sizeof(value) && value;
}
int apple_int8_lanes(void) {
    static atomic_int cache;
    int avail=atomic_load_explicit(&cache,memory_order_relaxed);
    if (!avail) {
        avail=feature("hw.optional.arm.FEAT_SME2")&&feature("hw.optional.arm.SME_I8I32") ? 1 : -1;
        atomic_store_explicit(&cache,avail,memory_order_relaxed);
    }
    return avail>0 && svcntsw()==16 ? 16 : 0;
}
__attribute__((constructor)) static void serial_blas(void) {
    if (BLASSetThreading(BLAS_THREADING_SINGLE_THREADED) ||
        BLASGetThreading()!=BLAS_THREADING_SINGLE_THREADED) abort();
}
int exact_blas_serial(void) {return BLASGetThreading()==BLAS_THREADING_SINGLE_THREADED;}
void exact_baseline(int m,int n,int k,const float*a,int la,const float*b,int lb,float*c,int lc) {
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,n,k,1.0f,a,la,b,lb,0.0f,c,lc);
}
static int scalar(float x,int8_t *dest) {
    /* Integer predicates use raw binary32 bits even with flush-to-zero enabled. */
    uint32_t bits;memcpy(&bits,&x,sizeof(bits));
    uint32_t mag=bits&0x7fffffffu;
    if (!mag){*dest=0;return 1;}
    if (mag<0x3f800000u || mag>0x427c0000u) return 0;
    int exp=(int)(mag>>23)-127;
    if(mag & ((UINT32_C(1)<<(23-exp))-1)) return 0;
    int value=(int)((0x800000u|(mag&0x7fffffu))>>(23-exp));
    *dest=(int8_t)(bits>>31 ? -value : value);
    return 1;
}
static inline int32x4_t four(const float*x,uint32x4_t *good) {
    float32x4_t in=vld1q_f32(x);
    uint32x4_t mag=vandq_u32(vreinterpretq_u32_f32(in),vdupq_n_u32(0x7fffffff));
    int32x4_t q=vcvtq_s32_f32(in);
    /* Compare raw magnitude after round-trip, preserving both zero signs. */
    uint32x4_t back=vandq_u32(vreinterpretq_u32_f32(vcvtq_f32_s32(q)),vdupq_n_u32(0x7fffffff));
    *good=vandq_u32(*good,vandq_u32(vceqq_u32(mag,back),vcleq_u32(mag,vdupq_n_u32(0x427c0000u))));
    return q;
}
static inline int8x16_t sixteen(const float*x,uint32x4_t *good) {
    int32x4_t q0=four(x,good),q1=four(x+4,good),q2=four(x+8,good),q3=four(x+12,good);
    int16x8_t lo=vcombine_s16(vmovn_s32(q0),vmovn_s32(q1));
    int16x8_t hi=vcombine_s16(vmovn_s32(q2),vmovn_s32(q3));
    return vcombine_s8(vmovn_s16(lo),vmovn_s16(hi));
}
static int pack(int m,int n,int k,const float*a,int la,const float*b,int lb,size_t kp,int8_t*ap,int8_t*bp) {
    for (int i=0;i<m;++i) {
        const float*ar=a+(size_t)i*la;
        int8_t*base=ap+(size_t)(i/32)*32*kp+(i%32)*4;
        uint32x4_t good=vdupq_n_u32(UINT32_MAX);int p=0;
        for(;p+3<k;p+=4) {
            const int16x4_t q=vmovn_s32(four(ar+p,&good));
            const int8x8_t bytes=vmovn_s16(vcombine_s16(q,q));
            const int32_t packed=vget_lane_s32(vreinterpret_s32_s8(bytes),0);
            memcpy(base+(size_t)(p/4)*128,&packed,4);
        }
        if(vminvq_u32(good)!=UINT32_MAX)return 0;
        for(;p<k;++p) if(!scalar(ar[p],base+(size_t)(p/4)*128+(p%4)))return 0;
    }
    for(int j=0;j<n;) {
        int width=n-j<32 ? n-j : 32;
        int8_t*base=bp+(size_t)(j/32)*32*kp;
        uint32x4_t good=vdupq_n_u32(UINT32_MAX);int p=0;
        for(;p+3<k;p+=4) {
            int8_t *dst=base+(size_t)(p/4)*128;
            const float *r=b+(size_t)p*lb+j;
            int q=0;
            for(;q+15<width;q+=16) {
                int8x16x4_t v;
                for(int t=0;t<4;++t)v.val[t]=sixteen(r+(size_t)t*lb+q,&good);
                vst4q_s8(dst+4*q,v);
            }
            for(;q<width;++q) for(int t=0;t<4;++t)
                if(!scalar(r[(size_t)t*lb+q],dst+4*q+t))return 0;
        }
        if(vminvq_u32(good)!=UINT32_MAX)return 0;
        for(;p<k;++p) for(int q=0;q<width;++q) {
            if(!scalar(b[(size_t)p*lb+j+q],base+(size_t)(p/4)*128+4*q+p%4))return 0;
        }
        j+=width;
    }
    return 1;
}
#define DRAIN(T,R,J) do { if(h>(R) && w>(J)) { \
    svbool_t pg=svwhilelt_b32_s32(0,w-(J));int re=h-(R)<16?h-(R):16; \
    for(int i=0;i<re;i++){svint32_t val=svread_hor_za32_s32_m(svdup_s32(0),pg,T,i); \
        svst1_f32(pg,out+(size_t)(i+(R))*lc+(J),svcvt_f32_s32_x(pg,val)); } }} while (0)
__arm_new("za") __arm_locally_streaming
static void tiles(int m,int n,int groups,const int8_t*a,const int8_t*b,float*c,int lc) {
    const size_t stride=(size_t)groups*128;const svbool_t all=svptrue_b8();
    for(int i=0;i<m;) {int h=m-i<32?m-i:32;
        for(int j=0;j<n;) {int w=n-j<32?n-j:32;
            const int8_t *aa=a+(size_t)(i/32)*stride,*bb=b+(size_t)(j/32)*stride;
            svzero_za();
#pragma clang loop unroll_count(4)
            for(int t=0;t<groups;++t) {
                svint8_t a0=svld1_s8(all,aa+(size_t)t*128),a1=svld1_s8(all,aa+(size_t)t*128+64);
                svint8_t b0=svld1_s8(all,bb+(size_t)t*128),b1=svld1_s8(all,bb+(size_t)t*128+64);
                svmopa_za32_s8_m(0,all,all,a0,b0);svmopa_za32_s8_m(1,all,all,a0,b1);
                svmopa_za32_s8_m(2,all,all,a1,b0);svmopa_za32_s8_m(3,all,all,a1,b1);
            }
            float*out=c+(size_t)i*lc+j;DRAIN(0,0,0);DRAIN(1,0,16);DRAIN(2,16,0);DRAIN(3,16,16);
            j+=w;
        }
        i+=h;
    }
}
static int span(const float *x,int rows,int cols,int ld,uintptr_t *end){
    size_t count=(size_t)(rows-1)*(size_t)ld+(size_t)cols;
    if(count>SIZE_MAX/sizeof(float))return 0;
    size_t last=count*sizeof(float)-1;
    if(last>UINTPTR_MAX-(uintptr_t)x)return 0;
    *end=(uintptr_t)x+last;return 1;
}
int apple_exact(int m,int n,int k,const float*a,int la,const float*b,int lb,float*c,int lc){
    if(m<0||n<0||k<0)return -1;
    if(!m||!n)return 0;
    uintptr_t ae,be,ce;
    if(!c||lc<n||!span(c,m,n,lc,&ce))return -1;
    if(!k){for(int i=0;i<m;i++)memset(c+(size_t)i*lc,0,(size_t)n*sizeof(float));return 0;}
    if(!a||!b||la<k||lb<n)return -1;
    if(!span(a,m,k,la,&ae)||!span(b,k,n,lb,&be) ||
       ((uintptr_t)c<=ae && (uintptr_t)a<=ce)||((uintptr_t)c<=be && (uintptr_t)b<=ce))return -1;
    int8_t check;
    if(k<=4096 && apple_int8_lanes() && scalar(a[0],&check) && scalar(b[0],&check)){
        size_t kp=((size_t)k+3)&~(size_t)3,mp=((size_t)m+31)&~(size_t)31,np=((size_t)n+31)&~(size_t)31;
        if(mp+np<=SIZE_MAX/kp) {
            int8_t*tmp=calloc((mp+np)*kp,1);
            if(tmp) {
                int8_t*bp=tmp+mp*kp;
                int ok=pack(m,n,k,a,la,b,lb,kp,tmp,bp);
                if(ok)tiles(m,n,(int)(kp/4),tmp,bp,c,lc);
                free(tmp);if(ok)return 1;
            }
        }
    }
    exact_baseline(m,n,k,a,la,b,lb,c,lc);return 0;
}
