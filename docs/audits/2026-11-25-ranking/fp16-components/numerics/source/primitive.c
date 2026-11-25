#include <stddef.h>
#include <math.h>
#ifdef WIDEN
#include <arm_neon.h>
#endif

void fp16_block(const _Float16 *a,const _Float16 *b,float *dst,int m,int n,int k,int stride)
{
    for(int i=0;i<m;i++){
#ifdef WIDEN
        for(int j=0;j<n;j+=4){
            float32x4_t s=vdupq_n_f32(0);
            for(int p=0;p<k;p++){
                float16x8_t x=vreinterpretq_f16_u16(vdupq_n_u16(((const __attribute__((__may_alias__)) unsigned short*)a)[(size_t)i*k+p]));
                float16x4_t lane=vreinterpret_f16_u16(vld1_u16((const uint16_t*)(b+(size_t)p*stride+j)));
                s=vfmlalq_low_f16(s,x,vcombine_f16(lane,vdup_n_f16(0)));
            }
            vst1q_f32(dst+(size_t)i*stride+j,s);
        }
#else
        for(int j=0;j<n;j++){
            float s=0;
            for(int p=0;p<k;p++) s=fmaf((float)a[(size_t)i*k+p],(float)b[(size_t)p*stride+j],s);
            dst[(size_t)i*stride+j]=s;
        }
#endif
    }
}
