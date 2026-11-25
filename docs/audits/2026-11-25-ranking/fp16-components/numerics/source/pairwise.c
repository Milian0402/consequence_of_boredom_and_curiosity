#include <arm_neon.h>
#include <stdlib.h>
#include <stdint.h>

void fp16_block(const _Float16 *a,const _Float16 *b,float *dst,int m,int n,int k,int stride)
{
    size_t width=1;
    while(width<(size_t)k)width*=2;
    float32x4_t *v=malloc(width*sizeof *v);
    if(!v)abort();
    for(int i=0;i<m;i++)for(int j=0;j<n;j+=4){
        for(int p=0;p<k;p++){
            float16x8_t x=vdupq_n_f16((float16_t)a[(size_t)i*k+p]);
            float16x4_t y=vld1_f16((const float16_t*)(b+(size_t)p*stride+j));
            v[p]=vfmlalq_low_f16(vdupq_n_f32(0),x,vcombine_f16(y,vdup_n_f16(0)));
        }
        for(size_t p=(size_t)k;p<width;p++)v[p]=vdupq_n_f32(0);
        for(size_t w=1;w<width;w*=2)
            for(size_t p=0;p<width;p+=2*w)v[p]=vaddq_f32(v[p],v[p+w]);
        vst1q_f32(dst+(size_t)i*stride+j,v[0]);
    }
    free(v);
}
