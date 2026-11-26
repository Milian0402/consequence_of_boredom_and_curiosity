#include <arm_neon.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Separate whole-call reference for binary FP32 inputs to C=A*B. */
int binary_dot(int m,int n,int k,const float *a,int la,const float*b,int lb,float*c,int lc){
    if(m<=0||n<=0||k<=0||k>4096||!a||!b||!c||la<k||lb<n||lc<n)return 0;
    size_t words=2*((size_t)(k+127)/128);
    if((size_t)m+n>SIZE_MAX/sizeof(uint64_t)/words)return 0;
    uint64_t*tmp=calloc(((size_t)m+n)*words,sizeof(*tmp));if(!tmp)return 0;
    uint64_t *pb=tmp+(size_t)m*words;
    for(int i=0;i<m;i++)for(int p=0;p<k;p++){uint32_t v;memcpy(&v,a+(size_t)i*la+p,4);
        if(!(v&0x7fffffffu))continue;if(v!=0x3f800000u){free(tmp);return 0;}
        tmp[(size_t)i*words+p/64]|=UINT64_C(1)<<(p%64);}
    for(int p=0;p<k;p++)for(int j=0;j<n;j++){uint32_t v;memcpy(&v,b+(size_t)p*lb+j,4);
        if(!(v&0x7fffffffu))continue;if(v!=0x3f800000u){free(tmp);return 0;}
        pb[(size_t)j*words+p/64]|=UINT64_C(1)<<(p%64);}
    for(int i=0;i<m;i++)for(int j=0;j<n;j++){
        const uint8_t*x=(const uint8_t*)(tmp+(size_t)i*words),*y=(const uint8_t*)(pb+(size_t)j*words);
        uint16x8_t acc=vdupq_n_u16(0);
        for(size_t p=0;p<words*8;p+=16)acc=vpadalq_u8(acc,vcntq_u8(vandq_u8(vld1q_u8(x+p),vld1q_u8(y+p))));
        c[(size_t)i*lc+j]=(float)vaddlvq_u16(acc);
    }
    free(tmp);return 1;
}
