#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* Independent scalar regeneration of logical values, without any parent allocations. */
static uint64_t matrix_hash(int rows, int cols, int operand, int mode, uint32_t* seed) {
    uint64_t hash = UINT64_C(14695981039346656037);
    const int hr = (rows+1)/2, hc = (cols+1)/2;
    for (int i=0; i<rows; ++i) {
        for (int j=0; j<cols; ++j) {
            float x;
            if (mode == 2) {
                int r=i%hr, q=j%hc;
                float v = operand ? ((r*31+q*13)%503-251)/263.0f : ((r*17+q*29)%509-254)/257.0f;
                float eps = operand ? (((r+2*q)&1)?0.0008f:-0.0006f) : (((r+q)&1)?0.0007f:-0.0009f);
                if (i<hr && j<hc) x=v;
                else if (i<hr) x=-v+2.0f*eps;
                else if (j<hc) x=v+eps;
                else x=-v+eps;
            } else {
                *seed=1664525u*(*seed)+1013904223u;
                x=((int)((*seed>>8)%2001u)-1000)/997.0f;
                if (mode==3 && (operand?i<rows/2:j>=cols/2)) x*=0x1p-8f;
            }
            uint32_t u; memcpy(&u,&x,sizeof u);
            for (int byte=0;byte<4;++byte) {
                hash ^= (u>>(byte*8))&255u;
                hash *= UINT64_C(1099511628211);
            }
        }
    }
    return hash;
}

int main(void) {
    const char* f[]={"repeat-win-deeper","repeat-win-ragged","fresh-short"};
    const int dims[][3]={{108,1728,320},{137,2179,347},{43,2291,137}};
    const char* mode[]={"random","random","cancel","scaled"};
    puts("family,mode,seed,a_hash,b_hash");
    for(int i=0;i<3;i++) for (int j=0;j<4;j++) {
        uint32_t seed=(j==1||j==3)?38641729u:73194513u, state=seed;
        const int* s=dims[i];
        uint64_t a=matrix_hash(s[0],s[2],0,j,&state);
        uint64_t b=matrix_hash(s[2],s[1],1,j,&state);
        printf("%s,%s,%" PRIu32 ",%016" PRIx64 ",%016" PRIx64 "\n",f[i],mode[j],seed,a,b);
    }
    return 0;
}
