/* Stream every dense result for an independent SHA-256 comparison of the
   pristine public release and the research tree compiled with the flag off. */
#include "cob_gemm.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t state=7367491;
static float next_float(void)
{
    state=1664525u*state+1013904223u;
    return ((int)(state>>16)-32768)*(1.0f/32768);
}
int main(int argc, char **argv)
{
    if(argc!=5) return 2;
    const int m=atoi(argv[1]),n=atoi(argv[2]),k=atoi(argv[3]),pad=atoi(argv[4]);
    if(m<=0 || n<=0 || k<=0 || pad<0 || m>8192 || n>8192 || k>8192 || pad>127) return 2;
    const int lda=k+pad,ld=n+pad;
    size_t ac=(size_t)m*lda,bc=(size_t)k*ld,cc=(size_t)m*ld;
    float* a=malloc(ac*sizeof(*a)); float* b=malloc(bc*sizeof(*b)); float* c=malloc(cc*sizeof(*c));
    if(!a || !b || !c) return 3;
    for(size_t i=0;i<ac;++i) a[i]=next_float();
    for(size_t i=0;i<bc;++i) b[i]=next_float();
    for(size_t i=0;i<cc;++i) c[i]=36571.0f;
    cob_sgemm_rowmajor(m,n,k,a,lda,b,ld,c,ld);
    state=7367491;
    for(size_t i=0;i<ac;++i) if(a[i]!=next_float()) return 1;
    for(size_t i=0;i<bc;++i) if(b[i]!=next_float()) return 1;
    for(int i=0;i<m;++i){
        const float* row=c+(size_t)i*ld;
        for(int j=0;j<n;++j) if(!isfinite(row[j]) || row[j]==36571.0f) return 1;
        for(int j=n;j<ld;++j) if(row[j]!=36571.0f) return 1;
        if(fwrite(row,sizeof(*row),(size_t)n,stdout)!=(size_t)n) return 1;
    }
    free(c);free(b);free(a);return fflush(stdout)!=0;
}
