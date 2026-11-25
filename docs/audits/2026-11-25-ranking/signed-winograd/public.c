#include "cob_gemm.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void)
{
    enum { m=2049,n=2051,k=2053,lda=k+3,ldb=n+5,ldc=n+11 };
    const size_t as=(size_t)m*lda,bs=(size_t)k*ldb,cs=(size_t)m*ldc;
    float *a=calloc(as,sizeof(float)),*b=malloc(bs*sizeof(float)),*c=malloc((cs+2)*sizeof(float));
    float *sa=malloc(as*sizeof(float)),*sb=malloc(bs*sizeof(float));
    if(!a || !b || !c || !sa || !sb) return 2;
    for(int i=0;i<m;++i) {
        a[i*lda]=(float)(i%13)/16;
        a[i*lda+k/2]=(float)(7-i%11)/8;
        a[i*lda+k-1]=(float)(i%9-4)/32;
        for(int p=k;p<lda;++p) a[i*lda+p]=768.0f;
    }
    for(int p=0;p<k;++p) for(int j=0;j<ldb;++j) b[p*ldb+j]=j<n?(float)((p%5)-(j%7))/16:777.0f;
    for(size_t x=0;x<cs+2;++x) c[x]=537.0f;
    memcpy(sa,a,as*sizeof(float)); memcpy(sb,b,bs*sizeof(float));
    cob_sgemm_rowmajor(m,n,k,a,lda,b,ldb,c+1,ldc);
    int ok=c[0]==537.0f && c[cs+1]==537.0f && !memcmp(sa,a,as*sizeof(float)) && !memcmp(sb,b,bs*sizeof(float));
    for(int i=0;i<m;++i) {
        for(int j=0;j<n;++j) {
            double expect=(double)a[i*lda]*b[j]+(double)a[i*lda+k/2]*b[(k/2)*ldb+j]+(double)a[i*lda+k-1]*b[(k-1)*ldb+j];
            if(c[1+i*ldc+j]!=expect) ok=0;
        }
        for(int j=n;j<ldc;++j) if(c[1+i*ldc+j]!=537.0f) ok=0;
    }
    printf("ordinary separately linked public API signed=%d checked=%d guards_and_inputs=1 exact=%d\n",COB_ENABLE_SIGNED_WINOGRAD_LEAVES,m*n,ok);
    free(sa);free(sb);free(a);free(b);free(c);
    return ok?0:1;
}
