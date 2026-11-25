#include "cob_gemm.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
unsigned rank_vendor_leaf_count(int tag);
int rank_vendor_init(void);

static float sign(int i,int j) { return (i+j)%2==0 ? 1.0f : -1.0f; }
int main(int argc, char **argv)
{
    if (argc != 2 || rank_vendor_init() != 0) return 1;
    const unsigned expected_add = (unsigned)atoi(argv[1]);
    const int m=2049,n=2115,k=2123, lda=k+3,ldb=n+4,ldc=n+5;
    float* a=malloc((size_t)m*lda*sizeof(float));
    float* b=malloc((size_t)k*ldb*sizeof(float));
    float* c=malloc((size_t)m*ldc*sizeof(float));
    if(!a||!b||!c) return 1;
    for(int i=0;i<m;i++) for(int p=0;p<lda;++p) a[i*lda+p]=p<k ? sign(i,p) : 32391;
    for(int p=0;p<k;p++) for(int j=0;j<ldb;++j) b[p*ldb+j]=j<n ? sign(p,j) : -9824;
    for(int i=0;i<m;i++) for(int j=0;j<ldc;++j) c[i*ldc+j]=j<n ? NAN : 531;
    cob_sgemm_rowmajor(m,n,k,a,lda,b,ldb,c,ldc);
    unsigned beta0 = rank_vendor_leaf_count(0), beta1 = rank_vendor_leaf_count(1);
    if (beta0 != 51 || beta1 != expected_add || rank_vendor_leaf_count(2) != beta0+beta1) return 2;
    for(int i=0;i<m;i++) for(int j=0;j<ldc;++j)
        if(c[i*ldc+j]!=(j<n ? k*sign(i,j) : 531)) return 3;
    for(int i=0;i<m;i++) for(int p=0;p<lda;++p)
        if(a[i*lda+p]!=(p<k ? sign(i,p) : 32391)) return 4;
    for(int p=0;p<k;p++) for(int j=0;j<ldb;++j)
        if(b[p*ldb+j]!=(j<n ? sign(p,j) : -9824)) return 5;
    printf("production public entry 2049x2115x2123, checked %d exact outputs, input and output padding, %u real overwrites, %u add calls\n",m*n,beta0,beta1);
    free(a);free(b);free(c);
}
