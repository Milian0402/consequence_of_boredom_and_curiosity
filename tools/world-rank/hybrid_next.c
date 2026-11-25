#define _POSIX_C_SOURCE 200112L
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "vendor.h"

extern rank_sgemm_fn rank_bridge;
extern void cob_sgemm_rowmajor(int,int,int,const float*,int,const float*,int,float*,int);

#ifndef RANK_HYBRID_LEVEL1_MIN
#define RANK_HYBRID_LEVEL1_MIN 896
#endif
#ifndef RANK_HYBRID_LEVEL2_MIN
#define RANK_HYBRID_LEVEL2_MIN 2048
#endif
#ifndef RANK_HYBRID_ALLOC
#define RANK_HYBRID_ALLOC(bytes) malloc(bytes)
#endif
#define MAX_WORK_FLOATS ((size_t)256*1024*1024/sizeof(float))

static int overlap(const float* x, size_t nx, const float* y, size_t ny)
{
    const uintptr_t px=(uintptr_t)x, py=(uintptr_t)y;
    return px <= py ? py-px < nx*sizeof(float) : px-py < ny*sizeof(float);
}

static size_t scratch_floats(int m,int n,int k,int levels)
{
    size_t count=0;
    for (int i=0;i<levels;++i) {
        m/=2; n/=2; k/=2;
        /* Limit all allocations before summing or converting to bytes. */
        const size_t sizes[3]={(size_t)m*k,(size_t)k*n,(size_t)m*n};
        for (int q=0;q<3;++q) {
            if (sizes[q] > MAX_WORK_FLOATS-count) return MAX_WORK_FLOATS+1;
            count+=sizes[q];
        }
    }
    return count;
}

static void combine(float* out,int rows,int cols,const float* a,int lda,
    const float* b,int ldb,int sign)
{
    for (int i=0;i<rows;++i) {
        float* d=out+(size_t)i*cols;
        const float* x=a+(size_t)i*lda;
        const float* y=b+(size_t)i*ldb;
        if (sign>0) {
            for (int j=0;j<cols;++j) d[j]=x[j]+y[j];
        } else {
            for (int j=0;j<cols;++j) d[j]=x[j]-y[j];
        }
    }
}

static void add_product(float* c,int ldc,const float* p,int m,int n,int sign)
{
    for (int i=0;i<m;++i) {
        float* out=c+(size_t)i*ldc;
        const float* in=p+(size_t)i*n;
        if (sign>0) {
            for (int j=0;j<n;++j) out[j]+=in[j];
        } else {
            for (int j=0;j<n;++j) out[j]-=in[j];
        }
    }
}

static void winograd(int m,int n,int k,const float* a,int lda,const float* b,int ldb,
    float* c,int ldc,float* scratch,int levels)
{
    if (levels==0) {
        rank_bridge(m,n,k,a,lda,b,ldb,c,ldc);
        return;
    }
    const int h=m/2,w=n/2,d=k/2;
    const float *a12=a+d,*a21=a+(size_t)h*lda,*a22=a21+d;
    const float *b12=b+w,*b21=b+(size_t)d*ldb,*b22=b21+w;
    float *c12=c+w,*c21=c+(size_t)h*ldc,*c22=c21+w;
    float *s=scratch,*t=s+(size_t)h*d,*p=t+(size_t)d*w,*child=p+(size_t)h*w;

    /* Four products go straight to the caller's output. Winograd reuses
       S1, S2 and T1, T2 in place and needs eight operand additions. */
#define CHILD(A,LD_A,B,LD_B,C,LD_C) winograd(h,w,d,A,LD_A,B,LD_B,C,LD_C,child,levels-1)
    CHILD(a,lda,b,ldb,c21,ldc);
    CHILD(a12,lda,b21,ldb,c,ldc);
    combine(s,h,d,a21,lda,a22,lda,1);
    combine(t,d,w,b12,ldb,b,ldb,-1);
    CHILD(s,d,t,w,c22,ldc);
    combine(s,h,d,s,d,a,lda,-1);
    combine(t,d,w,b22,ldb,t,w,-1);
    CHILD(s,d,t,w,c12,ldc);
    for (int i=0;i<h;++i) {
        for (int j=0;j<w;++j) {
            const size_t z=(size_t)i*ldc+j;
            const float p1=c21[z],u2=p1+c12[z],u4=u2+c22[z];
            c[z]+=p1; c12[z]=u4; c21[z]=u2; c22[z]=u4;
        }
    }
    combine(s,h,d,a12,lda,s,d,-1);
    CHILD(s,d,b22,ldb,p,w);
    add_product(c12,ldc,p,h,w,1);
    combine(t,d,w,t,w,b21,ldb,-1);
    CHILD(a22,lda,t,w,p,w);
    add_product(c21,ldc,p,h,w,-1);
    combine(s,h,d,a,lda,a21,lda,-1);
    combine(t,d,w,b22,ldb,b12,ldb,-1);
    CHILD(s,d,t,w,p,w);
    for (int i=0;i<h;++i) for (int j=0;j<w;++j) {
        const float last=p[(size_t)i*w+j];
        const size_t z=(size_t)i*ldc+j;
        c21[z]+=last; c22[z]+=last;
    }
#undef CHILD
}

static void complete_edges(int m,int n,int k,int mc,int nc,int kc,const float* a,int lda,
    const float* b,int ldb,float* c,int ldc)
{
    /* Peel at most three K terms. Rows and columns use the full depth and
       are disjoint. No dynamic allocation can fail after the core starts. */
    for (int i=0;k!=kc && i<mc;++i) {
        float* row=c+(size_t)i*ldc;
        for (int p=kc;p<k;++p) {
            const float av=a[(size_t)i*lda+p];
            const float* br=b+(size_t)p*ldb;
            for (int j=0;j<nc;++j) row[j]+=av*br[j];
        }
    }
    if (nc!=n) rank_bridge(mc,n-nc,k,a,lda,b+nc,ldb,c+nc,ldc);
    if (mc!=m) rank_bridge(m-mc,n,k,a+(size_t)mc*lda,lda,b,ldb,c+(size_t)mc*ldc,ldc);
}

void rank_hybrid_next(int m,int n,int k,const float* a,int lda,const float* b,int ldb,float* c,int ldc)
{
    if (m<=0 || n<=0 || k<=0 || !a || !b || !c || lda<k || ldb<n || ldc<n) return;
    const size_t out=(size_t)(m-1)*ldc+n;
    if (overlap(c,out,a,(size_t)(m-1)*lda+k) || overlap(c,out,b,(size_t)(k-1)*ldb+n)) {
        cob_sgemm_rowmajor(m,n,k,a,lda,b,ldb,c,ldc);
        return;
    }
    const int min=m<n?(m<k?m:k):(n<k?n:k);
    const int max=m>n?(m>k?m:k):(n>k?n:k);
    const double harmonic=(3.0*m*n*k)/((double)m*n+(double)m*k+(double)n*k);
    int levels=0;
    if (min>=RANK_HYBRID_LEVEL1_MIN/2 && (int64_t)max<=8*(int64_t)min &&
        harmonic>=RANK_HYBRID_LEVEL1_MIN) levels=1;
    if (levels && min>=RANK_HYBRID_LEVEL2_MIN/2 && harmonic>=RANK_HYBRID_LEVEL2_MIN) levels=2;
    while (levels>0) {
        const int alignment=1<<levels;
        const int mc=m-m%alignment,nc=n-n%alignment,kc=k-k%alignment;
        const size_t count=scratch_floats(mc,nc,kc,levels);
        float* scratch=count<=MAX_WORK_FLOATS ? RANK_HYBRID_ALLOC(count*sizeof(float)) : NULL;
        if (scratch) {
            winograd(mc,nc,kc,a,lda,b,ldb,c,ldc,scratch,levels);
            free(scratch);
            complete_edges(m,n,k,mc,nc,kc,a,lda,b,ldb,c,ldc);
            return;
        }
        --levels;
    }
    rank_bridge(m,n,k,a,lda,b,ldb,c,ldc);
}
