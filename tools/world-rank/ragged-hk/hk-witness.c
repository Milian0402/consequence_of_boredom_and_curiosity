#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define ARGS int m,int n,int k,const float *a,int lda,const float *b,int ldb,float *c,int ldc
#define FORWARD m,n,k,a,lda,b,ldb,c,ldc
typedef void (*fn)(ARGS);
struct call { int m,n,k; const float *a,*b; float *c; int lda,ldb,ldc; };
static struct call caller;
static int public_calls,parents,leaves,lowered,fused_leaves,active_leaf,finishes,right,bottom,k_terms,ok=1;
static size_t core_checked;
static fn target;
static float *sx,*sy,*sp;

void __real_cob_sgemm_rowmajor(ARGS);
void __wrap_cob_sgemm_rowmajor(ARGS)
{
    ++public_calls;
    caller=(struct call){m,n,k,a,b,c,lda,ldb,ldc};
    __real_cob_sgemm_rowmajor(FORWARD);
}
void hk_trace_bind(int mh,int nh,int kh,float* x,float* y,float* product,fn real,int lower)
{
    ++parents;
    ok &= public_calls==1 && mh==caller.m/2 && nh==caller.n/3 && kh==caller.k/3 &&
        x && y==x+(size_t)mh*kh && product==y+(size_t)kh*nh && real!=NULL;
    lowered+=lower; sx=x;sy=y;sp=product;target=real;
}
void hk_trace_fused(void)
{
    if (active_leaf==1) {++fused_leaves;active_leaf=2;}
}
static int block_index(const float *value,const float *temporary,const float *base,
    int rows,int cols,int parent_ld,int arg_ld,int blocks)
{
    if(value==temporary) {ok &= arg_ld==cols; return -1;}
    for(int block=0;block<blocks;++block)
        if(value==base+(size_t)(block/3)*rows*parent_ld+(block%3)*cols) {
            ok &= arg_ld==parent_ld; return block;
        }
    ok=0; return -2;
}
void hk_trace_leaf(ARGS)
{
    /* Per-call positions are read directly from the immutable hk233 schedule. */
    static const signed char aa[15]={3,-1,1,-1,-1,-1,2,-1,-1,-1,-1,5,-1,-1,-1};
    static const signed char bb[15]={-1,-1,-1,1,-1,2,-1,-1,-1,3,5,-1,7,-1,6};
    static const signed char cc[15]={0,-1,-1,-1,-1,2,-1,-1,-1,-1,-1,-1,-1,-1,-1};
    const int ia=block_index(a,sx,caller.a,m,k,caller.lda,lda,6);
    const int ib=block_index(b,sy,caller.b,k,n,caller.ldb,ldb,9);
    const int ic=block_index(c,sp,caller.c,m,n,caller.ldc,ldc,6);
    ok &= leaves<15 && ia==aa[leaves%15] && ib==bb[leaves%15] && ic==cc[leaves%15] &&
          m==caller.m/2 && n==caller.n/3 && k==caller.k/3 && active_leaf==0;
    ++leaves; active_leaf=1; target(FORWARD); active_leaf=0;
}

#ifdef HK_RAGGED_TRACE
void __real_cob_hk15_ragged_finish(ARGS,float*,fn);
static float *edge_workspace;
static fn edge_target;
static void forward_edge(ARGS)
{
    const int mc=caller.m-caller.m%2,nc=caller.n-caller.n%3;
    ok &= k==caller.k && lda==caller.lda && ldc==caller.ldc;
    if (c==caller.c+nc) {
        ++right;
        ok &= m==mc && n==caller.n%3 && n>0 && a==caller.a && b==edge_workspace && ldb==n;
        if(ok) for(int p=0;p<k;++p)
            ok &= memcmp(b+(size_t)p*ldb,caller.b+(size_t)p*caller.ldb+nc,(size_t)n*sizeof(float))==0;
    } else {
        ++bottom;
        ok &= m==1 && caller.m%2==1 && n==caller.n && a==caller.a+(size_t)mc*caller.lda &&
              b==caller.b && ldb==caller.ldb && c==caller.c+(size_t)mc*caller.ldc;
    }
    edge_target(FORWARD);
}
static inline float expected_madd(float a,float b,float c)
{
#if defined(__FMA__) || defined(__ARM_FEATURE_FMA)
    return fmaf(a,b,c);
#else
    return c+a*b;
#endif
}
void __wrap_cob_hk15_ragged_finish(ARGS,float* workspace,fn base)
{
    ++finishes;
    ok &= public_calls==1 && parents==1 && leaves==15 && m==caller.m && n==caller.n && k==caller.k &&
        a==caller.a && lda==caller.lda && b==caller.b && ldb==caller.ldb && c==caller.c && ldc==caller.ldc &&
        (m%2 || n%3 || k%3) && workspace==sx && base!=NULL;
    edge_workspace=workspace;edge_target=base;
    const int mc=m-m%2,nc=n-n%3,kc=k-k%3;
    k_terms=k-kc;
    float *expected=NULL;
    if(k_terms) {
        expected=malloc((size_t)mc*nc*sizeof(float));
        if(!expected) {fputs("witness allocation failed\n",stderr);exit(2);}
        for(int i=0;i<mc;++i) for(int j=0;j<nc;++j) {
            float sum=c[(size_t)i*ldc+j];
            for(int p=kc;p<k;++p) sum=expected_madd(a[(size_t)i*lda+p],b[(size_t)p*ldb+j],sum);
            expected[(size_t)i*nc+j]=sum;
        }
    }
    __real_cob_hk15_ragged_finish(FORWARD,workspace,forward_edge);
    if(expected) {
        for(int i=0;i<mc;++i)
            ok &= !memcmp(c+(size_t)i*ldc,expected+(size_t)i*nc,(size_t)nc*sizeof(float));
        core_checked=(size_t)mc*nc;free(expected);
    }
}
#endif
__attribute__((destructor)) static void report(void)
{
    printf("HK,public_calls=%d,parents=%d,leaves=%d,lowered=%d,fused_leaves=%d,finishes=%d,right=%d,bottom=%d,k_terms=%d,core_checked=%zu,args_ok=%d\n",
        public_calls,parents,leaves,lowered,fused_leaves,finishes,right,bottom,k_terms,core_checked,ok);
}
