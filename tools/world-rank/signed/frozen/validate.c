#define _GNU_SOURCE
#include <cblas.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static struct {
    int m, n, k, calls, core_calls, plus, minus, depth, right, bottom;
    int stop_after, fallback_seen, bad;
    float* dest[7];
    int strides[7];
} trace;
static int provider_threads(void)
{
    if (trace.stop_after >= 0 && trace.calls >= trace.stop_after) {
        trace.fallback_seen = 1;
        return 2;
    }
    return openblas_get_num_threads();
}
static void trace_sgemm(enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE ta, enum CBLAS_TRANSPOSE tb,
    blasint m, blasint n, blasint k, float alpha, const float* a, blasint lda, const float* b, blasint ldb,
    float beta, float* c, blasint ldc)
{
    const int cm=trace.m/64*64, cn=trace.n/64*64, ck=trace.k/32*32;
    ++trace.calls;
    if (order != CblasRowMajor || ta != CblasNoTrans || tb != CblasNoTrans) trace.bad=1;
    if (m==cm/4 && n==cn/4 && k==ck/4) {
        int pos=trace.core_calls++%7;
        trace.dest[pos]=c;
        trace.strides[pos]=ldc;
        if (COB_ENABLE_SIGNED_WINOGRAD_LEAVES && (pos==4 || pos==5)) {
            int ref=pos==4?3:0;
            if (beta!=1 || alpha!=(pos==4?1:-1) || c!=trace.dest[ref] || ldc!=trace.strides[ref])
                trace.bad=1;
            if (pos==4) ++trace.plus; else ++trace.minus;
        } else if (alpha!=1 || beta!=0) trace.bad=1;
    } else if (m==cm && n==cn && k==trace.k-ck && beta==1 && alpha==1) ++trace.depth;
    else if (m==cm && n==trace.n-cn && k==trace.k && beta==0 && alpha==1) ++trace.right;
    else if (m==trace.m-cm && n==trace.n && k==trace.k && beta==0 && alpha==1) ++trace.bottom;
    else trace.bad=1;
    cblas_sgemm(order,ta,tb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc);
}
#define cblas_sgemm trace_sgemm
#define openblas_get_num_threads provider_threads
#include "source/src/gemm.c"
#undef cblas_sgemm
#undef openblas_get_num_threads

static uint32_t seed=6823273;
static float fraction(void)
{
    seed=1664525u*seed+1013904223u;
    return ((int)(seed>>16)-32768)*(1.0f/32768);
}
enum { guard=16 };
static float* buffer(size_t length)
{
    float* data=malloc((length+2*guard)*sizeof(float));
    if (!data) exit(2);
    for (size_t i=0;i<length+2*guard;++i) data[i]=9351.5f;
    return data+guard;
}
static int check(int m,int n,int k,int pattern,int actual_public,int stop_after)
{
    const int lda=k+3, ldb=n+5, ldc=n+11;
    size_t as=(size_t)m*lda, bs=(size_t)k*ldb, cs=(size_t)m*ldc;
    float *a=buffer(as),*b=buffer(bs),*c=buffer(cs),*copy_a=buffer(as),*copy_b=buffer(bs);
    for (int i=0;i<m;++i) for(int p=0;p<k;++p) {
        float x=pattern==2?0.0f:fraction();
        if(pattern==1) x=0.5f+(float)(i%4)/128;
        if(pattern==2 && (p==0 || p==k-1)) x=1+(float)((i+p)%7)/8;
        a[(size_t)i*lda+p]=x;
    }
    for(int p=0;p<k;++p) for(int j=0;j<n;++j) {
        float x=pattern==2?(float)((p%7)-(j%11))/8:fraction();
        if(pattern==1) x=(p%2?-1.0f:1.0f)*(0.5f+(j%9)/128.0f)+x/4096;
        b[(size_t)p*ldb+j]=x;
    }
    for(int i=0;i<m;++i) for(int j=0;j<n;++j) c[(size_t)i*ldc+j]=NAN;
    memcpy(copy_a-guard,a-guard,(as+2*guard)*sizeof(float));
    memcpy(copy_b-guard,b-guard,(bs+2*guard)*sizeof(float));
    memset(&trace,0,sizeof(trace));
    trace.m=m;trace.n=n;trace.k=k;trace.stop_after=stop_after;
    int taken=1;
    if (actual_public) cob_sgemm_rowmajor(m,n,k,a,lda,b,ldb,c,ldc);
    else taken=cob_cblas_winograd_try(m,n,k,a,lda,b,ldb,c,ldc,64);
    int ok=taken && !trace.bad && !memcmp(copy_a-guard,a-guard,(as+2*guard)*sizeof(float)) &&
        !memcmp(copy_b-guard,b-guard,(bs+2*guard)*sizeof(float));
    if(stop_after>=0) ok=ok && trace.fallback_seen && trace.calls==stop_after;
    else ok=ok && trace.core_calls==49 && trace.plus==7*COB_ENABLE_SIGNED_WINOGRAD_LEAVES &&
        trace.minus==7*COB_ENABLE_SIGNED_WINOGRAD_LEAVES && trace.depth==(k%32!=0) &&
        trace.right==(n%64!=0) && trace.bottom==(m%64!=0);
    double err=0.0;
    for(int i=0;i<m;++i) {
        for(int j=0;j<n;++j) {
            double v=0;
            if(pattern==2) v=(double)a[(size_t)i*lda]*b[j]+(double)a[(size_t)i*lda+k-1]*b[(size_t)(k-1)*ldb+j];
            else for (int p=0;p<k;++p) v+=(double)a[(size_t)i*lda+p]*b[(size_t)p*ldb+j];
            double diff=fabs((double)c[(size_t)i*ldc+j]-v);
            if(!isfinite(diff) || diff>0.002) ok=0;
            if(diff>err) err=diff;
        }
        for(int j=n;j<ldc;++j) if(c[(size_t)i*ldc+j]!=9351.5f) ok=0;
    }
    for(int j=1;j<=guard;++j) if(c[-j]!=9351.5f || c[cs+j-1]!=9351.5f) ok=0;
    printf("signed=%d public=%d m=%d n=%d k=%d pattern=%d stop=%d core=%d plus=%d minus=%d depth=%d right=%d bottom=%d checked=%d max_abs=%.12g passed=%d\n",
        COB_ENABLE_SIGNED_WINOGRAD_LEAVES,actual_public,m,n,k,pattern,stop_after,trace.core_calls,trace.plus,trace.minus,trace.depth,trace.right,trace.bottom,m*n,err,ok);
    free(a-guard);free(b-guard);free(c-guard);free(copy_a-guard);free(copy_b-guard);
    return ok;
}
int main(void)
{
    trace.stop_after=-1;
    printf("provider=%s threads=%d\n",openblas_get_config(),openblas_get_num_threads());
    if(openblas_get_num_threads()!=1) return 1;
    int ok=1;
    for(unsigned mask=0;mask<8;++mask) {
        int m=128+(mask&1), n=192+((mask>>1)&1), k=160+((mask>>2)&1);
        ok &=check(m,n,k,0,0,-1);
        ok &=check(m,n,k,1,0,-1);
    }
    ok &=check(129,193,161,0,0,4);
    ok &=check(129,193,161,1,0,5);
    ok &=check(2049,2051,2053,2,1,-1);
    return ok?0:1;
}
