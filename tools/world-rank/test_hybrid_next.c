#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define RANK_HYBRID_LEVEL1_MIN 48
#define RANK_HYBRID_LEVEL2_MIN 96
static int fail_mode, allocations, native_calls, vendor_calls;
static void* checked_alloc(size_t size)
{
    ++allocations;
    if (fail_mode==2 || (fail_mode==1 && allocations==1)) return NULL;
    return malloc(size);
}
#define RANK_HYBRID_ALLOC(bytes) checked_alloc(bytes)
#include "hybrid_next.c"

static void mock_blas(int m,int n,int k,const float* a,int lda,const float* b,int ldb,
    float* c,int ldc)
{
    ++vendor_calls;
    for (int i=0;i<m;++i) {
        for (int j=0;j<n;++j) {
            float total=0;
            for (int p=0;p<k;++p) total=fmaf(a[(size_t)i*lda+p],b[(size_t)p*ldb+j],total);
            c[(size_t)i*ldc+j]=total;
        }
    }
}

rank_sgemm_fn rank_bridge=mock_blas;
void cob_sgemm_rowmajor(int m,int n,int k,const float* a,int lda,const float* b,int ldb,
    float* c,int ldc)
{
    (void)m; (void)n; (void)k; (void)a; (void)lda; (void)b; (void)ldb; (void)c; (void)ldc;
    ++native_calls;
}

static uint32_t rng=8384731u;
static float next_value(void)
{
    rng=1664525u*rng+1013904223u;
    return ((int)(rng>>12)%2001-1000)/997.0f;
}

static double worst;
static size_t checked;
static int check(int m,int n,int k,int pad,int mode,int fail,int wanted_calls)
{
    const int lda=k+pad,ldb=n+pad,ldc=n+pad+3;
    float *a=malloc((size_t)m*lda*sizeof(float)), *b=malloc((size_t)k*ldb*sizeof(float));
    float *c=malloc(((size_t)m*ldc+1)*sizeof(float));
    if (!a || !b || !c) exit(1);
    for (int i=0;i<m;++i) for (int p=0;p<lda;++p) {
        float x=next_value();
        if (mode==1 && (p&1)) x=a[(size_t)i*lda+p-1];
        if (mode==2 && p>k/2) x*=0x1p-9f;
        a[(size_t)i*lda+p]=p<k?x:NAN;
    }
    for (int p=0;p<k;++p) for (int j=0;j<ldb;++j) {
        float x=next_value();
        if (mode==1 && (p&1)) x=-b[(size_t)(p-1)*ldb+j];
        if (mode==2 && p<=k/2) x*=0x1p-9f;
        b[(size_t)p*ldb+j]=j<n?x:NAN;
    }
    for (size_t p=0;p<=(size_t)m*ldc;++p) c[p]=-32001;
    for (int i=0;i<m;++i)
        for (int j=0;j<n;++j) c[(size_t)i*ldc+j]=NAN;
    allocations=native_calls=vendor_calls=0;
    fail_mode=fail;
    rank_hybrid_next(m,n,k,a,lda,b,ldb,c,ldc);
    if (vendor_calls!=wanted_calls || native_calls) {
        fprintf(stderr,"wrong route %d x %d x %d: vendor %d, native %d, want %d\n",
                m,n,k,vendor_calls,native_calls,wanted_calls);
        return 1;
    }
    for (int i=0;i<m;++i) {
        for (int j=0;j<n;++j) {
            double ref=0;
            for (int p=0;p<k;++p) ref+=(double)a[(size_t)i*lda+p]*b[(size_t)p*ldb+j];
            double err=fabs(ref-c[(size_t)i*ldc+j]);
            if (!isfinite(err) || err>0.002) return 2;
            if (err>worst) worst=err;
            ++checked;
        }
        for (int j=n;j<ldc;++j) if(c[(size_t)i*ldc+j]!=-32001) return 3;
    }
    if(c[(size_t)m*ldc]!=-32001) return 3;
    rank_hybrid_next(m,n,k,a,lda,b,ldb,a+1,ldc);
    rank_hybrid_next(m,n,k,a,lda,b,ldb,b+1,ldc);
    if (native_calls!=2 || vendor_calls!=wanted_calls) return 4;
    free(c); free(b); free(a);
    return 0;
}
int main(void)
{
    const int layouts[][6]={{32,64,17,0,0,1},{64,64,64,0,0,7},{67,71,69,5,0,9},
      {96,96,96,0,0,49},{99,123,109,5,0,51},{103,143,110,7,1,9},
      {100,144,124,0,1,7},{99,123,109,5,2,1},{101,121,107,4,0,51},{112,192,96,7,0,49}};
    for (size_t i=0;i<sizeof(layouts)/sizeof(layouts[0]);++i) {
        const int* s=layouts[i];
        for (int mode=0;mode<3;++mode)
            if (check(s[0],s[1],s[2],s[3],mode,s[4],s[5])) return 1;
    }
    if (scratch_floats(2147483644,2147483644,2147483644,2)<=MAX_WORK_FLOATS) return 1;
    printf("%zu FP64 checks, greatest absolute error %.9g, overflow/alias/failure/route checks passed\n",
           checked,worst);
    return 0;
}
