#define _POSIX_C_SOURCE 200112L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int allocations, fail_allocation, fail_from;
static int checked_memalign(void** p, size_t alignment, size_t size)
{
    ++allocations;
    if (allocations == fail_allocation || (fail_from && allocations >= fail_from)) {
        return 12;
    }
    return posix_memalign(p, alignment, size);
}
#define posix_memalign checked_memalign
#ifndef COB_TEST_SOURCE
#define COB_TEST_SOURCE "../src/gemm.c"
#endif
#include COB_TEST_SOURCE
#undef posix_memalign

static uint32_t rng = 25;
static float draw(void)
{
    rng = rng * 1664525u + 1013904223u;
    return ((int)(rng >> 8) % 2001 - 1000) / 997.0f;
}

static const float canary = 123456.0f;
static double worst_error;
static int cases;

static int check_full(int m, int n, int k, int lda, int ldb, int ldc,
    int expected_route, int pattern, int fail, int after)
{
    size_t ac = (size_t)m * lda, bc = (size_t)k * ldb, cc = (size_t)m * ldc;
    float* a = malloc(ac*sizeof(float)); float* b = malloc(bc*sizeof(float));
    float* c = malloc(cc*sizeof(float)); float* old = malloc(cc*sizeof(float));
    if (!a || !b || !c || !old) exit(1);
    for (size_t i=0;i<ac;++i) a[i] = draw();
    for (size_t i=0;i<bc;++i) b[i] = draw();
    if (pattern) {
        for (int p=1;p<k;p+=2) {
            for (int i=0;i<m;++i) a[(size_t)i*lda+p] = a[(size_t)i*lda+p-1];
            for (int j=0;j<n;++j) b[(size_t)p*ldb+j] = -b[(size_t)(p-1)*ldb+j];
        }
    }
    for (size_t i=0;i<cc;++i) old[i]=c[i]=canary;
    allocations=fail_from=fail_allocation=0;
    int route=cob_sgemm_rowmajor_strassen1(m,n,k,a,lda,b,ldb,c,ldc);
    if (route!=expected_route || (expected_route && allocations!=8) ||
        (!expected_route && allocations!=0)) {
        fprintf(stderr,"route/allocations %d/%d expected %d for %d %d %d\n",route,allocations,expected_route,m,n,k);
        return 1;
    }
    for (size_t i=0;i<cc;++i) c[i]=canary;
    allocations=0;fail_allocation=fail;fail_from=after;
    cob_sgemm_rowmajor(m,n,k,a,lda,b,ldb,c,ldc);
    fail_allocation=fail_from=0;
    if (!expected_route || fail==1) {
        cob_sgemm_rowmajor_fallback(m,n,k,a,lda,b,ldb,old,ldc);
        if (memcmp(c,old,cc*sizeof(float))) return 2;
    }
    for (int i=0;i<m;++i) {
        for (int j=0;j<n;++j) {
            double sum=0;
            for (int p=0;p<k;++p) sum+=(double)a[(size_t)i*lda+p]*b[(size_t)p*ldb+j];
            double err=fabs(c[(size_t)i*ldc+j]-sum);
            if (!isfinite(err) || err>0.002) return 3;
            if (err>worst_error) worst_error=err;
        }
        for (int j=n;j<ldc;++j) if (c[(size_t)i*ldc+j]!=canary) return 4;
    }
    ++cases;free(old);free(c);free(b);free(a);return 0;
}

static int check_alias(int alias_a)
{
    int n=128;size_t nc=(size_t)n*n;
    float* a=malloc((nc+1)*sizeof(float));float* b=malloc((nc+1)*sizeof(float));
    float* a2=malloc((nc+1)*sizeof(float));float* b2=malloc((nc+1)*sizeof(float));
    for (size_t i=0;i<nc+1;++i) {a[i]=draw(); b[i]=draw();}
    memcpy(a2,a,(nc+1)*sizeof(float));memcpy(b2,b,(nc+1)*sizeof(float));
    float* c=(alias_a?a:b)+1;float* c2=(alias_a?a2:b2)+1;
    allocations=0;
    if (cob_sgemm_rowmajor_strassen1(n,n,n,a,n,b,n,c,n) || allocations) return 5;
    cob_sgemm_rowmajor(n,n,n,a,n,b,n,c,n);
    cob_sgemm_rowmajor_fallback(n,n,n,a2,n,b2,n,c2,n);
    if (memcmp(c,c2,nc*sizeof(float))) return 6;
    ++cases;free(b2);free(a2);free(b);free(a);return 0;
}

int main(void)
{
    const int shapes[][10] = {
        {128,128,128,128,128,128,1,0,0,0}, {192,256,192,192,256,263,1,1,0,0},
        {128,128,128,128,128,135,1,0,1,0}, {128,128,128,128,128,135,1,0,2,0},
        {128,128,128,128,128,135,1,0,0,2}, {64,128,128,128,128,128,0,0,0,0},
        {128,129,128,128,129,131,0,0,0,0}, {128,128,192,192,128,128,0,0,0,0},
        {128,128,128,131,128,128,0,0,0,0}, {128,128,128,128,133,128,0,0,0,0}
    };
    for (size_t i=0;i<sizeof(shapes)/sizeof(shapes[0]);++i) {
        const int* s=shapes[i];
        int result=check_full(s[0],s[1],s[2],s[3],s[4],s[5],s[6],s[7],s[8],s[9]);
        if (result) {fprintf(stderr,"case %zu failed %d\n",i,result);return 1;}
    }
    if (check_alias(0) || check_alias(1)) return 1;
    printf("%d forced native Strassen and guard/alias/fallback cases passed; max full-matrix FP64 error %.9g\n",cases,worst_error);
    return 0;
}
