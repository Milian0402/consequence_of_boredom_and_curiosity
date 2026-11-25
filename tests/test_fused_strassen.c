#define _POSIX_C_SOURCE 200112L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Force multi-slab geometry and the cost threshold independently of machine speed. */
#define COB_ENABLE_PORTABLE_STRASSEN 1
#if defined(__aarch64__)
#define COB_USE_NEON 1
#endif
#define COB_FUSED_STRASSEN_MIN_HARMONIC_DIM 64
#define COB_FUSED_MC 47
#define COB_FUSED_NC 61

static int refuse_next;
static int count_allocations;
static size_t allocations, workspace_bytes;

static int test_memalign(void** p, size_t align, size_t bytes)
{
    if (count_allocations) {
        ++allocations;
        workspace_bytes = bytes;
    }
    if (refuse_next) {
        refuse_next = 0;
        return 12;
    }
    return posix_memalign(p, align, bytes);
}

#define posix_memalign test_memalign
#include "../src/gemm.c"
#undef posix_memalign

static uint32_t state = 25;
static float draw(void)
{
    state = 1664525u*state + 1013904223u;
    return (int)((state >> 8) % 2001)/997.0f - 1;
}

static size_t checked;
static double worst_error;

static int check(int m, int n, int k, int pad, int pattern)
{
    const int lda = k+pad, ldb = n+pad, ldc = n+pad;
    float* a = malloc((size_t)m*lda*sizeof(float));
    float* b = malloc((size_t)k*ldb*sizeof(float));
    float* guard = malloc(((size_t)m*ldc+2)*sizeof(float));
    if (!a || !b || !guard) exit(1);
    float* c = guard+1;
    float poison = 12345.0f;
    for (int i=0; i<m; ++i)
        for (int p=0; p<lda; ++p)
            a[(size_t)i*lda+p] = p<k ? draw() : NAN;
    for (int p=0; p<k; ++p)
        for (int j=0; j<ldb; ++j)
            b[(size_t)p*ldb+j] = j<n ? draw() : NAN;
    if (pattern) {
        for (int p=1; p<k; p+=2) {
            for (int i=0; i<m; ++i) a[(size_t)i*lda+p] = a[(size_t)i*lda+p-1];
            for (int j=0; j<n; ++j) b[(size_t)p*ldb+j] = -b[(size_t)(p-1)*ldb+j];
        }
    }
    for (size_t i=0; i<(size_t)m*ldc+2; ++i) guard[i] = poison;
    count_allocations = 1;
    refuse_next = 1;
    allocations = 0;
    if (cob_sgemm_strassen_fused(m,n,k,a,lda,b,ldb,c,ldc) || allocations != 1) return 1;
    for (size_t i=0; i<(size_t)m*ldc+2; ++i) if (guard[i] != poison) return 2;
    allocations = 0;
    cob_sgemm_rowmajor(m,n,k,a,lda,b,ldb,c,ldc);
    count_allocations = 0;
    if (allocations != 1 || workspace_bytes > 256*1024) return 3;
    for (int i=0; i<m; ++i) {
        for (int j=0; j<n; ++j) {
            double ref = 0;
            for (int p=0; p<k; ++p) ref += (double)a[(size_t)i*lda+p]*b[(size_t)p*ldb+j];
            double err = fabs(ref-c[(size_t)i*ldc+j]);
            if (!isfinite(err) || err>0.002) return 4;
            if (err>worst_error) worst_error = err;
            ++checked;
        }
        for (int j=n; j<ldc; ++j) if (c[(size_t)i*ldc+j] != poison) return 5;
    }
    if (guard[0] != poison || guard[(size_t)m*ldc+1] != poison) return 6;
    if (cob_sgemm_strassen_fused(m,n,k,a,lda,b,ldb,a+1,ldc) ||
        cob_sgemm_strassen_fused(m,n,k,a,lda,b,ldb,b+1,ldc) ||
        cob_sgemm_strassen_fused(m,n,k,a,k-1,b,ldb,c,ldc) ||
        cob_sgemm_strassen_fused(m,n,k,a,lda,b,n-1,c,ldc) ||
        cob_sgemm_strassen_fused(m,n,k,a,lda,b,ldb,c,n-1) ||
        cob_sgemm_strassen_fused(16,16,16,a,lda,b,ldb,c,ldc) ||
        cob_sgemm_strassen_fused(16,1024,16,a,lda,b,ldb,c,ldc)) return 7;
    free(guard);
    free(b);
    free(a);
    return 0;
}

int main(void)
{
    cob_blocked_kernel kernel;
    if (!cob_blocked_select_kernel(&kernel)) {
        fprintf(stderr,"No supported blocked SIMD kernel on this CPU\n");
        return 77;
    }
    const int dims[][3] = {{64,64,64}, {88,90,68}, {89,90,69}, {88,91,68}, {101,135,97},
                            {193,261,151}, {169,99,521}, {194,131,523}};
    for (size_t q=0; q<sizeof(dims)/sizeof(dims[0]); ++q) {
        for (int pattern = 0; pattern < 2; ++pattern) {
            int error = check(dims[q][0],dims[q][1],dims[q][2],q%2 ? 7 : 0,pattern);
            if (error) {
                fprintf(stderr,"Shape %zu, pattern %d failed check %d\n",q,pattern,error);
                return 1;
            }
        }
    }
    printf("tile %dx%d, %zu FP64 comparisons, worst absolute error %.9g\n",
           kernel.mr,kernel.nr,checked,worst_error);
    return 0;
}
