#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200112L
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(COB_ENABLE_OPENBLAS_LEAF) && COB_ENABLE_OPENBLAS_LEAF && \
    defined(COB_ENABLE_ALIGNED_WINOGRAD) && COB_ENABLE_ALIGNED_WINOGRAD
#include <cblas.h>
#define COB_TEST_THREAD_TRANSITION 1
static int simulate_thread_change;
static unsigned simulated_calls;
#define openblas_get_num_threads() \
    (simulate_thread_change && simulated_calls >= 49 ? 2 : openblas_get_num_threads())
#define cblas_sgemm(...) ((void)(simulate_thread_change && ++simulated_calls), cblas_sgemm(__VA_ARGS__))
#endif

static int fail_workspace;
static unsigned workspace_calls;
static int test_memalign(void** ptr, size_t align, size_t size)
{
    ++workspace_calls;
    return fail_workspace ? 12 : posix_memalign(ptr, align, size);
}
#define posix_memalign test_memalign
#include "../src/gemm.c"
#undef posix_memalign
#if defined(COB_TEST_THREAD_TRANSITION)
#undef openblas_get_num_threads
#undef cblas_sgemm
#endif

/* Toggle every combination of bottom, right, and depth tails. The first and
   last depth terms make both beta=0 and beta=1 errors visible. */
static int check_edge(unsigned mask)
{
    enum { max_m = 129, max_n = 65, max_k = 97, lda = max_k+3,
           ldb = max_n+5, ldc = max_n+7 };
    float a[max_m*lda], b[max_k*ldb], c[max_m*ldc];
    const int m=128+(int)(mask&1u), n=64+(int)((mask>>1)&1u), k=96+(int)((mask>>2)&1u);
    for (int i=0; i<m; ++i) {
        for (int p=0; p<lda; ++p) a[i*lda+p] = p<k ? 0.0f : NAN;
        a[i*lda] = (float)(1+i%5);
        a[i*lda+k-1] = (float)(3+i%3);
        for (int j=0; j<ldc; ++j) c[i*ldc+j] = j<n ? NAN : 531.0f;
    }
    for (int p=0; p<k; ++p) {
        for (int j=0; j<ldb; ++j) {
            float value = 0.0f;
            if (p==0) value = (float)(1+j%7);
            if (p==k-1) value = (float)(-1-j%4);
            b[p*ldb+j] = j<n ? value : NAN;
        }
    }
    if (!cob_cblas_winograd_try(m,n,k,a,lda,b,ldb,c,ldc,64)) return 0;
    for (int i=0; i<m; ++i) {
        for (int j=0; j<n; ++j) {
            float expected = a[i*lda]*b[j] + a[i*lda+k-1]*b[(k-1)*ldb+j];
            if (c[i*ldc+j] != expected) return 0;
        }
        for (int j=n; j<ldc; ++j) if (c[i*ldc+j] != 531.0f) return 0;
    }
    return 1;
}

int main(void)
{
    enum { m = 71, n = 74, k = 73, lda = k+5, ldb = n+3, ldc = n+7 };
    float a[m*lda], b[k*ldb], c[m*ldc], original_a[m*lda], original_b[k*ldb];
    for (int i=0; i<m; ++i) {
        for (int p=0; p<lda; ++p)
            a[i*lda+p] = p<k ? (float)(((i*3+p*5)%29)-14)/13 : NAN;
    }
    for (int p=0; p<k; ++p) {
        for (int j=0; j<ldb; ++j)
            b[p*ldb+j] = j<n ? (float)(((p*7+j*2)%31)-15)/17 : NAN;
    }
    memcpy(original_a, a, sizeof(a));
    memcpy(original_b, b, sizeof(b));
    for (int i=0; i<m*ldc; ++i) c[i] = 531.0f;
#if COB_ENABLE_OPENBLAS_LEAF
    openblas_set_num_threads(1);
#endif
    if (!cob_cblas_leaf_is_serial()) return 1;
    if (cob_cblas_winograd_try(m,n,k,a,lda,b,ldb,c,ldc,2048) ||
        cob_cblas_winograd_try(m,n,k,a,k-1,b,ldb,c,ldc,64) ||
        cob_cblas_winograd_try(m,n,k,a,lda,b,ldb,a,ldc,64) ||
        cob_cblas_winograd_try(64,64,64,a,lda,b,ldb,c,ldc,64)) return 1;
    if (workspace_calls != 0) return 1;
    fail_workspace = 1;
    if (cob_cblas_winograd_try(m,n,k,a,lda,b,ldb,c,ldc,64)) return 1;
    if (workspace_calls != 1) return 1;
    fail_workspace = 0;
#if COB_ENABLE_OPENBLAS_LEAF
    openblas_set_num_threads(2);
    if (openblas_get_num_threads() > 1 &&
        cob_cblas_winograd_try(m,n,k,a,lda,b,ldb,c,ldc,64)) return 1;
    openblas_set_num_threads(1);
#endif
    for (int i=0; i<m*ldc; ++i) if (c[i] != 531.0f) return 1;
    if (!cob_cblas_winograd_try(m,n,k,a,lda,b,ldb,c,ldc,64)) return 1;
    if (workspace_calls != 2) return 1;
    /* A second multiplication must allocate again and overwrite the old C. */
    if (!cob_cblas_winograd_try(m,n,k,a,lda,b,ldb,c,ldc,64) || workspace_calls != 3) return 1;
    if (memcmp(original_a,a,sizeof(a)) || memcmp(original_b,b,sizeof(b))) return 1;
    for (int i=0; i<m; ++i) {
        for (int j=0; j<n; ++j) {
            double ref = 0;
            for (int p=0; p<k; ++p) ref += (double)a[i*lda+p]*b[p*ldb+j];
            if (!isfinite(c[i*ldc+j]) || fabs(ref-c[i*ldc+j]) > 0.002) return 1;
        }
        for (int j=n; j<ldc; ++j) if (c[i*ldc+j] != 531.0f) return 1;
    }
    for (unsigned mask=0; mask<8; ++mask) if (!check_edge(mask)) return 1;
    if (workspace_calls != 11) return 1;
#if defined(COB_TEST_THREAD_TRANSITION)
    /* Switch the simulated setting immediately after the last core product.
       Mask 4 has a depth remainder and no bottom or right strip. */
    simulate_thread_change = 1;
    if (!check_edge(4) || simulated_calls != 49) return 1;
#endif
    puts("Rectangular CBLAS Winograd passed FP64, depth-edge, stride, alias and allocation checks.");
    return 0;
}
