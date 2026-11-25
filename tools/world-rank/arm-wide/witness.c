/* Link this file only into the route worker, against the ordinary archive.
   Both modes wrap cob_sgemm_rowmajor. The enabled build also wraps
   cob_arm_stream_b_product and defines ARM_STREAM_ENABLED=1. */
#include <stdio.h>

#define ARGS int m, int n, int k, const float* a, int lda, const float* b, int ldb, float* c, int ldc
#define FORWARD m, n, k, a, lda, b, ldb, c, ldc
static int public_calls, stream_calls, arguments_ok = 1;
#ifdef ARM_STREAM_ENABLED
static int active;
static struct {
    int m,n,k,lda,ldb,ldc;
    const float *a,*b,*c;
} current;
#endif
void __real_cob_sgemm_rowmajor(ARGS);
void __wrap_cob_sgemm_rowmajor(ARGS)
{
    ++public_calls;
#ifdef ARM_STREAM_ENABLED
    current.m=m; current.n=n; current.k=k;
    current.lda=lda; current.ldb=ldb; current.ldc=ldc;
    current.a=a; current.b=b; current.c=c;
    ++active;
#endif
    __real_cob_sgemm_rowmajor(FORWARD);
#ifdef ARM_STREAM_ENABLED
    --active;
#endif
}
#ifdef ARM_STREAM_ENABLED
void __real_cob_arm_stream_b_product(ARGS);
void __wrap_cob_arm_stream_b_product(ARGS)
{
    ++stream_calls;
    arguments_ok &= active == 1 && current.m == m && current.n == n && current.k == k &&
                    current.a == a && current.b == b && current.c == c &&
                    current.lda == lda && current.ldb == ldb && current.ldc == ldc;
    __real_cob_arm_stream_b_product(FORWARD);
}
#endif
int arm_stream_route_count(void) { return stream_calls; }
__attribute__((destructor)) static void report(void)
{
    printf("ARMSTREAM,public_calls=%d,stream_calls=%d,args_ok=%d\n",
           public_calls,stream_calls,arguments_ok);
}
