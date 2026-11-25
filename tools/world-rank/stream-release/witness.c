/* Both test-only wrappers forward the complete call to the original kernel
   resolved from the same static archive as the timed public executable. */
#include <stdio.h>
static int calls_avx2, calls_avx512;
#define ARGUMENTS int m, int n, int k, const float *a, int lda, const float *b, int ldb, float *c, int ldc, float *panel
#define FORWARD m,n,k,a,lda,b,ldb,c,ldc,panel
#ifdef STREAM_RELEASE_ENABLED
void __real_cob_stream_b_kernel_avx2(ARGUMENTS);
void __real_cob_stream_b_kernel_avx512(ARGUMENTS);
void __wrap_cob_stream_b_kernel_avx2(ARGUMENTS)
{
    ++calls_avx2;
    __real_cob_stream_b_kernel_avx2(FORWARD);
}
void __wrap_cob_stream_b_kernel_avx512(ARGUMENTS)
{
    ++calls_avx512;
    __real_cob_stream_b_kernel_avx512(FORWARD);
}
#endif
__attribute__((destructor)) static void report(void)
{
    printf("DISPATCH,avx2=%d,avx512=%d\n",calls_avx2,calls_avx512);
}
