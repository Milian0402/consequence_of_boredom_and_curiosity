/* Test-only scalar FP64 oracle used for the real kernel's FP32 rescue hook. */
#if defined(__x86_64__)
#include <xmmintrin.h>
#endif

static int calls;

void precision_range_rescue(int m, int n, int k, const float *a, int lda,
                           const float *b, int ldb, float *c, int ldc)
{
    ++calls;
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            double sum = 0;
            for (int p = 0; p < k; ++p)
                sum += (double)a[i*lda+p] * b[p*ldb+j];
            c[i*ldc+j] = (float)sum;
        }
}

int test_fused_rescue_calls(void) { return calls; }

int test_enable_ftz_daz(void)
{
#if defined(__x86_64__)
    unsigned old = _mm_getcsr();
    _mm_setcsr(old | (1u << 15) | (1u << 6));
    return (int)old;
#else
    return -1;
#endif
}

void test_restore_mxcsr(int old)
{
#if defined(__x86_64__)
    _mm_setcsr((unsigned)old);
#else
    (void)old;
#endif
}
