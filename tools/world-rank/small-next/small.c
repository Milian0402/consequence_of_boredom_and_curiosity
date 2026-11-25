/* One-shot runtime shapes with no heap allocation or external kernel dependencies.
   Define SMALL_NO_PLUGIN to link the try function directly into a dispatcher. */
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define SMALL_INLINE static inline __attribute__((always_inline))
#if defined(__aarch64__) && defined(__ARM_NEON)
#define SMALL_SUPPORTED 1
#include "arm.inc"
#elif defined(__SSE2__)
#define SMALL_SUPPORTED 1
#include "x86.inc"
#endif

#if defined(SMALL_SUPPORTED)
static int small_overlap(const float *x, size_t nx, const float *y, size_t ny)
{
    uintptr_t px = (uintptr_t)x, py = (uintptr_t)y;
    return px <= py ? py - px < nx * sizeof(float) : px - py < ny * sizeof(float);
}
#endif

int small_try_sgemm(int m, int n, int k, const float *a, int lda,
                    const float *b, int ldb, float *c, int ldc)
{
#ifdef SMALL_SUPPORTED
    if (m < 8 || m > 128 || n < 8 || n > 128 || k < 8 || k > 128 ||
        a == NULL || b == NULL || c == NULL || lda < k || ldb < n || ldc < n)
        return 0;
    const size_t count = (size_t)(m - 1) * ldc + n;
    if (small_overlap(c,count,a,(size_t)(m - 1) * lda + k) ||
        small_overlap(c,count,b,(size_t)(k - 1) * ldb + n)) return 0;
    small_native(m,n,k,a,lda,b,ldb,c,ldc);
    return 1;
#else
    (void)m; (void)n; (void)k; (void)a; (void)lda;
    (void)b; (void)ldb; (void)c; (void)ldc;
    return 0;
#endif
}

#ifndef SMALL_NO_PLUGIN
int rank_vendor_init(void) { return 0; }

void rank_vendor_sgemm(int m, int n, int k, const float *a, int lda,
                       const float *b, int ldb, float *c, int ldc)
{
    if (small_try_sgemm(m,n,k,a,lda,b,ldb,c,ldc)) return;
    if (m < 1 || n < 1 || k < 1 || a == NULL || b == NULL || c == NULL ||
        lda < k || ldb < n || ldc < n) return;
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            float value = 0.0f;
            for (int p = 0; p < k; ++p) value += a[(size_t)i*lda+p] * b[(size_t)p*ldb+j];
            c[(size_t)i*ldc+j] = value;
        }
}
#endif
