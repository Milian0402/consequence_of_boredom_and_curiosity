#ifndef RANK_PREFIX_H
#define RANK_PREFIX_H
#include <stddef.h>
#include <stdint.h>

#define GEMM_ARGS int m, int n, int k, const float *a, int lda, const float *b, int ldb, float *c, int ldc
#define GEMM_PASS m,n,k,a,lda,b,ldb,c,ldc
typedef void (*gemm_fn)(GEMM_ARGS);
struct call {
    int m, n, k, lda, ldb, ldc;
    uintptr_t a, b, c;
};
struct receipt {
    int intel_avx2, eligible, prefix_calls;
    struct call incoming, prefix;
};
static inline struct call actual(GEMM_ARGS)
{
    return (struct call){m,n,k,lda,ldb,ldc,(uintptr_t)a,(uintptr_t)b,(uintptr_t)c};
}
#endif
