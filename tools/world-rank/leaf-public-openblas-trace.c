#include <cblas.h>

/* Linux linker interposition on the actual leaf entry, in the trace DSO only. */
static _Thread_local int calls;

int rank_openblas_winograd_calls(void)
{
    return calls;
}

void __real_cblas_sgemm(enum CBLAS_ORDER, enum CBLAS_TRANSPOSE, enum CBLAS_TRANSPOSE,
    blasint, blasint, blasint, float, const float*, blasint, const float*, blasint,
    float, float*, blasint);
void __wrap_cblas_sgemm(enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE ta, enum CBLAS_TRANSPOSE tb,
    blasint m, blasint n, blasint k, float alpha, const float* a, blasint lda,
    const float* b, blasint ldb, float beta, float* c, blasint ldc)
{
    ++calls;
    __real_cblas_sgemm(order, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
