#define _DARWIN_C_SOURCE
#include <dlfcn.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "mtmm.h"

typedef void (*compact_gemm)(int, int, int, const float *, int, const float *, int,
                             float *, int, const mtmm_config *);
static compact_gemm native;
__attribute__((constructor)) static void init(void)
{
    const char *path = getenv(COMPACT_BACKEND_ENV);
    if (!path) abort();
    void *lib = dlopen(path, RTLD_NOW | RTLD_LOCAL | RTLD_FIRST);
    if (!lib) abort();
    void *p = dlsym(lib, "mtmm_sgemm_rowmajor");
    if (!p) abort();
    _Static_assert(sizeof(p) == sizeof(native), "function pointer size");
    memcpy(&native, &p, sizeof(native));
}
static size_t span(int rows, int cols)
{
    size_t r = (size_t)rows, c = (size_t)cols;
    if (!r || !c || r > (SIZE_MAX - 95) / c) abort();
    /* Native kernels receive the same 64-float allocation guard as the original harness. */
    return (r * c + 95) & ~(size_t)31;
}
void mtmm_sgemm_rowmajor(int m, int n, int k, const float *a, int lda, const float *b,
                         int ldb, float *out, int ldc, const mtmm_config *cfg)
{
    if (!cfg || cfg->num_threads != 1 || m < 1 || n < 1 || k < 1 ||
        lda < k || ldb < n || ldc < n || !a || !b || !out) abort();
    const size_t ac = span(m, k), bc = span(k, n), cc = span(m, n);
    const size_t limit = SIZE_MAX / sizeof(float);
    if (ac > limit || bc > limit - ac || cc > limit - ac - bc) abort();
    float *left = NULL;
    if (posix_memalign((void **)&left, 128, (ac + bc + cc) * sizeof(float))) abort();
    float *right = left + ac, *result = right + bc;
    for (int i = 0; i < m; ++i) memcpy(left + (size_t)i*k, a + (size_t)i*lda, (size_t)k*sizeof(float));
    for (int p = 0; p < k; ++p) memcpy(right + (size_t)p*n, b + (size_t)p*ldb, (size_t)n*sizeof(float));
    native(m, n, k, left, k, right, n, result, n, cfg);
    for (int i = 0; i < m; ++i) memcpy(out + (size_t)i*ldc, result + (size_t)i*n, (size_t)n*sizeof(float));
    free(left);
}
void cob_sgemm_rowmajor(int m, int n, int k, const float *a, int lda, const float *b,
                        int ldb, float *out, int ldc)
{
    const mtmm_config cfg = {.num_threads=1};
    mtmm_sgemm_rowmajor(m,n,k,a,lda,b,ldb,out,ldc,&cfg);
}
