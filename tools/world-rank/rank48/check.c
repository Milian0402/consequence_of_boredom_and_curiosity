#define _POSIX_C_SOURCE 200809L
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../vendor.h"

static uint64_t hash_bytes(const void *p, size_t n)
{
    const unsigned char *b = p;
    uint64_t hash = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < n; ++i) hash = (hash ^ b[i]) * 0x100000001b3ULL;
    return hash;
}

int main(int argc, char **argv)
{
    if (argc != 7) return 2;
    int rank = atoi(argv[2]), m = atoi(argv[3]), n = atoi(argv[4]), k = atoi(argv[5]), pad = atoi(argv[6]);
    if (m < 4 || n < 4 || k < 4 || pad < 0 || m > 8192 || n > 8192 || k > 8192 || pad > 512) return 2;
    int lda = k + pad, ldb = n + pad, ldc = n + pad;
    size_t na = (size_t)m * lda, nb = (size_t)k * ldb, nc = (size_t)m * ldc;
    float *a = malloc(na * sizeof *a), *b = malloc(nb * sizeof *b), *c = malloc(nc * sizeof *c);
    void *lib = dlopen(argv[1], RTLD_LOCAL | RTLD_NOW);
    if (!a || !b || !c || !lib) return 2;
    int (*init)(void) = (int (*)(void))dlsym(lib, "rank_vendor_init");
    rank_sgemm_fn gemm = (rank_sgemm_fn)dlsym(lib, "rank_vendor_sgemm");
    int (*core)(void) = (int (*)(void))dlsym(lib, "rank_core_products");
    int (*fringe)(void) = (int (*)(void))dlsym(lib, "rank_fringe_products");
    if (!init || !gemm || !core || !fringe || init()) return 2;
    for (int i = 0; i < m; ++i)
        for (int p = 0; p < lda; ++p)
            a[(size_t)i * lda + p] = p < k ? ((i % 7) - 3) * (p % 2 ? -1 : 1) : NAN;
    for (int p = 0; p < k; ++p)
        for (int j = 0; j < ldb; ++j)
            b[(size_t)p * ldb + j] = j < n ? ((j % 5) - 2) * (p % 2 ? -1 : 1) : NAN;
    for (size_t i = 0; i < nc; ++i) c[i] = -17103.5f;
    uint64_t before_a = hash_bytes(a, na * sizeof *a), before_b = hash_bytes(b, nb * sizeof *b);
    gemm(m, n, k, a, lda, b, ldb, c, ldc);
    size_t errors = 0;
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < ldc; ++j) {
            float expect = j < n ? ((i % 7) - 3) * ((j % 5) - 2) * k : -17103.5f;
            errors += c[(size_t)i * ldc + j] != expect;
        }
    int unchanged = before_a == hash_bytes(a, na * sizeof *a) && before_b == hash_bytes(b, nb * sizeof *b);
    int tested = core() == rank && fringe() == ((m % 4 != 0) + (n % 4 != 0));
    printf("%d,%d,%d,%d,%d,%d,%d,%zu,%zu,%d,%d\n",
           rank,m,n,k,pad,core(),fringe(),(size_t)m*n,errors,unchanged,tested);
    free(a); free(b); free(c);
    return errors || !unchanged || !tested;
}
