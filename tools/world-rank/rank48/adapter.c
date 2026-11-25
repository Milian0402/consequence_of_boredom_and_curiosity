/* Benchmark adapter for the certified rational block transforms in schedule.h. */
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../vendor.h"

struct view { const float *p; int ld; };
struct target { float *p; int ld; };
typedef float vec __attribute__((vector_size(16)));
static inline vec load_vec(const float *p) { vec x; memcpy(&x, p, sizeof x); return x; }
static inline void store_vec(float *p, vec x) { memcpy(p, &x, sizeof x); }
#include "schedule.h"

static rank_sgemm_fn gemm;
static _Thread_local int last_core, last_fringe;

int rank_vendor_init(void)
{
    const char *path = getenv("RANK_FINITE_LEAF");
    if (!path) return 1;
    void *library = dlopen(path, RTLD_LOCAL | RTLD_NOW);
    if (!library) { fprintf(stderr, "%s\n", dlerror()); return 1; }
    int (*init)(void) = (int (*)(void))dlsym(library, "rank_vendor_init");
    gemm = (rank_sgemm_fn)dlsym(library, "rank_vendor_sgemm");
    return !gemm || !init || init();
}

int rank_core_products(void) { return last_core; }
int rank_fringe_products(void) { return last_fringe; }

static size_t count_forms(const int *aliases)
{
    size_t count = 0;
    for (int i = 0; i < PRODUCT_COUNT; ++i) count += aliases[i] < 0;
    return count;
}

static void make_input(int rows, int cols, const float *base, int ld, const int *aliases,
                       void (*transform)(int, int, const struct view *, const struct target *),
                       float **space, struct view *forms)
{
    struct view blocks[16];
    struct target outputs[PRODUCT_COUNT] = {{0}};
    for (int i = 0; i < 16; ++i) {
        blocks[i] = (struct view){base + (size_t)(i / 4) * rows * ld + (i % 4) * cols, ld};
    }
    for (int i = 0; i < PRODUCT_COUNT; ++i) {
        if (aliases[i] >= 0) {
            forms[i] = blocks[aliases[i]];
        } else {
            outputs[i] = (struct target){*space, cols};
            forms[i] = (struct view){*space, cols};
            *space += (size_t)rows * cols;
        }
    }
    transform(rows, cols, blocks, outputs);
}

void rank_vendor_sgemm(int m, int n, int k, const float *a, int lda,
                      const float *b, int ldb, float *c, int ldc)
{
    last_core = last_fringe = 0;
    if (!gemm || m < 4 || n < 4 || k < 4 || lda < k || ldb < n || ldc < n) abort();
    const int h = m / 4, w = n / 4, d = k / 4;
    const size_t na = (size_t)h * d, nb = (size_t)d * w, nc = (size_t)h * w;
    const size_t la = count_forms(left_aliases), lb = count_forms(right_aliases);
    const size_t capacity = SIZE_MAX / sizeof(float);
    if (na > capacity / PRODUCT_COUNT || nb > (capacity - na * la) / PRODUCT_COUNT ||
        nc > (capacity - na * la - nb * lb) / PRODUCT_COUNT) abort();
    float *storage = malloc((na * la + nb * lb + nc * PRODUCT_COUNT) * sizeof(float));
    if (!storage) abort();
    struct view l[PRODUCT_COUNT], r[PRODUCT_COUNT], products[PRODUCT_COUNT];
    struct target outputs[16];
    float *next = storage;
    make_input(h, d, a, lda, left_aliases, left, &next, l);
    make_input(d, w, b, ldb, right_aliases, right, &next, r);
    for (int i = 0; i < PRODUCT_COUNT; ++i) {
        products[i] = (struct view){next, w};
        gemm(h, w, d, l[i].p, l[i].ld, r[i].p, r[i].ld, next, w);
        ++last_core;
        next += nc;
    }
    for (int i = 0; i < 16; ++i)
        outputs[i] = (struct target){c + (size_t)(i / 4) * h * ldc + (i % 4) * w, ldc};
    recombine(h, w, products, outputs);
    free(storage);
    const int mc = h * 4, nc_core = w * 4, kc = d * 4;
    for (int depth = kc; depth < k; ++depth) {
        for (int row = 0; row < mc; ++row) {
            float scale = a[(size_t)row * lda + depth];
            for (int col = 0; col < nc_core; ++col)
                c[(size_t)row * ldc + col] += scale * b[(size_t)depth * ldb + col];
        }
    }
    if (n > nc_core) {
        gemm(mc, n - nc_core, k, a, lda, b + nc_core, ldb, c + nc_core, ldc);
        ++last_fringe;
    }
    if (m > mc) {
        gemm(m - mc, n, k, a + (size_t)mc * lda, lda, b, ldb, c + (size_t)mc * ldc, ldc);
        ++last_fringe;
    }
}
