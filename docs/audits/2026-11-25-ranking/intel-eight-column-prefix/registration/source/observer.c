#define _GNU_SOURCE
#include "prefix.h"
#include <dlfcn.h>
#include <stdlib.h>
static gemm_fn multiply;
static _Thread_local struct call calls[2];
static _Thread_local int count;
int rank_observed(int index, struct call *out)
{
    if (index >= 0 && index < count && index < 2) *out = calls[index];
    return count;
}
void rank_observer_reset(void) { count = 0; }
int rank_vendor_init(void)
{
    if (multiply) return 0;
    const char *path = getenv("RANK_PREFIX_REAL_MKL");
    if (!path || path[0]!='/') return 1;
    void *lib = dlopen(path, RTLD_NOW|RTLD_LOCAL);
    if (!lib) return 1;
    int (*initialize)(void) = (int (*)(void))dlsym(lib,"rank_vendor_init");
    gemm_fn entry = (gemm_fn)dlsym(lib,"rank_vendor_sgemm");
    if (!initialize || !entry || initialize()) return 1;
    multiply=entry;
    return 0;
}
void rank_vendor_sgemm(GEMM_ARGS)
{
    if (count < 2) calls[count] = actual(GEMM_PASS);
    ++count;
    multiply(GEMM_PASS);
}
