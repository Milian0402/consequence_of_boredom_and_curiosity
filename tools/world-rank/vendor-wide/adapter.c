#define _GNU_SOURCE
#include <dlfcn.h>
#include <link.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
typedef void (*multiply)(int,int,int,const float*,int,const float*,int,float*,int);
static multiply entry;
#ifdef WIDE_TRACE
static void (*snapshot)(uint64_t*);
#endif
int rank_vendor_init(void)
{
    if (entry) return 0;
    void *handle = dlmopen(LM_ID_NEWLM, WIDE_BACKEND_PATH, RTLD_NOW | RTLD_LOCAL);
    if (!handle) { fprintf(stderr, "%s\n", dlerror()); return 1; }
    int (*initialize)(void);
    void *sym=dlsym(handle,"wide_backend_init");
    memcpy(&initialize,&sym,sizeof initialize);
    sym=dlsym(handle,"wide_backend_gemm");
    multiply fn;
    memcpy(&fn,&sym,sizeof fn);
#ifdef WIDE_TRACE
    sym=dlsym(handle,"wide_backend_counts");
    memcpy(&snapshot,&sym,sizeof snapshot);
    if (!snapshot) { dlclose(handle); return 1; }
#endif
    if (!initialize || !fn || initialize()) { dlclose(handle); return 1; }
    entry=fn;
    return 0;
}
void rank_vendor_sgemm(int m,int n,int k,const float*a,int lda,const float*b,int ldb,float*c,int ldc)
{
    entry(m,n,k,a,lda,b,ldb,c,ldc);
}
#ifdef WIDE_TRACE
void rank_wide_snapshot(uint64_t out[8]) { snapshot(out); }
#endif
