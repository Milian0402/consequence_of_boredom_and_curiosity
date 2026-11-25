#define _GNU_SOURCE
#include <dlfcn.h>
#include <link.h>
#include <stdint.h>
#include <stdio.h>

typedef void (*multiply_fn)(int,int,int,const float*,int,const float*,int,float*,int);
static multiply_fn multiply;
static const char* (*provider)(void);
#ifdef RANK_TRACE
static unsigned (*counter)(int);
static uint64_t (*field)(unsigned,unsigned);
unsigned rank_vendor_leaf_count(int tag) { return counter(tag); }
uint64_t rank_vendor_leaf_field(unsigned seq, unsigned key) { return field(seq,key); }
#endif
const char* rank_vendor_provider(void) { return provider(); }

int rank_vendor_init(void)
{
    if (multiply) return 0;
    void* handle = dlmopen(LM_ID_NEWLM, RANK_BACKEND_PATH, RTLD_NOW | RTLD_LOCAL);
    if (!handle) { fprintf(stderr, "private leaf namespace: %s\n", dlerror()); return 1; }
    int (*initialize)(void) = (int (*)(void))dlsym(handle, "rank_backend_init");
    multiply_fn entry = (multiply_fn)dlsym(handle,"rank_backend_sgemm");
    provider = (const char* (*)(void))dlsym(handle,"rank_backend_provider");
    if (!initialize || !entry || !provider || initialize()) { dlclose(handle); return 1; }
#ifdef RANK_TRACE
    counter = (unsigned (*)(int))dlsym(handle,"rank_backend_leaf_count");
    field = (uint64_t (*)(unsigned,unsigned))dlsym(handle,"rank_backend_leaf_field");
    if (!counter || !field) { dlclose(handle); return 1; }
#endif
    multiply = entry;
    return 0;
}

void rank_vendor_sgemm(int m,int n,int k,const float* a,int lda,const float* b,int ldb,float* c,int ldc)
{
    multiply(m,n,k,a,lda,b,ldb,c,ldc);
}
