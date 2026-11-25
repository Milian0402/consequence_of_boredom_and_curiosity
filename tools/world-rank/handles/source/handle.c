#define _GNU_SOURCE
#include "handle.h"
#include <mkl_cblas.h>
#include <mkl_service.h>
#include <dlfcn.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
_Static_assert(sizeof(MKL_INT)==sizeof(int), "LP64 oneMKL required");

struct plan {
    struct geometry g;
    void *handle;
    sgemm_jit_kernel_t kernel;
    int status;
};
static _Thread_local int last_status=-1;
#if HANDLE_PREPARED
static _Thread_local struct plan retained;
#endif
int rank_handle_last_status(void) { return last_status; }

int rank_vendor_init(void)
{
    /* The hosted worker runs one thread and loads a single benchmark plugin.
       Confirm that the exact sequential, LP64 and core DSOs are resident. */
    const char *paths[]= {MKL_PIN_DIR "/libmkl_intel_lp64.so.3",MKL_PIN_DIR "/libmkl_sequential.so.3",MKL_PIN_DIR "/libmkl_core.so.3"};
    for(size_t i=0;i<3;++i) {
        void *loaded=dlopen(paths[i],RTLD_NOW|RTLD_LOCAL|RTLD_NOLOAD);
        if (!loaded) return 1;
        dlclose(loaded);
    }
    return mkl_get_max_threads()!=1;
}
static int close_plan(struct plan *p)
{
    void *h=p->handle;
    *p=(struct plan){0};
    if (h && mkl_jit_destroy(h)!=MKL_JIT_SUCCESS) return -4;
    return 0;
}
static int open_plan(struct plan *p,struct geometry g)
{
    if (g.m<1 || g.n<1 || g.k<1 || g.lda<g.k || g.ldb<g.n || g.ldc<g.n) return -5;
    p->g=g;
    mkl_jit_status_t st=mkl_jit_create_sgemm(&p->handle,MKL_ROW_MAJOR,MKL_NOTRANS,MKL_NOTRANS,
        g.m,g.n,g.k,1.0f,g.lda,g.ldb,0.0f,g.ldc);
    if ((st!=MKL_JIT_SUCCESS && st!=MKL_NO_JIT) || !p->handle) {
        if (p->handle) (void)close_plan(p);
        return -2;
    }
    p->status=(int)st;
    p->kernel=mkl_jit_get_sgemm_ptr(p->handle);
    if (!p->kernel) { (void)close_plan(p); return -3; }
    return p->status;
}
#if HANDLE_PREPARED
int rank_handle_release(void) { return close_plan(&retained); }
__attribute__((destructor)) static void end_worker(void) { (void)rank_handle_release(); }

int rank_handle_prepare(int m,int n,int k,int lda,int ldb,int ldc)
{
    struct geometry g={m,n,k,lda,ldb,ldc};
    if (retained.handle && g.m==retained.g.m && g.n==retained.g.n && g.k==retained.g.k &&
            g.lda==retained.g.lda && g.ldb==retained.g.ldb && g.ldc==retained.g.ldc) return retained.status;
    struct plan fresh={0};
    int st=open_plan(&fresh,g);
    if (st<0) return st;
    if (close_plan(&retained)<0) { (void)close_plan(&fresh); return -4; }
    retained=fresh;
    return st;
}
#endif
void rank_vendor_sgemm(int m,int n,int k,const float *a,int lda,const float *b,int ldb,float *c,int ldc)
{
    if (m<=0 || n<=0) return;
    if (c==NULL || k<0 || ldc<n || (k>0 && (a==NULL || b==NULL || lda<k || ldb<n))) {last_status=-5;return;}
    if (k==0) {
        for(int i=0;i<m;++i) memset(c+(size_t)i*ldc,0,(size_t)n*sizeof(float));
        last_status=-5;
        return;
    }
#if HANDLE_PREPARED
    int st=rank_handle_prepare(m,n,k,lda,ldb,ldc);
    if (st>=0) retained.kernel(retained.handle,(float *)a,(float *)b,c);
#else
    struct plan fresh={0};
    int st=open_plan(&fresh,(struct geometry){m,n,k,lda,ldb,ldc});
    if(st>=0) {
        fresh.kernel(fresh.handle,(float *)a,(float *)b,c);
        if (close_plan(&fresh)<0) st=-4;
    }
#endif
    last_status=st;
    if(st<0) for(int i=0;i<m;++i) for(int j=0;j<n;++j) c[(size_t)i*ldc+j]=NAN;
}
