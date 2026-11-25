#define _GNU_SOURCE
#include "handle.h"
#include <mkl_cblas.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
static _Thread_local char op_owner[160],op_perms[8];
const char *rank_handle_callable_owner(void) {return op_owner;}
const char *rank_handle_callable_perms(void) {return op_perms;}
static void observe_address(uintptr_t address)
{
    FILE *maps=fopen("/proc/self/maps","r");
    if (!maps) abort();
    char line[2048];
    while (fgets(line,sizeof line,maps)) {
        unsigned long lo,hi;char perms[8];int n=0;
        if (sscanf(line,"%lx-%lx %7s %*s %*s %*s %n",&lo,&hi,perms,&n)!=3 || address<lo || address>=hi) continue;
        char *name=line+n;while(*name==' ' || *name=='\t') ++name;
        name[strcspn(name,"\r\n")]=0;
        const char *basename=strrchr(name,'/');if(basename) name=(char *)basename+1;
        snprintf(op_owner,sizeof op_owner,"%.150s", *name ? name : "anonymous");
        snprintf(op_perms,sizeof op_perms,"%s",perms);
        if (!strchr(op_perms,'x')) abort();
        fclose(maps);return;
    }
    abort();
}

static _Thread_local struct observation obs;
struct slot {void *handle;sgemm_jit_kernel_t kernel;int status;};
static _Thread_local struct slot slots[3];
static _Thread_local int fault;
void rank_handle_fault(int f) {fault=f;}
void rank_handle_trace_reset(void) {op_owner[0]=op_perms[0]=0;obs=(struct observation){.create_status=-1,.execution_status=-1,.destroy_status=-1};}
void rank_handle_trace_read(struct observation *out) {*out=obs;}
static struct slot *find(void *h)
{
    for(int i=0;i<3;++i) if(slots[i].handle==h) return &slots[i];
    abort();
}
#define CREATE_ARGS void **h,const MKL_LAYOUT layout,const MKL_TRANSPOSE ta,const MKL_TRANSPOSE tb, \
    const MKL_INT m,const MKL_INT n,const MKL_INT k,const float alpha,const MKL_INT lda, \
    const MKL_INT ldb,const float beta,const MKL_INT ldc
mkl_jit_status_t __real_mkl_cblas_jit_create_sgemm(CREATE_ARGS);
sgemm_jit_kernel_t __real_mkl_jit_get_sgemm_ptr(const void *);
mkl_jit_status_t __real_mkl_jit_destroy(void *);
mkl_jit_status_t __wrap_mkl_cblas_jit_create_sgemm(CREATE_ARGS)
{
    obs.creates++;obs.m=m;obs.n=n;obs.k=k;obs.lda=lda;obs.ldb=ldb;obs.ldc=ldc;
    obs.layout=layout;obs.transa=ta;obs.transb=tb;obs.alpha=alpha;obs.beta=beta;
    mkl_jit_status_t st=fault==1 ? MKL_JIT_ERROR :
        __real_mkl_cblas_jit_create_sgemm(h,layout,ta,tb,m,n,k,alpha,lda,ldb,beta,ldc);
    obs.create_status=(int)st;
    if (*h) {struct slot *s=find(NULL);s->handle=*h;s->status=(int)st;}
    return st;
}
static void call_real(void *handle,float *a,float *b,float *c)
{
    struct slot *s=find(handle);
    if(!s->kernel) abort();
    observe_address((uintptr_t)s->kernel);obs.calls++;obs.handle=(uintptr_t)handle;obs.kernel=(uintptr_t)s->kernel;obs.execution_status=s->status;
    obs.a=(uintptr_t)a;obs.b=(uintptr_t)b;obs.c=(uintptr_t)c;
    s->kernel(handle,a,b,c);
}
sgemm_jit_kernel_t __wrap_mkl_jit_get_sgemm_ptr(const void *h)
{
    obs.gets++;
    struct slot *s=find((void *)h);
    s->kernel=(fault==2) ? NULL : __real_mkl_jit_get_sgemm_ptr(h);
    obs.handle=(uintptr_t)h;obs.kernel=(uintptr_t)s->kernel;
    if(s->kernel) observe_address(obs.kernel);
    return s->kernel ? call_real : NULL;
}
mkl_jit_status_t __wrap_mkl_jit_destroy(void *h)
{
    obs.destroys++;struct slot *s=find(h);*s=(struct slot){0};
    mkl_jit_status_t st=__real_mkl_jit_destroy(h);
    obs.destroy_status=(int)st;return st;
}
