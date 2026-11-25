#define _POSIX_C_SOURCE 200809L
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(__x86_64__)
#include <cpuid.h>
#endif

typedef void (*sgemm_fn)(int,int,int,const float *,int,const float *,int,float *,int);
static sgemm_fn multiply;
static char provider[16];

#ifdef HOST_PAD_TRACE
struct pad_trace { int padded, m, n, k, lda, ldb, ldc, calls; };
static _Thread_local struct pad_trace last;
void rank_host_trace(int *out, const char **name)
{
    const int fields[] = {last.padded,last.m,last.n,last.k,last.lda,last.ldb,last.ldc,last.calls};
    memcpy(out,fields,sizeof fields);
    *name=provider;
}
#define RECORD(p,rows,cols,depth,la,lb,lc) \
    do { last = (struct pad_trace){p,rows,cols,depth,la,lb,lc,1}; } while (0)
#else
#define RECORD(p,rows,cols,depth,la,lb,lc) ((void)0)
#endif

int rank_vendor_init(void)
{
    if (multiply) return 0;
#if defined(__x86_64__)
    unsigned a,b,c,d;
    char id[13] = {0};
    if (!__get_cpuid(0,&a,&b,&c,&d)) return 1;
    memcpy(id,&b,4); memcpy(id+4,&d,4); memcpy(id+8,&c,4);
    if (!strcmp(id,"GenuineIntel")) strcpy(provider,"mkl");
    else if (!strcmp(id,"AuthenticAMD")) strcpy(provider,"aocl");
    else return 1;
#else
    return 1;
#endif
    const char *root = getenv("RANK_HOST_BLAS_DIR");
    char path[4096];
    if (!root) return 1;
    int count=snprintf(path,sizeof path,"%s/%s.so",root,provider);
    if (count < 0 || (size_t)count >= sizeof path) return 1;
    void *handle=dlopen(path,RTLD_NOW|RTLD_LOCAL);
    if (!handle) { fprintf(stderr,"%s\n",dlerror()); return 1; }
    int (*init)(void)=(int (*)(void))dlsym(handle,"rank_vendor_init");
    sgemm_fn entry=(sgemm_fn)dlsym(handle,"rank_vendor_sgemm");
    if (!init || !entry || init()) { dlclose(handle); return 1; }
    multiply=entry;
    fprintf(stderr,"host-provider=%s policy=pad8-%d\n",provider,HOST_PAD);
    return 0;
}

void rank_vendor_sgemm(int m,int n,int k,const float *a,int lda,const float *b,int ldb,float *c,int ldc)
{
#if HOST_PAD
    if (m>=24 && m<=96 && n>=1024 && n<=4096 && k>=96 && k<=384 && n>=8*m &&
        lda>=k && ldb>=n && ldc>=n) {
        int rows=(m+7)&~7;
        if (rows>m && 8*(rows-m)<=m) {
            int left_stride=(k+15)&~15, out_stride=(n+15)&~15;
            size_t left_size=(size_t)rows*left_stride;
            float *scratch=NULL;
            size_t bytes=(left_size+(size_t)rows*out_stride)*sizeof(float);
            if (!posix_memalign((void **)&scratch,64,bytes)) {
                float *result=scratch+left_size;
                for (int i=0;i<m;++i)
                    memcpy(scratch+(size_t)i*left_stride,a+(size_t)i*lda,k*sizeof(float));
                memset(scratch+(size_t)m*left_stride,0,(size_t)(rows-m)*left_stride*sizeof(float));
                RECORD(1,rows,n,k,left_stride,ldb,out_stride);
                multiply(rows,n,k,scratch,left_stride,b,ldb,result,out_stride);
                for (int i=0;i<m;++i)
                    memcpy(c+(size_t)i*ldc,result+(size_t)i*out_stride,n*sizeof(float));
                free(scratch);
                return;
            }
        }
    }
#endif
    RECORD(0,m,n,k,lda,ldb,ldc);
    multiply(m,n,k,a,lda,b,ldb,c,ldc);
}
