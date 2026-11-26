#include <dlfcn.h>
#include <stdint.h>
#include <stdlib.h>
#include <pthread.h>
typedef void (*gemm)(int,int,int,const float*,int,const float*,int,float*,int);
static gemm large,other;
static pthread_once_t once=PTHREAD_ONCE_INIT;
static _Thread_local int used_large;
static gemm bind(const char*env) {
    const char*path=getenv(env);if(!path)return NULL;
    void*h=dlopen(path,RTLD_NOW|RTLD_LOCAL);if(!h)return NULL;
    int(*init)(void)=dlsym(h,"rank_vendor_init");gemm fn=(gemm)dlsym(h,"rank_vendor_sgemm");
    if(!fn||!init||init())return NULL;
    return fn;
}
static void setup(void){large=bind("EXACT_LARGE_FP32");other=bind("EXACT_OTHER_FP32");}
int rank_vendor_init(void) {return pthread_once(&once,setup)!=0 || !large || !other;}
int exact_policy_last_large(void){return used_large;}
void rank_vendor_sgemm(int m,int n,int k,const float*a,int la,const float*b,int lb,float*c,int lc) {
    if(rank_vendor_init())abort();
    int min=m<n?m:n,max=m>n?m:n;
    if(k<min)min=k;if(k>max)max=k;
    used_large=min>=4096 && (int64_t)max<=2*(int64_t)min;
    (used_large?large:other)(m,n,k,a,la,b,lb,c,lc);
}
