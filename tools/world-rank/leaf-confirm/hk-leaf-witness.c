/* Link only to the diagnostic executable. All CBLAS calls forward to the same
   installed archive or LP64 DSO used by its ordinary executable. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../vendor-public/trace.c"

#define SG_ARGS int m,int n,int k,const float *a,int lda,const float *b,int ldb,float *c,int ldc
#define SG_VARS m,n,k,a,lda,b,ldb,c,ldc
typedef void (*sgemm_fn)(SG_ARGS);
struct input {int m,n,k,lda,ldb,ldc;const float *a,*b;float *c;};
static struct input outer;
static unsigned first;
static int inside,checked_ok,finish_count,bottom_count,right_count,depth,probe_mode;
static unsigned long core_checked;
static float *fringe_work;
static sgemm_fn bottom_right;
static const char *provider;
static int reports;

#if COB_ENABLE_OPENBLAS_LEAF
extern int __real_openblas_get_num_threads(void);
int __wrap_openblas_get_num_threads(void) {
    if(inside && (probe_mode==1 || (probe_mode==2 && counts[2]-first>=5))) return 2;
    return __real_openblas_get_num_threads();
}
#elif COB_ENABLE_MKL_SEQUENTIAL_LEAF
extern int __real_MKL_Get_Max_Threads(void);
int __wrap_MKL_Get_Max_Threads(void) {
    if(inside && (probe_mode==1 || (probe_mode==2 && counts[2]-first>=5))) return 2;
    return __real_MKL_Get_Max_Threads();
}
#endif
static const char *bound_provider(void)
{
#if COB_ENABLE_OPENBLAS_LEAF
    return openblas_get_parallel()==0 && __real_openblas_get_num_threads()==1 ? openblas_get_config() : NULL;
#else
    Dl_info ci,fi;
    void *cs=dlsym(RTLD_DEFAULT,"cblas_sgemm"),*fs=dlsym(RTLD_DEFAULT,"sgemm_");
    if(!cs || !fs || !dladdr(cs,&ci) || !dladdr(fs,&fi) || ci.dli_fbase!=fi.dli_fbase || !ci.dli_fname) return NULL;
    const char *base=strrchr(ci.dli_fname,'/');base=base?base+1:ci.dli_fname;
#if COB_ENABLE_MKL_SEQUENTIAL_LEAF
    if(__real_MKL_Get_Max_Threads()!=1 || mkl_domain_get_max_threads(MKL_DOMAIN_BLAS)!=1) return NULL;
    const char *expect="libmkl_intel_lp64.so";
#else
    const char *expect="libarmpl_lp64.so";
#endif
    size_t n=strlen(expect);
    return strncmp(base,expect,n)==0 && (base[n]=='.'||base[n]==0) ? base : NULL;
#endif
}
static int tile(uintptr_t arg,uintptr_t scratch,const float *base,int rows,int cols,int ld,int blocks)
{
    if(scratch && arg==scratch) return -1;
    for(int i=0;i<blocks;++i)
        if(arg==(uintptr_t)(base+(size_t)(i/3)*rows*ld+(i%3)*cols)) return i;
    return -2;
}
static void edge(SG_ARGS)
{
    int mr=outer.m-(outer.m%2),nr=outer.n-(outer.n%3);
    checked_ok &= k==outer.k && lda==outer.lda && ldc==outer.ldc;
    if(c==outer.c+nr) {
        ++right_count;
        checked_ok &= m==mr && n==outer.n%3 && n>0 && a==outer.a && b==fringe_work && ldb==n;
        if(checked_ok)for(int p=0;p<k;++p)
            checked_ok &= !memcmp(b+(size_t)p*ldb,outer.b+(size_t)p*outer.ldb+nr,(size_t)n*sizeof(float));
    } else {
        ++bottom_count;
        checked_ok &= m==1 && outer.m%2 && n==outer.n && ldb==outer.ldb && b==outer.b &&
            a==outer.a+(size_t)mr*outer.lda && c==outer.c+(size_t)mr*outer.ldc;
    }
    bottom_right(SG_VARS);
}
static inline float madd(float a,float b,float c) {
#if defined(__FMA__) || defined(__ARM_FEATURE_FMA)
    return fmaf(a,b,c);
#else
    return c+a*b;
#endif
}
void __real_cob_hk15_ragged_finish(SG_ARGS,float*,sgemm_fn);
void __wrap_cob_hk15_ragged_finish(SG_ARGS,float *workspace,sgemm_fn base)
{
    ++finish_count;
    checked_ok &= inside && m==outer.m && n==outer.n && k==outer.k && a==outer.a && b==outer.b && c==outer.c &&
        lda==outer.lda && ldb==outer.ldb && ldc==outer.ldc && base!=NULL;
    const int mc=m-m%2,nc=n-n%3,kc=k-k%3;
    depth=k-kc;fringe_work=workspace;bottom_right=base;
    float *expected=NULL;
    if(depth) {
        expected=malloc((size_t)mc*nc*sizeof(float));
        if(!expected) abort();
        for(int i=0;i<mc;++i)for(int j=0;j<nc;++j) {
            float value=c[(size_t)i*ldc+j];
            for(int p=kc;p<k;++p) value=madd(a[(size_t)i*lda+p],b[(size_t)p*ldb+j],value);
            expected[(size_t)i*nc+j]=value;
        }
    }
    __real_cob_hk15_ragged_finish(SG_VARS,workspace,edge);
    if(expected) {
        for(int i=0;i<mc;++i) checked_ok &= !memcmp(c+(size_t)i*ldc,expected+(size_t)i*nc,(size_t)nc*sizeof(float));
        core_checked=(size_t)mc*nc;free(expected);
    }
}
void __real_cob_sgemm_rowmajor(SG_ARGS);
void __wrap_cob_sgemm_rowmajor(SG_ARGS)
{
    if(inside)abort();
    ++reports;outer=(struct input){m,n,k,lda,ldb,ldc,a,b,c};
    first=counts[2];provider=bound_provider();
    checked_ok=1;finish_count=bottom_count=right_count=depth=0;core_checked=0;fringe_work=NULL;
    const char *mode=getenv("HK_LEAF_TEST_SERIAL");probe_mode=mode?atoi(mode):0;
    inside=1;__real_cob_sgemm_rowmajor(SG_VARS);inside=0;
    unsigned total=counts[2]-first;
    int mh=m/2,nh=n/3,kh=k/3;
    int lo=mh<nh?mh:nh;lo=lo<kh?lo:kh;
    int eligible=mh>=768&&mh<=1280&&nh>=768&&nh<=1280&&kh>=768&&kh<=1280&&
        4*mh<=5*lo&&4*nh<=5*lo&&4*kh<=5*lo;
#ifdef HK_LEAF_SMALL_TEST
    eligible=mh>=4&&mh<=1280&&nh>=4&&nh<=1280&&kh>=4&&kh<=1280&&4*mh<=5*lo&&4*nh<=5*lo&&4*kh<=5*lo;
#endif
    uintptr_t x=0,y=0,product=0;
    int workspace_ok=1;
    if(eligible && total>=2 && total<=slots) {
        uint64_t *r=record[(first+1)%slots];
        x=r[6];y=r[7];product=r[8];
        workspace_ok=x%128==0 && y==x+(size_t)mh*kh*sizeof(float) && product==y+(size_t)kh*nh*sizeof(float);
        if(fringe_work)workspace_ok &= x==(uintptr_t)fringe_work;
    }
    printf("HKLEAF,%d,%s,%d,%u,%d,%d,%d,%d,%lu,%d,%d,%d,%d\n",reports,provider?provider:"unresolved",
        eligible,total,finish_count,right_count,bottom_count,depth,core_checked,checked_ok,workspace_ok,probe_mode,total<=slots);
    if(total<=slots) for(unsigned i=0;i<total;++i) {
        const uint64_t *r=record[(first+i)%slots];
        printf("HKCALL,%u,%llu,%llu,%llu,%llu,%llu,%llu,%d,%d,%d,%llu,%llu\n",i,
            (unsigned long long)r[0],(unsigned long long)r[1],(unsigned long long)r[2],
            (unsigned long long)r[3],(unsigned long long)r[4],(unsigned long long)r[5],
            tile(r[6],x,a,mh,kh,lda,6),tile(r[7],y,b,kh,nh,ldb,9),tile(r[8],product,c,mh,nh,ldc,6),
            (unsigned long long)r[9],(unsigned long long)r[10]);
    }
}
