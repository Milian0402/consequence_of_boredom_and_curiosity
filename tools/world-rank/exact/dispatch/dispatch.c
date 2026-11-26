/* Content-checked, single-product BF16 experiment for an overwrite row-major SGEMM. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <float.h>
#include <limits.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

_Static_assert(sizeof(float) == 4 && FLT_RADIX == 2 && FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128,
               "requires IEEE binary32");

typedef void (*gemm_fn)(int,int,int,const float*,int,const float*,int,float*,int);
typedef int (*integer_fn)(void);
typedef int (*plan_fn)(int);
typedef const char *(*impl_fn)(int);
struct provider { void *handle; gemm_fn gemm; };
static struct provider fallback, fast;
static integer_fn fast_route, fast_products, fast_plans;
static plan_fn fast_native;
static impl_fn fast_impl;
static pthread_once_t initialization = PTHREAD_ONCE_INIT;
static int ready;
static const char *unavailable = "native-disabled";
struct observation {
    int route, amax, bmax, fast_calls, fallback_calls, products, native;
    unsigned long long inspected;
    const char *reason, *impl;
};
static _Thread_local struct observation last;

static int same_object(const char *path, const void *symbol) {
    Dl_info found;
    struct stat expected, actual;
    return symbol && dladdr(symbol,&found) && found.dli_fname &&
           !stat(path,&expected) && !stat(found.dli_fname,&actual) &&
           expected.st_dev == actual.st_dev && expected.st_ino == actual.st_ino;
}

/* Resolve both public entry points to the selected object, not a transitive BLAS. */
static int bind_provider(struct provider *provider,const char *path) {
    if (!path || !*path || same_object(path,(const void*)bind_provider)) return 0;
    void *handle=dlopen(path,RTLD_NOW|RTLD_LOCAL);
    if (!handle) return 0;
    integer_fn init=(integer_fn)dlsym(handle,"rank_vendor_init");
    gemm_fn gemm=(gemm_fn)dlsym(handle,"rank_vendor_sgemm");
    if (!same_object(path,(void*)init) || !same_object(path,(void*)gemm) || init()!=0) {
        dlclose(handle);
        return 0;
    }
    provider->handle=handle;
    provider->gemm=gemm;
    return 1;
}

static void initialize(void) {
    if (!bind_provider(&fallback,getenv("EXACT_FALLBACK_PLUGIN"))) return;
    const char *flag=getenv("EXACT_NATIVE_ENABLED");
    if (flag && strcmp(flag,"0") && strcmp(flag,"1")) return;
    ready=1;
    if (!flag || strcmp(flag,"1")) return;
    unavailable="fast-plugin-unavailable";
    if (!bind_provider(&fast,getenv("EXACT_FAST_PLUGIN"))) return;
    fast_route=(integer_fn)dlsym(fast.handle,"precision_last_route");
    fast_products=(integer_fn)dlsym(fast.handle,"precision_last_products");
    fast_plans=(integer_fn)dlsym(fast.handle,"precision_last_plan_count");
    fast_native=(plan_fn)dlsym(fast.handle,"precision_last_plan_native");
    fast_impl=(impl_fn)dlsym(fast.handle,"precision_last_plan_impl");
    if (fast.gemm==fallback.gemm || !fast_route || !fast_products || !fast_plans || !fast_native || !fast_impl) {
        unavailable="fast-native-unverified";
        fast.gemm=NULL;
    }
}

int rank_vendor_init(void) {
    return pthread_once(&initialization,initialize)==0 && ready ? 0 : 1;
}

/* The encoding of nonnegative finite binary32 is monotone as an unsigned word.
   A number in [1,256] is integral iff its low 150-biased_exponent bits are zero.
   Masking the shift count defines the operation for every possible word, including NaNs. */
static int maximum_integer(const float *matrix,int rows,int cols,int ld,unsigned long long *inspected) {
    uint32_t largest=0;
    for (int i=0;i<rows;++i) {
        const float *row=matrix+(size_t)i*(size_t)ld;
        for (int start=0;start<cols;) {
            const int count=cols-start < 64 ? cols-start : 64;
            uint32_t chunk_max=0, invalid=0;
            for (int j=0;j<count;++j) {
                uint32_t bits;
                memcpy(&bits,row+start+j,sizeof(bits));
                uint32_t mag=bits & UINT32_C(0x7fffffff);
                uint32_t fraction_mask=(UINT32_C(1)<<((150u-(mag>>23))&31u))-1u;
                invalid|=(mag!=0) & ((mag<UINT32_C(0x3f800000)) | (mag>UINT32_C(0x43800000)) | ((mag&fraction_mask)!=0));
                if (mag>chunk_max) chunk_max=mag;
            }
            start+=count;
            if (invalid) {
                *inspected+=(unsigned long long)i*cols+start;
                return -1;
            }
            if (chunk_max>largest) largest=chunk_max;
        }
    }
    *inspected+=(unsigned long long)rows*cols;
    float maximum;
    memcpy(&maximum,&largest,sizeof(maximum));
    return (int)maximum;
}

void rank_vendor_sgemm(int m,int n,int k,const float *a,int lda,const float *b,int ldb,float *c,int ldc) {
    if (rank_vendor_init()!=0) { fputs("exact dispatch requires a valid ordinary FP32 provider\n",stderr); abort(); }
    last=(struct observation){.route=2,.amax=-1,.bmax=-1,.reason="empty-or-invalid-view",.impl=""};
    if (!fast.gemm) {
        last.route=3; last.reason=unavailable;
    } else if (m>0 && n>0 && k>0 && a && b && c && lda>=k && ldb>=n && ldc>=n) {
        if (k>4096) {
            last.reason="depth-outside";
        } else {
            last.amax=maximum_integer(a,m,k,lda,&last.inspected);
            if (last.amax>=0) last.bmax=maximum_integer(b,k,n,ldb,&last.inspected);
            if (last.amax<0 || last.bmax<0) {
                last.reason="integer-domain";
            } else if ((uint64_t)k*(unsigned)last.amax*(unsigned)last.bmax > (UINT64_C(1)<<24)) {
                last.reason="accumulator-bound";
            } else {
                last.fast_calls=1;
                fast.gemm(m,n,k,a,lda,b,ldb,c,ldc);
                last.products=fast_products();
                if (fast_route()==1 && last.products==1 && fast_plans()==1) {
                    last.native=fast_native(0);
                    last.impl=fast_impl(0);
                    if (last.native==1 && last.impl && *last.impl) {
                        last.route=1; last.reason="native-exact";
                        return;
                    }
                }
                last.route=3; last.reason="fast-native-unverified";
            }
        }
    }
    last.fallback_calls=1;
    fallback.gemm(m,n,k,a,lda,b,ldb,c,ldc);
}

int exact_last_route(void) { return last.route; }
const char *exact_last_reason(void) { return last.reason ? last.reason : "no-call"; }
unsigned long long exact_last_inspected(void) { return last.inspected; }
int exact_last_amax(void) { return last.amax; }
int exact_last_bmax(void) { return last.bmax; }
int exact_last_fast_calls(void) { return last.fast_calls; }
int exact_last_fallback_calls(void) { return last.fallback_calls; }
int exact_last_products(void) { return last.products; }
int exact_last_native(void) { return last.native; }
const char *exact_last_impl(void) { return last.impl ? last.impl : ""; }
