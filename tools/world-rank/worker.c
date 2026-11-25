#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L
#include "cob_gemm.h"
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if defined(__linux__)
#include <dirent.h>
#include <sched.h>
#include <unistd.h>
#endif
#include <dlfcn.h>
#include <inttypes.h>
#include <sys/stat.h>
#include <sys/resource.h>
#include <ctype.h>

typedef struct { int m, n, k, pad; } shape;
enum { GUARD_FLOATS = 64 };
/* A/B padding is NaN-poisoned so accidental reads fail the FP64 check.
   C padding has a distinct bit pattern; all arrays have a trailing guard. */
static const uint32_t input_guard = 0x7fc12345u;
static const uint32_t output_guard = 0xc7428000u;

static void set_bits(float* p, uint32_t bits)
{
    memcpy(p, &bits, sizeof(bits));
}

static uint32_t get_bits(const float* p)
{
    uint32_t bits;
    memcpy(&bits, p, sizeof(bits));
    return bits;
}

static void set_guards(float* p, int rows, int cols, int ld, uint32_t bits)
{
    for (int i=0; i<rows; ++i)
        for (int j=cols; j<ld; ++j) set_bits(p+(size_t)i*ld+j, bits);
    for (int j=0; j<GUARD_FLOATS; ++j) set_bits(p+(size_t)rows*ld+j, bits);
}

static size_t changed_guards(const float* p, int rows, int cols, int ld, uint32_t bits)
{
    size_t changed=0;
    for (int i=0; i<rows; ++i)
        for (int j=cols; j<ld; ++j) changed += get_bits(p+(size_t)i*ld+j)!=bits;
    for (int j=0; j<GUARD_FLOATS; ++j) changed += get_bits(p+(size_t)rows*ld+j)!=bits;
    return changed;
}

static uint32_t random_bits(uint32_t* state)
{
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

static void fill(float* out, size_t n, uint32_t* seed)
{
    for (size_t i=0; i<n; ++i)
        out[i] = ((int)((random_bits(seed) >> 8) % 2001u) - 1000) / 997.0f;
}

static float* matrix(int rows, int cols)
{
    if ((size_t)rows > (SIZE_MAX / sizeof(float) - GUARD_FLOATS) / (size_t)cols) return NULL;
    void* ptr = NULL;
    return posix_memalign(&ptr, 128, ((size_t)rows * cols + GUARD_FLOATS) * sizeof(float)) == 0 ? ptr : NULL;
}

static double seconds(void)
{
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t)) { perror("clock_gettime"); exit(1); }
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

#if defined(__linux__)
static int single_task(void)
{
    DIR* dir = opendir("/proc/self/task");
    if (!dir) { perror("/proc/self/task"); return 0; }
    int count = 0;
    struct dirent* ent;
    while ((ent = readdir(dir))) if (ent->d_name[0] >= '0' && ent->d_name[0] <= '9') ++count;
    closedir(dir);
    if (count != 1) fprintf(stderr, "Observed %d process threads, expected 1.\n", count);
    return count == 1;
}

static int pin_cpu(void)
{
    cpu_set_t allowed, selected;
    if (sched_getaffinity(0, sizeof allowed, &allowed)) { perror("sched_getaffinity"); return 0; }
    int cpu = 0;
    while (cpu < CPU_SETSIZE && !CPU_ISSET(cpu, &allowed)) ++cpu;
    if (cpu == CPU_SETSIZE) return 0;
    CPU_ZERO(&selected); CPU_SET(cpu, &selected);
    if (sched_setaffinity(0, sizeof selected, &selected)) { perror("sched_setaffinity"); return 0; }
    fprintf(stderr, "affinity_cpu=%d\n", cpu);
    return single_task();
}
#else
static int single_task(void) { return 1; }
static int pin_cpu(void) { fputs("affinity_cpu=unverified_non_linux_smoke\n", stderr); return 1; }
#endif

/* Nearly equal/opposite quadrants reproduce cancellation in a rectangular
   product. An odd final row or column receives the corresponding perturbation. */
static void cancellation_input(float* x, int rows, int cols, int ld, int is_a)
{
    const int hr=(rows+1)/2, hc=(cols+1)/2;
    for (int i=0;i<rows;++i) for (int j=0;j<cols;++j) {
        const int r=i%hr,q=j%hc;
        const float value=is_a ? (float)((r*17+q*29)%509-254)/257.0f
                               : (float)((r*31+q*13)%503-251)/263.0f;
        const float eps=is_a ? (((r+q)&1)?7.0e-4f:-9.0e-4f)
                             : (((r+2*q)&1)?8.0e-4f:-6.0e-4f);
        x[(size_t)i*ld+j]=i<hr ? (j<hc?value:-value+2.0f*eps)
                                : (j<hc?value+eps:-value+eps);
    }
}

#include "vendor.h"

extern void hybrid_sgemm_rowmajor(int,int,int,const float*,int,const float*,int,float*,int);
rank_sgemm_fn rank_bridge;
static rank_sgemm_fn selected_sgemm;
static int vendor_calls;
static void traced_sgemm(int m,int n,int k,const float* a,int lda,const float* b,int ldb,float* c,int ldc)
{
    ++vendor_calls;
    selected_sgemm(m,n,k,a,lda,b,ldb,c,ldc);
}
static void fatal(const char* why) { fprintf(stderr,"rank worker: %s\n",why); exit(1); }

static void* allocate(size_t count,size_t width)
{
    if (width && count>SIZE_MAX/width) fatal("size overflow");
    void* p=malloc(count*width);
    if (!p) fatal("allocation failed");
    return p;
}
static uint32_t number(const char* text)
{
    char* end; errno=0;
    unsigned long long x=strtoull(text,&end,10);
    if (errno || !isdigit((unsigned char)*text) || *end || x>UINT32_MAX) fatal("invalid unsigned argument");
    return (uint32_t)x;
}
static void* symbol(void* lib,const char* name)
{
    void* ptr=dlsym(lib,name);
    if (!ptr) fatal(dlerror());
    return ptr;
}
static void* plugin(const char* path)
{
    void* lib=dlopen(path,RTLD_NOW|RTLD_LOCAL);
    if (!lib) fatal(dlerror());
    int (*initialize)(void)=NULL;
    void* entry=symbol(lib,"rank_vendor_init");
    memcpy(&initialize,&entry,sizeof initialize);
    if (initialize() || !single_task()) fatal("failed single-thread initialization");
    entry=symbol(lib,"rank_vendor_sgemm");
    memcpy(&selected_sgemm,&entry,sizeof selected_sgemm);
    return lib;
}
static void make_inputs(shape s,const char* mode,uint32_t seed,float* a,float* b)
{
    const int lda=s.k+s.pad,ldb=s.n+s.pad;
    if (!strcmp(mode,"cancel")) {
        cancellation_input(a,s.m,s.k,lda,1); cancellation_input(b,s.k,s.n,ldb,0);
    } else {
        if (strcmp(mode,"random") && strcmp(mode,"scaled")) fatal("unrecognized input distribution");
        for (int i=0;i<s.m;++i) fill(a+(size_t)i*lda,s.k,&seed);
        for (int i=0;i<s.k;++i) fill(b+(size_t)i*ldb,s.n,&seed);
        if (!strcmp(mode,"scaled")) {
            for (int i=0;i<s.m;++i) for (int p=0;p<s.k;++p) a[(size_t)i*lda+p] *= p<s.k/2 ? 1.0f : 0x1p-8f;
            for (int p=0;p<s.k;++p) for (int j=0;j<s.n;++j) b[(size_t)p*ldb+j] *= p<s.k/2 ? 0x1p-8f : 1.0f;
        }
    }
    set_guards(a,s.m,s.k,lda,input_guard); set_guards(b,s.k,s.n,ldb,input_guard);
}
static void write_reference(shape s,const float* a,const float* b,const char* path,void* lib)
{
    rank_dgemm_fn dgemm=NULL;
    void* entry=symbol(lib,"rank_vendor_dgemm");
    memcpy(&dgemm,&entry,sizeof dgemm);
    FILE* output=fopen(path,"wb");
    if (!output) fatal("cannot create reference");
    const int rows=s.m<256?s.m:256;
    double* bd=allocate((size_t)s.k*s.n,sizeof *bd);
    double* ad=allocate((size_t)rows*s.k,sizeof *ad);
    double* cd=allocate((size_t)rows*s.n,sizeof *cd);
    for (int i=0;i<s.k;++i) for (int j=0;j<s.n;++j) bd[(size_t)i*s.n+j]=b[(size_t)i*(s.n+s.pad)+j];
    for (int row=0;row<s.m;row+=rows) {
        const int count=rows<s.m-row?rows:s.m-row;
        for (int i=0;i<count;++i) for (int p=0;p<s.k;++p) ad[(size_t)i*s.k+p]=a[(size_t)(row+i)*(s.k+s.pad)+p];
        dgemm(count,s.n,s.k,ad,s.k,bd,s.n,cd,s.n);
        const size_t length=(size_t)count*s.n;
        if (fwrite(cd,sizeof *cd,length,output)!=length) fatal("incomplete reference write");
    }
    if (fclose(output) || !single_task()) fatal("unverified reference");
    free(ad); free(bd); free(cd);
}
static rank_sgemm_fn choose(const char* variant,void* lib)
{
    if (!strcmp(variant,"native")) return cob_sgemm_rowmajor;
    if (!lib) fatal("selected variant requires a vendor plugin");
    if (!strcmp(variant,"direct")) return selected_sgemm;
    if (!strcmp(variant,"hybrid")) return hybrid_sgemm_rowmajor;
    fatal("unknown variant");
    return NULL;
}
static void measure(shape s,const char* variant,const char* file,const float* a,const float* b,void* lib)
{
    const int lda=s.k+s.pad,ldb=s.n+s.pad,ldc=s.n+s.pad;
    float* c=matrix(s.m,ldc);
    if (!c) fatal("output allocation");
    for (int i=0;i<s.m;++i) for (int j=0;j<s.n;++j) c[(size_t)i*ldc+j]=NAN;
    set_guards(c,s.m,s.n,ldc,output_guard);
    rank_sgemm_fn multiply=choose(variant,lib);
    rank_bridge=traced_sgemm;
    multiply(s.m,s.n,s.k,a,lda,b,ldb,c,ldc);
    rank_bridge=selected_sgemm;
    if (!single_task()) fatal("warm multiply launched multiple threads");
    FILE* ref=fopen(file,"rb");
    if (!ref) fatal("missing FP64 reference");
    double* row=allocate(s.n,sizeof *row);
    double maximum=0,min_signed=INFINITY,max_signed=-INFINITY,checksum=0;
    size_t nonfinite=0;
    for (int i=0;i<s.m;++i) {
        if (fread(row,sizeof *row,s.n,ref)!=(size_t)s.n) fatal("truncated FP64 reference");
        for (int j=0;j<s.n;++j) {
            double actual=c[(size_t)i*ldc+j],delta=actual-row[j];
            if (!isfinite(delta)) { ++nonfinite; continue; }
            checksum+=actual;
            if (fabs(delta)>maximum) maximum=fabs(delta);
            if (delta<min_signed) min_signed=delta;
            if (delta>max_signed) max_signed=delta;
        }
    }
    if (fgetc(ref)!=EOF || ferror(ref)) fatal("wrong reference length");
    fclose(ref); free(row);
    size_t guards=changed_guards(c,s.m,s.n,ldc,output_guard)+
        changed_guards(a,s.m,s.k,lda,input_guard)+changed_guards(b,s.k,s.n,ldb,input_guard);
#if defined(__linux__)
    int actual_cpu=sched_getcpu();
#else
    int actual_cpu=-1;
#endif
    printf("READY,%.12g,%zu,%zu,%zu,cpu=%d,threads=1,signed_min_error=%.12g,signed_max_error=%.12g,"
           "checksum=%.12g,hybrid_vendor_calls=%d\n",maximum,nonfinite,guards,(size_t)s.m*s.n,
           actual_cpu,min_signed,max_signed,checksum,vendor_calls);
    fflush(stdout);
    if (nonfinite || guards || maximum>0.002) fatal("FP64 or guard rejection");
    const double work=2.0*s.m*s.n*s.k;
    int calls=(int)fmin(512.0,fmax(1.0,ceil(8e7/work)));
    char request[16];
    while (fgets(request,sizeof request,stdin)) {
        if (!strcmp(request,"quit\n")) break;
        if (strcmp(request,"time\n")) fatal("unknown timing request");
        double start=seconds();
        for (int i=0;i<calls;++i) multiply(s.m,s.n,s.k,a,lda,b,ldb,c,ldc);
        double elapsed=seconds()-start;
        if (elapsed<=0 || !single_task() ||
            changed_guards(c,s.m,s.n,ldc,output_guard) ||
            changed_guards(a,s.m,s.k,lda,input_guard) ||
            changed_guards(b,s.k,s.n,ldb,input_guard)) fatal("timed call failed guard or affinity");
        printf("TIME,%.12g,%d\n",elapsed/calls,calls); fflush(stdout);
    }
    free(c);
}
int main(int argc,char** argv)
{
    int oracle=argc>1&&!strcmp(argv[1],"oracle");
    if (argc!=(oracle?10:11) || (!oracle && strcmp(argv[1],"measure"))) return 2;
    int arg=2;
    const char* variant=oracle?"reference":argv[arg++];
    const char* path=argv[arg++]; const char* reference=argv[arg++];
    uint32_t m=number(argv[arg++]),n=number(argv[arg++]),k=number(argv[arg++]);
    uint32_t pad=number(argv[arg++]),seed=number(argv[arg++]);
    if (!m || !n || !k || m>INT_MAX || n>INT_MAX || k>INT_MAX || pad>INT_MAX-n || pad>INT_MAX-k)
        fatal("unsupported shape");
    shape s={(int)m,(int)n,(int)k,(int)pad};
    if (!pin_cpu()) return 1;
    void* lib=strcmp(path,"-")?plugin(path):NULL;
    float* a=matrix(s.m,s.k+s.pad); float* b=matrix(s.k,s.n+s.pad);
    if (!a || !b) fatal("input allocation");
    make_inputs(s,argv[arg],seed,a,b);
    if (oracle) { if (!lib) return 2; write_reference(s,a,b,reference,lib); }
    else measure(s,variant,reference,a,b,lib);
    free(a); free(b);
    return 0;
}
