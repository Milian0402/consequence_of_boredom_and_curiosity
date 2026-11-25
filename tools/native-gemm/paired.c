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
#include <cblas.h>

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

typedef void (*gemm)(int,int,int,const float*,int,const float*,int,float*,int);
extern void baseline_sgemm_rowmajor(int,int,int,const float*,int,const float*,int,float*,int);

static void blas_sgemm(int m,int n,int k,const float* a,int lda,const float* b,int ldb,float* c,int ldc)
{
    if (m==1) {
        cblas_sgemv(CblasRowMajor,CblasTrans,k,n,1,b,ldb,a,1,0,c,1);
    } else {
        cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,n,k,1,a,lda,b,ldb,0,c,ldc);
    }
}

static const char* impl_name(int impl,int m)
{
    if (impl==2) return m==1 ? "openblas-sgemv" : "openblas-sgemm";
    return impl==0 ? "baseline" : "candidate";
}
static const gemm implementations[] = {baseline_sgemm_rowmajor,cob_sgemm_rowmajor,blas_sgemm};

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

static int verify_full(shape s,const float* a,const float* b,float* const out[3],
                       const char* input,uint32_t seed,int block)
{
    const int rows=s.m < 256 ? s.m : 256;
    const int lda=s.k+s.pad, ldb=s.n+s.pad, ldc=s.n+s.pad;
    const size_t bcount=(size_t)s.k*s.n;
    double* bd=malloc(bcount*sizeof(*bd));
    double* ad=malloc((size_t)rows*s.k*sizeof(*ad));
    double* cd=malloc((size_t)rows*s.n*sizeof(*cd));
    if (!ad || !bd || !cd) { free(ad); free(bd); free(cd); return 0; }
    for (int i=0; i<s.k; ++i) for (int j=0; j<s.n; ++j)
        bd[(size_t)i*s.n+j]=b[(size_t)i*ldb+j];
    double max_abs[3]={0}, sums[3]={0};
    int valid[3]={1,1,1};
    for (int row=0;row<s.m;row+=rows) {
        const int mr=rows<s.m-row ? rows : s.m-row;
        for (int i=0;i<mr;++i) for (int p=0;p<s.k;++p)
            ad[(size_t)i*s.k+p]=a[(size_t)(row+i)*lda+p];
        cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,mr,s.n,s.k,1,ad,s.k,bd,s.n,0,cd,s.n);
        for (int i=0;i<mr;++i) for (int j=0;j<s.n;++j) {
            for (int impl=0;impl<3;++impl) {
                const float value=out[impl][(size_t)(row+i)*ldc+j];
                const double diff=fabs((double)value-cd[(size_t)i*s.n+j]);
                if (!isfinite(diff)) valid[impl]=0;
                max_abs[impl]=fmax(max_abs[impl],diff);
                sums[impl]+=value;
            }
        }
    }
    free(ad); free(bd); free(cd);
    int ok=1;
    for (int impl=0;impl<3;++impl) {
        const size_t changed=changed_guards(out[impl], s.m,s.n,ldc,output_guard)
            +changed_guards(a,s.m,s.k,lda,input_guard)+changed_guards(b,s.k,s.n,ldb,input_guard);
        valid[impl]=valid[impl] && max_abs[impl]<=0.002 && !changed;
        fprintf(stderr,"accuracy,%s,%u,%d,%s,%d,%d,%d,%d,%zu,%.9g,%.12g,%zu,%s\n",
            input,seed,block,impl_name(impl,s.m),s.m,s.n,s.k,s.pad,(size_t)s.m*s.n,max_abs[impl],sums[impl],changed,
            valid[impl]?"pass":"fail");
        if (!valid[impl]) ok=0;
    }
    return ok;
}

static int measure(shape s,const char* mode,const char* phase,int block,int pairs)
{
    const int cancel=block==0, scaled=block==-1;
    const int calls=1;
    const int lda=s.k+s.pad, ldb=s.n+s.pad, ldc=s.n+s.pad;
    const uint32_t seed=block==1 ? 0x12345678u : block==2 ? 0x649f539bu : scaled ? 117u : 0;
    uint32_t state=seed;
    float* a=matrix(s.m,lda),*b=matrix(s.k,ldb);
    float* out[]={matrix(s.m,ldc),matrix(s.m,ldc),matrix(s.m,ldc)};
    int ok=a && b && out[0] && out[1] && out[2];
    if (!ok) { perror("FP32 allocation"); goto cleanup; }
    if (cancel) {
        cancellation_input(a,s.m,s.k,lda,1); cancellation_input(b,s.k,s.n,ldb,0);
    } else {
        for (int i=0;i<s.m;++i) fill(a+(size_t)i*lda,s.k,&state);
        for (int i=0;i<s.k;++i) fill(b+(size_t)i*ldb,s.n,&state);
        if (scaled) {
            /* Starting from A0,B0 = draws/16, apply opposite power-of-two
               factors in the two K halves. Exact A0*B0 is unchanged. */
            for (int i=0;i<s.m;++i) for (int k=0;k<s.k;++k)
                a[(size_t)i*lda+k] *= k < s.k/2 ? 1.0f : 1.0f/256.0f;
            for (int k=0;k<s.k;++k) for (int j=0;j<s.n;++j)
                b[(size_t)k*ldb+j] *= k < s.k/2 ? 1.0f/256.0f : 1.0f;
        }
    }
    set_guards(a,s.m,s.k,lda,input_guard); set_guards(b,s.k,s.n,ldb,input_guard);
    for (int impl=0;impl<3;++impl) {
        for (int i=0;i<s.m;++i) for (int j=0;j<s.n;++j) out[impl][(size_t)i*ldc+j]=NAN;
        set_guards(out[impl],s.m,s.n,ldc,output_guard);
        implementations[impl](s.m,s.n,s.k,a,lda,b,ldb,out[impl],ldc);
    }
    ok=verify_full(s,a,b,out,scaled?"scaled":cancel?"cancel":"random",seed,scaled?0:block) && single_task();
    if (cancel || scaled || !ok) goto cleanup;
    for (int c=0;c<2 && ok;++c) {
        const int competitor=(block==1 ? c : 1-c)*2;
        for (int pair=0;pair<pairs && ok;++pair) {
            const int first_candidate=(pair+block-1)&1;
            double times[3]={0};
            for (int slot=0;slot<2;++slot) {
                const int impl=(slot==0 ? first_candidate : !first_candidate) ? 1 : competitor;
                const double start=seconds();
                for (int call=0;call<calls;++call)
                    implementations[impl](s.m,s.n,s.k,a,lda,b,ldb,out[impl],ldc);
                times[impl]=(seconds()-start)/calls;
                if (!(times[impl]>0) || !single_task() || changed_guards(out[impl],s.m,s.n,ldc,output_guard) ||
                    changed_guards(a,s.m,s.k,lda,input_guard) || changed_guards(b,s.k,s.n,ldb,input_guard)) { ok=0; break; }
            }
            if (ok) printf("%s,%s,%d,%d,%d,%d,%d,%u,%d,%d,%s,%s,%.9g,%.9g,%.9g\n",
                phase,mode,s.m,s.n,s.k,s.pad,block,seed,pair+1,calls,impl_name(competitor,s.m),
                first_candidate?"candidate":impl_name(competitor,s.m),times[competitor],times[1],
                times[competitor]/times[1]);
        }
    }
    fflush(stdout);
cleanup:
    free(a); free(b);
    for (int i=0;i<3;++i) free(out[i]);
    return ok;
}

int main(int argc,char** argv)
{
    if (argc!=4 || (strcmp(argv[2],"discovery") && strcmp(argv[2],"confirm") && strcmp(argv[2],"recursive") && strcmp(argv[2],"bilinear")) ||
        (strcmp(argv[3],"1") && strcmp(argv[3],"2") && strcmp(argv[3],"cancel"))) return 2;
    const int block=!strcmp(argv[3],"cancel") ? 0 : atoi(argv[3]);
    const int discovery=!strcmp(argv[2],"discovery");
    const int recursive=!strcmp(argv[2],"recursive"), bilinear=!strcmp(argv[2],"bilinear");
    openblas_set_num_threads(1);
    if (openblas_get_num_threads()!=1 || !pin_cpu()) return 1;
    fprintf(stderr,"block=%d phase=%s mode=%s openblas_threads=1 configuration=%s core=%s\n",
            block,argv[2],argv[1],openblas_get_config(),openblas_get_corename());
    const shape tune[]={{2048,4096,2048,0},{3200,1920,2432,0},{1985,2753,2017,0},{3072,3072,3072,0},
                        {4096,4096,4096,0},{2304,4608,2304,0},{3,2048,1536,0},{6,3072,2048,0}};
    const shape rank40[]={{2304,4608,2304,0},{2048,4096,2048,0}};
    const shape recursion[]={{3072,3072,3072,0},{4096,4096,4096,0}};
    const shape holdout[]={{2240,4480,2240,0},{3840,2304,2944,0},{2560,2560,5120,0},
                           {2689,3329,2497,0},{3073,3073,3073,0},{1536,1536,1536,0},
                           {320,960,768,0},{2688,5376,2688,0},{2240,4480,2240,7},{3584,3584,3584,0},{2,1537,2049,0},
                           {5,4095,1025,0},{8,1024,3072,0},{16,1536,1024,0}};
    const shape cancellations[]={{2048,2048,2048,0},{1793,2049,1537,7}};
    const shape* cases=block==0 ? cancellations : bilinear ? rank40 : recursive ? recursion : discovery ? tune : holdout;
    const int n=block==0 ? (int)(sizeof cancellations/sizeof *cancellations)
        : bilinear ? (int)(sizeof rank40/sizeof *rank40)
        : recursive ? (int)(sizeof recursion/sizeof *recursion)
        : discovery ? (int)(sizeof tune/sizeof *tune) : (int)(sizeof holdout/sizeof *holdout);
    int ok=1;
    for (int j=0;j<n;++j)
        if (!measure(cases[block==2 ? n-1-j : j],argv[1],argv[2],block,(discovery || recursive || bilinear) ? 1 : 3)) ok=0;
    if (block==0) {
        if (!measure((shape){1536,2688,1664,0},argv[1],argv[2],-1,0)) ok=0;
        if (recursive && !measure(recursion[0],argv[1],argv[2],-1,0)) ok=0;
        if (bilinear && !measure(rank40[0],argv[1],argv[2],-1,0)) ok=0;
    }
    return ok ? 0 : 1;
}
