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

typedef struct { int m, n, k; } shape;

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
    if ((size_t)rows > SIZE_MAX / sizeof(float) / (size_t)cols) return NULL;
    void* ptr = NULL;
    return posix_memalign(&ptr, 128, (size_t)rows * cols * sizeof(float)) == 0 ? ptr : NULL;
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
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,n,k,1,a,lda,b,ldb,0,c,ldc);
}

static int finite_fp64_check(shape s, const float* a, const float* b, float** out)
{
    size_t ac=(size_t)s.m*s.k, bc=(size_t)s.k*s.n, cc=(size_t)s.m*s.n;
    double* ad=malloc(ac*sizeof(double)), *bd=malloc(bc*sizeof(double)), *cd=malloc(cc*sizeof(double));
    if (!ad || !bd || !cd) { free(ad); free(bd); free(cd); perror("FP64 matrix"); return 0; }
    for (size_t i=0;i<ac;++i) ad[i]=a[i];
    for (size_t i=0;i<bc;++i) bd[i]=b[i];
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,s.m,s.n,s.k,1,ad,s.k,bd,s.n,0,cd,s.n);
    free(ad); free(bd);
    static const char* names[]={"baseline","candidate","openblas"};
    int ok=1;
    for (int impl=0;impl<3;++impl) {
        double max_abs=0, checksum=0;
        for (size_t i=0;i<cc;++i) {
            double diff=fabs(out[impl][i]-cd[i]);
            if (!isfinite(diff)) { ok=0; max_abs=INFINITY; break; }
            max_abs=fmax(max_abs,diff); checksum+=out[impl][i];
        }
        fprintf(stderr,"accuracy_full,%s,%d,%d,%d,%zu,%.9g,%.12g\n",names[impl],s.m,s.n,s.k,cc,max_abs,checksum);
        if (max_abs>0.002) ok=0;
    }
    free(cd); return ok;
}

/* Check all FP32 results for finiteness and compare the two outputs in
   full. Independently recompute 256 stratified output cells in FP64. */
static int sampled_fp64_check(shape s,const float* a,const float* b,float** out)
{
    const size_t count=(size_t)s.m*s.n;
    enum { checks=256 };
    double sum[3]={0}, max_abs[3]={0}, max_diff=0;
    int ok=1;
    for (size_t cell=0;cell<count;++cell) {
        for (int impl=1;impl<=2;++impl) {
            if (!isfinite(out[impl][cell])) ok=0;
            sum[impl]+=out[impl][cell];
        }
        max_diff=fmax(max_diff,fabs((double)out[1][cell]-out[2][cell]));
    }
    uint32_t sample_seed=0x50cb7c1u;
    for (size_t j=0;j<checks;++j) {
        const size_t lo=count*j/checks, hi=count*(j+1)/checks;
        size_t cell=lo + random_bits(&sample_seed)%(hi-lo);
        if (j==0) cell=0;
        if (j+1==checks) cell=count-1;
        const float* row=a+(cell/(size_t)s.n)*s.k;
        const float* col=b+(cell%(size_t)s.n);
        double dot=0;
        for (int k=0;k<s.k;++k) dot+=(double)row[k]*col[(size_t)k*s.n];
        for (int impl=1;impl<=2;++impl) max_abs[impl]=fmax(max_abs[impl],fabs(out[impl][cell]-dot));
    }
    for (int impl=1;impl<=2;++impl) {
        fprintf(stderr,"accuracy_sample,%s,%d,%d,%d,%d,%.9g,%.12g,finite_scan=%zu,full_fp32_max_diff=%.9g\n",
            impl==1?"candidate":"openblas",s.m,s.n,s.k,checks,max_abs[impl],sum[impl],count,max_diff);
        if (max_abs[impl]>0.002) ok=0;
    }
    return ok && max_diff<=0.002;
}

static int measure(shape s, const char* mode, int block, int calls, int large, int accuracy_only)
{
    static const gemm implementations[]={baseline_sgemm_rowmajor,cob_sgemm_rowmajor,blas_sgemm};
    static const char* names[]={"baseline","candidate","openblas"};
    static const int competitors[]={0,2};
    size_t ac=(size_t)s.m*s.k, bc=(size_t)s.k*s.n, cc=(size_t)s.m*s.n;
    uint32_t seed=0x12345678u;
    float* a=matrix(s.m,s.k), *b=matrix(s.k,s.n);
    float* out[]={large ? NULL : matrix(s.m,s.n),matrix(s.m,s.n),matrix(s.m,s.n)};
    int ok=a && b && (large || out[0]) && out[1] && out[2];
    if (!ok) { perror("FP32 matrix"); goto cleanup; }
    fill(a,ac,&seed); fill(b,bc,&seed);
    for (int impl=large ? 1 : 0;impl<3;++impl) {
        for (size_t i=0;i<cc;++i) out[impl][i]=NAN;
        implementations[impl](s.m,s.n,s.k,a,s.k,b,s.n,out[impl],s.n);
    }
    ok=(large ? sampled_fp64_check(s,a,b,out) : finite_fp64_check(s,a,b,out)) && single_task();
    if (accuracy_only) goto cleanup;
    for (int c=0;c<(large ? 1 : 2) && ok;++c) {
        /* Reverse competitor ordering and which implementation goes first in block two. */
        int competitor=large ? 2 : competitors[block==1 ? c : 1-c];
        for (int pair=0; pair<(large ? 3 : 5) && ok; ++pair) {
            double times[3]={0};
            int candidate_first=(pair+block-1)&1;
            for (int slot=0;slot<2;++slot) {
                int impl=(slot==0 ? candidate_first : !candidate_first) ? 1 : competitor;
                double start=seconds();
                for (int i=0;i<calls;++i)
                    implementations[impl](s.m,s.n,s.k,a,s.k,b,s.n,out[impl],s.n);
                times[impl]=(seconds()-start)/calls;
                if (!(times[impl]>0) || !single_task()) { ok=0; break; }
            }
            if (ok) printf("%s,%d,%d,%d,%d,%d,%d,%s,%s,%.9g,%.9g,%.9g\n",
                mode,s.m,s.n,s.k,block,pair+1,calls,names[competitor],
                candidate_first ? "candidate" : names[competitor],
                times[competitor],times[1],times[competitor]/times[1]);
        }
    }
    fflush(stdout);
cleanup:
    free(a);free(b);for(int i=0;i<3;++i)free(out[i]);return ok;
}

int main(int argc,char** argv)
{
    if(argc!=3 || (strcmp(argv[2],"1") && strcmp(argv[2],"2"))) return 2;
    int block=atoi(argv[2]);
    openblas_set_num_threads(1);
    if(openblas_get_num_threads()!=1 || !pin_cpu()) return 1;
    fprintf(stderr,"block=%d mode=%s openblas_threads=1 config=%s core=%s seed=0x12345678\n",
        block,argv[1],openblas_get_config(),openblas_get_corename());

    if (!strcmp(argv[1],"forced-avx2-smoke")) {
        /* One full SIMD tile, ragged output edges, and more than one K panel. */
        return measure((shape){17,37,273},argv[1],block,1,0,1) ? 0 : 1;
    }
    const shape medium[]={{192,192,192},{383,385,389},{768,768,768},{256,1024,640},
                          {1024,192,768},{7,11,13},{1,1536,1024},{1,2049,1025},{256,256,256}};
    const shape large[]={{1024,1024,1024},{1536,1536,1536},{2048,2048,2048},
                         {2049,2112,1984},{4096,4096,4096}};
    const int nmed=(int)(sizeof(medium)/sizeof(medium[0]));
    const int nlarge=!strcmp(argv[1],"plain-c-autovec") ? 0 : (int)(sizeof(large)/sizeof(large[0]));
    for (int j=0;j<nmed+nlarge;++j) {
        const int i=block==1 ? j : nmed+nlarge-1-j;
        const int is_large=i>=nmed;
        const shape s=is_large ? large[i-nmed] : medium[i];
        const int calls=s.m==7 ? 2000 : s.m==1 ? 8 : 1;
        if (!measure(s,argv[1],block,calls,is_large,0)) return 1;
    }
    return 0;
}
