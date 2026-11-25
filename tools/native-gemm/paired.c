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

/* Make corresponding input quadrants nearly equal or opposite, as in the
   existing COB cancellation fixture. The 2048 case is square and even. */
static void cancellation_input(float* x,int n,int is_a)
{
    const int h=n/2;
    for (int i=0;i<n;++i) for (int j=0;j<n;++j) {
        const int r=i%h,q=j%h;
        const float value=is_a ? (float)((r*17+q*29)%509-254)/257.0f
                               : (float)((r*31+q*13)%503-251)/263.0f;
        const float eps=is_a ? (((r+q)&1)?7.0e-4f:-9.0e-4f)
                             : (((r+2*q)&1)?8.0e-4f:-6.0e-4f);
        x[(size_t)i*n+j]=i<h ? (j<h?value:-value+2.0f*eps)
                             : (j<h?value+eps:-value+eps);
    }
}

static int verify_full(shape s,const float* a,const float* b,float* const out[3],
                       const char* input,uint32_t seed,int block)
{
    const int rows=s.m < 256 ? s.m : 256;
    const size_t bcount=(size_t)s.k*s.n;
    double* bd=malloc(bcount*sizeof(*bd));
    double* ad=malloc((size_t)rows*s.k*sizeof(*ad));
    double* cd=malloc((size_t)rows*s.n*sizeof(*cd));
    if (!ad || !bd || !cd) { free(ad); free(bd); free(cd); return 0; }
    for (size_t i=0;i<bcount;++i) bd[i]=b[i];
    double max_abs[3]={0}, sums[3]={0};
    int valid[3]={1,1,1};
    for (int row=0;row<s.m;row+=rows) {
        const int mr=rows<s.m-row ? rows : s.m-row;
        for (size_t i=0;i<(size_t)mr*s.k;++i) ad[i]=a[(size_t)row*s.k+i];
        cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,mr,s.n,s.k,1,ad,s.k,bd,s.n,0,cd,s.n);
        for (size_t i=0;i<(size_t)mr*s.n;++i) {
            for (int impl=0;impl<3;++impl) {
                const float value=out[impl][(size_t)row*s.n+i];
                const double diff=fabs((double)value-cd[i]);
                if (!isfinite(diff)) valid[impl]=0;
                max_abs[impl]=fmax(max_abs[impl],diff);
                sums[impl]+=value;
            }
        }
    }
    free(ad); free(bd); free(cd);
    int ok=1;
    for (int impl=0;impl<3;++impl) {
        valid[impl]=valid[impl] && max_abs[impl]<=0.002;
        fprintf(stderr,"accuracy,%s,%u,%d,%s,%d,%d,%d,%zu,%.9g,%.12g,%s\n",
            input,seed,block,impl_name(impl,s.m),s.m,s.n,s.k,(size_t)s.m*s.n,max_abs[impl],sums[impl],
            valid[impl]?"pass":"fail");
        if (!valid[impl]) ok=0;
    }
    return ok;
}

static int measure(shape s,const char* mode,int block)
{
    const int cancel=block==0;
    const int calls=s.m==1 ? 8 : 1;
    const size_t ac=(size_t)s.m*s.k, bc=(size_t)s.k*s.n, cc=(size_t)s.m*s.n;
    const uint32_t seed=block==1 ? 0x12345678u : block==2 ? 0x649f539bu : 0;
    uint32_t state=seed;
    float* a=matrix(s.m,s.k),*b=matrix(s.k,s.n);
    float* out[]={matrix(s.m,s.n),matrix(s.m,s.n),matrix(s.m,s.n)};
    int ok=a && b && out[0] && out[1] && out[2];
    if (!ok) { perror("FP32 allocation"); goto cleanup; }
    if (cancel) {
        cancellation_input(a,s.m,1); cancellation_input(b,s.m,0);
    } else {
        fill(a,ac,&state); fill(b,bc,&state);
    }
    for (int impl=0;impl<3;++impl) {
        for (size_t i=0;i<cc;++i) out[impl][i]=NAN;
        implementations[impl](s.m,s.n,s.k,a,s.k,b,s.n,out[impl],s.n);
    }
    ok=verify_full(s,a,b,out,cancel?"cancel":"random",seed,block) && single_task();
    if (cancel || !ok) goto cleanup;
    for (int c=0;c<2 && ok;++c) {
        const int competitor=(block==1 ? c : 1-c)*2;
        for (int pair=0;pair<3 && ok;++pair) {
            const int first_candidate=(pair+block-1)&1;
            double times[3]={0};
            for (int slot=0;slot<2;++slot) {
                const int impl=(slot==0 ? first_candidate : !first_candidate) ? 1 : competitor;
                const double start=seconds();
                for (int call=0;call<calls;++call)
                    implementations[impl](s.m,s.n,s.k,a,s.k,b,s.n,out[impl],s.n);
                times[impl]=(seconds()-start)/calls;
                if (!(times[impl]>0) || !single_task()) { ok=0; break; }
            }
            if (ok) printf("%s,%d,%d,%d,%d,%u,%d,%d,%s,%s,%.9g,%.9g,%.9g\n",
                mode,s.m,s.n,s.k,block,seed,pair+1,calls,impl_name(competitor,s.m),
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
    if (argc!=3 || (strcmp(argv[2],"1") && strcmp(argv[2],"2") && strcmp(argv[2],"cancel"))) return 2;
    const int block=!strcmp(argv[2],"cancel") ? 0 : atoi(argv[2]);
    openblas_set_num_threads(1);
    if (openblas_get_num_threads()!=1 || !pin_cpu()) return 1;
    fprintf(stderr,"block=%d mode=%s openblas_threads=1 configuration=%s core=%s\n",
            block,argv[1],openblas_get_config(),openblas_get_corename());
    if (block==0) return measure((shape){2048,2048,2048},argv[1],0) ? 0 : 1;
    const shape shapes[]={{1536,1536,1536},{2048,2048,2048},{3072,3072,3072},{4096,4096,4096},
                          {3072,3328,2944},{2049,2112,1984},{1024,1024,1024},
                          {1,1536,1024},{1,2049,1025}};
    const int n=(int)(sizeof shapes/sizeof shapes[0]);
    int ok=1;
    for (int j=0;j<n;++j) if (!measure(shapes[block==1 ? j : n-1-j],argv[1],block)) ok=0;
    return ok ? 0 : 1;
}
