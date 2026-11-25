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
        fprintf(stderr,"accuracy,%s,%d,%d,%d,%zu,%.9g,%.12g\n",names[impl],s.m,s.n,s.k,cc,max_abs,checksum);
        if (max_abs>0.002) ok=0;
    }
    free(cd); return ok;
}

static int measure(shape s, const char* mode, int block, int calls)
{
    static const gemm implementations[]={baseline_sgemm_rowmajor,cob_sgemm_rowmajor,blas_sgemm};
    static const char* names[]={"baseline","candidate","openblas"};
    static const int competitors[]={0,2};
    size_t ac=(size_t)s.m*s.k, bc=(size_t)s.k*s.n, cc=(size_t)s.m*s.n;
    uint32_t seed=0x12345678u;
    float* a=matrix(s.m,s.k), *b=matrix(s.k,s.n);
    float* out[]={matrix(s.m,s.n),matrix(s.m,s.n),matrix(s.m,s.n)};
    int ok=a && b && out[0] && out[1] && out[2];
    if (!ok) { perror("FP32 matrix"); goto cleanup; }
    fill(a,ac,&seed); fill(b,bc,&seed);
    for (int impl=0;impl<3;++impl) {
        for (size_t i=0;i<cc;++i) out[impl][i]=NAN;
        implementations[impl](s.m,s.n,s.k,a,s.k,b,s.n,out[impl],s.n);
    }
    ok=finite_fp64_check(s,a,b,out) && single_task();
    for (int c=0;c<2 && ok;++c) {
        /* Reverse competitor ordering and which implementation goes first in block two. */
        int competitor=competitors[block==1 ? c : 1-c];
        for (int pair=0; pair<5 && ok; ++pair) {
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
    /* Holdouts fixed during the portable experiment; the final two are controls. */
    const shape shapes[]={{192,192,192},{383,385,389},{768,768,768},{256,1024,640},
                          {1024,192,768},{7,11,13},{1,1536,1024}};
    const int count=(int)(sizeof(shapes)/sizeof(shapes[0]));
    for (int j=0;j<count;++j) {
        int i=block==1 ? j : count-1-j;
        int calls=i==5 ? 2000 : i==6 ? 8 : 1;
        if (!measure(shapes[i],argv[1],block,calls)) return 1;
    }
    return 0;
}
