#define _POSIX_C_SOURCE 200809L
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

typedef void (*multiply)(int, int, int, const float *, int, const float *, int, float *, int);
enum fixture { CENTERED, POSITIVE, ROWCOL, HALF_BIASED, STRONG_BIAS, ONE_SIDED, TILTED };
static const char *name[] = {"centered","positive","row-col-signs","partial-bias","biased-signed","one-sided","depth-signs"};
static uint32_t step(uint32_t *s) {
    *s ^= *s << 13;
    *s ^= *s >> 17;
    *s ^= *s << 5;
    return *s;
}
static uint32_t mix(uint32_t s) {
    s ^= s >> 16;
    s *= 0x7feb352dU;
    s ^= s >> 15;
    s *= 0x846ca68bU;
    return s ^ (s >> 16);
}
static float value(enum fixture f, int right, int i, int j, uint32_t seed, uint32_t *state) {
    const uint32_t u = step(state);
    float x = (float)(u >> 9) * 0x1p-23f;
    int out = right ? j : i, depth = right ? i : j;
    uint32_t tag = mix(seed ^ (right ? 77383u : 23917u) ^ (unsigned)out);
    float sign = tag & 1 ? -1.f : 1.f;
    if (f == CENTERED) return x - .5f;
    if (f == POSITIVE) return x;
    if (f == ROWCOL) return sign*x;
    if (f == HALF_BIASED) return (tag & 2) ? sign*x : x - .5f;
    if (f == STRONG_BIAS) return sign*(x - 0.09375f);
    if (f == ONE_SIDED) return right ? x - .5f : sign*x;
    return (mix(seed ^ (unsigned)depth) & 1 ? x : -x) * sign;

}
static void *array(size_t cells, size_t width) {
    void *p = calloc(cells, width);
    if (!p) {
        perror("calloc");
        exit(2);
    }
    return p;
}
static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec*1e-9;
}
int main(int argc, char **argv) {
    if(argc!=6) return 1;
    const int m=atoi(argv[1]), n=atoi(argv[2]), k=atoi(argv[3]);
    const uint32_t seed=(uint32_t)strtoul(argv[4], NULL,10);
    const int lda=k+7,ldb=n+11,ldc=n+13;
    float* a=(float*)array((size_t)m*lda+3,4)+3;
    float* b=(float*)array((size_t)k*ldb+5,4)+5;
    float* c=(float*)array((size_t)m*ldc+7,4)+7;
    const char* tags[]={"coherence","old","classical","accelerate"};
    multiply fs[4];
    for(int u=0;u<4;u++) {
        char path[4096];
        if(snprintf(path,sizeof path,"%s/%s.dylib",argv[5],tags[u]) >= (int)sizeof path) return 2;
        void* lib=dlopen(path,RTLD_NOW|RTLD_LOCAL);
        if(!lib || !(fs[u]=(multiply)dlsym(lib,"cob_sgemm_rowmajor"))) return 3;
    }
    const int order[][4]={{0,1,3,2},{1,2,0,3},{2,3,1,0},{3,0,2,1}};
    puts("m,n,k,seed,fixture,block,position,method,seconds,spot_max_abs");
    for(int f=0;f<=TILTED;f++){
        uint32_t gen=seed ^ ((uint32_t)f+7)*234749u;
        for(int i=0;i<m;i++) for(int l=0;l<k;l++) a[(size_t)i*lda+l]=value(f,0,i,l,seed,&gen);
        for(int l=0;l<k;l++) for(int j=0;j<n;j++) b[(size_t)l*ldb+j]=value(f,1,l,j,seed,&gen);
        int rows[96],cols[96]; double refs[96]={0};
        for(int t=0;t<96;t++){
            rows[t]=(int)(step(&gen)%(uint32_t)m); cols[t]=(int)(step(&gen)%(uint32_t)n);
            for(int l=0;l<k;l++) refs[t]+=(double)a[(size_t)rows[t]*lda+l]*b[(size_t)l*ldb+cols[t]];
        }
        for(int i=0;i<m;i++) for(int j=n;j<ldc;j++) c[(size_t)i*ldc+j]=-334.f;
        for(int block=-1;block<8;block++) for(int pos=0;pos<4;pos++){
            int method=block<0?pos:order[(f+block)%4][block>=4?3-pos:pos];
            double t=now();
            fs[method](m,n,k,a,lda,b,ldb,c,ldc);
            double elapsed=now()-t,err=0.;
            for(int h=0;h<96;h++) {
                double x=c[(size_t)rows[h]*ldc+cols[h]];
                if(!isfinite(x)) return 4;
                err=fmax(err,fabs(x-refs[h]));
            }
            for(int i=0;i<m;i++) for(int j=n;j<ldc;j++) if(c[(size_t)i*ldc+j]!=-334.f) return 5;
            if(block>=0) printf("%d,%d,%d,%u,%s,%d,%d,%s,%.10g,%.10g\n",m,n,k,seed,name[f],block,pos,tags[method],elapsed,err);
        }
        fflush(stdout);
    }
    free(a-3);free(b-5);free(c-7);return 0;
}
