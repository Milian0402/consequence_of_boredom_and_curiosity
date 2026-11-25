/* Compile this independent test with the patched src directory on the include path. */
#define _GNU_SOURCE 1
#define _DARWIN_C_SOURCE 1
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include "gemm.c"

static uint32_t rng=0xbeef1237;
static float sample(void) {
    rng=rng*1664525u+1013904223u;
    return (float)(int32_t)(rng>>8)*(1.f/8388608.f)-1.f;
}

typedef struct { char* start; size_t len,page; float* c; int stride; } guarded;
static guarded guarded_rows(int rows,int cols) {
    const size_t page=(size_t)sysconf(_SC_PAGESIZE);
    const size_t len=(2*rows+3)*page;
    char* start=mmap(NULL,len,PROT_NONE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    if(start==MAP_FAILED || cols*sizeof(float)+sizeof(float)>page) exit(2);
    for(int r=0;r<rows;r++) {
        char* at=start+(1+2*r)*page;
        if(mprotect(at,page,PROT_READ|PROT_WRITE)) exit(2);
    }
    return (guarded){start,len,page,(float*)(start+2*page)-cols,(int)(2*page/sizeof(float))};
}

static size_t checked;
static double worst;
static void value(float actual,float old,double ref,int r,int j,int mr,int nr) {
    double err=fabs((double)actual-ref);
    if(!isfinite(actual)||!isfinite(old)||!isfinite(ref)||err>0.002||memcmp(&actual,&old,4)!=0) {
        fprintf(stderr,"mismatch %dx%d [%d,%d]: candidate %.9g baseline %.9g FP64 %.15g\n",
            mr,nr,r,j,actual,old,ref);
        exit(1);
    }
    if(err>worst) worst=err;
    ++checked;
}

/* Each active row ends exactly at an inaccessible page, including the last row.
   The next would-be row is also unmapped. Both overwrite and resumed products
   are evaluated against the old full tile and an independent FP64 dot product. */
static void tile_shapes(cob_blocked_kernel kernel) {
    enum { K=27, CUT=9 };
    _Alignas(64) float ap[K*12],bp[K*32],old[12*32];
    float init[12*32];
    const int mr=kernel.mr,nr=kernel.nr;
    int trials=0;
    checked=0; worst=0;
    for(int rows=1;rows<=mr;rows++) for(int cols=1;cols<=nr;cols++) {
        guarded g=guarded_rows(rows,cols);
        for(int t=0;t<K;t++) {
            for(int r=0;r<mr;r++) ap[t*mr+r]=r<rows ? sample() : NAN;
            for(int j=0;j<nr;j++) bp[t*nr+j]=j<cols ? sample() : NAN;
        }
        for(int i=0;i<mr*nr;i++) init[i]=sample();
        for(int mode=0;mode<3;mode++) {
            int resume=mode==1;
            memcpy(old,init,mr*nr*sizeof(float));
            for(int r=0;r<rows;r++) {
                float* dst=g.c+(size_t)r*g.stride;
                dst[-1]=713.125f;
                for(int j=0;j<cols;j++) dst[j]=resume?init[r*nr+j]:NAN;
            }
            int first=mode==2?CUT:K;
            if(!cob_x86_cache_edge(kernel,first,ap,bp,g.c,g.stride,rows,cols,resume)) exit(2);
            kernel.tile(first,ap,bp,old,nr,resume);
            if(first<K) {
                if(!cob_x86_cache_edge(kernel,K-first,ap+first*mr,bp+first*nr,g.c,g.stride,rows,cols,1)) exit(2);
                kernel.tile(K-first,ap+first*mr,bp+first*nr,old,nr,1);
            }
            for(int r=0;r<rows;r++) {
                float* dst=g.c+(size_t)r*g.stride;
                if(dst[-1]!=713.125f) exit(1);
                for(int j=0;j<cols;j++) {
                    double ref=resume ? init[r*nr+j] : 0;
                    for(int t=0;t<K;t++) ref+=(double)ap[t*mr+r]*bp[t*nr+j];
                    value(dst[j],old[r*nr+j],ref,r,j,rows,cols);
                }
            }
            trials++;
        }
        if(munmap(g.start,g.len)) exit(2);
    }
    printf("exact-edge %dx%d: %d shape/mode combinations, %zu bit-identical outputs, max FP64 error %.9g\n",
        mr,nr,trials,checked,worst);
}

static void entry(cob_blocked_kernel kernel) {
    enum { M=73, N=273, K=271, LDA=K+7, LDB=N+11 };
    float *a=malloc(sizeof(float)*M*LDA),*b=malloc(sizeof(float)*K*LDB);
    guarded g=guarded_rows(M,N);
    if(!a||!b) exit(2);
    for(int r=0;r<M;r++) for(int t=0;t<LDA;t++) a[r*LDA+t]=t<K?sample():NAN;
    for(int t=0;t<K;t++) for(int j=0;j<LDB;j++) b[t*LDB+j]=j<N?sample():NAN;
    for(int r=0;r<M;r++) {
        (g.c+(size_t)r*g.stride)[-1]=713.125f;
        for(int j=0;j<N;j++) g.c[(size_t)r*g.stride+j]=NAN;
    }
    if(!cob_sgemm_cache_panel(kernel,M,N,K,a,LDA,b,LDB,g.c,g.stride)) exit(1);
    checked=0; worst=0;
    for(int r=0;r<M;r++) {
        if((g.c+(size_t)r*g.stride)[-1]!=713.125f) exit(1);
        for(int j=0;j<N;j++) {
            double ref=0;
            float serial=0;
            for(int t=0;t<K;t++) {
                ref+=(double)a[r*LDA+t]*b[t*LDB+j];
                serial=fmaf(a[r*LDA+t],b[t*LDB+j],serial);
            }
            value(g.c[(size_t)r*g.stride+j],serial,ref,r,j,M,N);
        }
    }
    printf("cache-panel %dx%d on %dx%dx%d: %zu outputs, max FP64 error %.9g\n",
        kernel.mr,kernel.nr,M,N,K,checked,worst);
    if(munmap(g.start,g.len)) exit(2);
    free(a);free(b);
}

int main(int argc,char** argv) {
    int attempted=0;
    int translated=argc==2&&!strcmp(argv[1],"translated-avx2");
    if(cob_x86_cache_edge((cob_blocked_kernel){14,32,0,NULL},1,NULL,NULL,NULL,0,1,1,0)) return 1;
    if(translated||(__builtin_cpu_supports("avx2")&&__builtin_cpu_supports("fma"))) {
        cob_blocked_kernel avx2={6,16,120,cob_sgemm_tile_avx2};
        tile_shapes(avx2);entry(avx2);attempted=1;
    } else puts("AVX2/FMA skipped for host features");
    if(__builtin_cpu_supports("avx512f")&&__builtin_cpu_supports("fma")) {
        cob_blocked_kernel avx512={12,32,120,cob_blocked_tile_12x32};
        tile_shapes(avx512);entry(avx512);attempted=1;
    } else puts("AVX-512/FMA skipped for host features");
    return attempted?0:77;
}
