#define _DARWIN_C_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
typedef void (*gemm_fn)(int,int,int,const float*,int,const float*,int,float*,int);
typedef struct { char *map; float *data; size_t bytes, pg, band; int stride,rows,cols; } guarded;
static guarded rows(int r,int c) {
    size_t pg=(size_t)sysconf(_SC_PAGESIZE),bytes=(size_t)c*sizeof(float),band=(bytes+pg-1)/pg*pg;
    size_t span=(r*(band+pg));char *base=mmap(NULL,span,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANON,-1,0);
    if(base==MAP_FAILED)abort();
    for(int i=0;i<r;++i){if(mprotect(base+(size_t)i*(band+pg)+band,pg,PROT_NONE))abort();}
    return (guarded){base,(float*)(base+band-bytes),span,pg,band,(int)((band+pg)/sizeof(float)),r,c};
}
static void readonly(guarded g) {for(int i=0;i<g.rows;++i) if(mprotect(g.map+(size_t)i*(g.band+g.pg),g.band,PROT_READ))abort();}
static uint32_t roll(uint32_t *r) {*r=*r*1664525u+1013904223u;return *r;}
static float next(uint32_t *s,int i,int j,int mode) {
    float v=(float)((int)(roll(s)%8191)-4095)/4095.0f;
    if(mode==1) return (1.0f+0.000016f*v)*(((i+j)&1)?-1:1);
    if(mode==2) return ldexpf(v,(int)(roll(s)%13)-10);
    return v;
}
typedef struct {int m,n,k;} shape;
int main(int argc,char **argv) {
    if(argc!=2 && argc!=3)return 2;
    void *h=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL|RTLD_FIRST);if(!h)return 2;
    gemm_fn run=(gemm_fn)dlsym(h,"cob_sgemm_rowmajor"); if(!run)return 2;
    uint64_t (*seen)(void)=(uint64_t(*)(void))dlsym(h,"frontier_calls");

    shape cases[]={{28,1057,107},{31,4033,199},{34,1799,241},{46,2983,353},{49,2059,127},{52,3403,283},{24,1024,96},{55,4096,384}};
    puts("m,n,k,fixture,checked,max_abs_error,expected_route,actual_route,prefix_guard,inputs_readonly");
    for(size_t ci=0;ci<sizeof(cases)/sizeof(*cases);++ci) for(int mode=0;mode<3;++mode) {
        shape s=cases[ci];int m=s.m,n=s.n,k=s.k;
        guarded a=rows(m,k),b=rows(k,n),c=rows(m,n);uint32_t seed=1228381+1231*mode;
        for(int i=0;i<m;++i)for(int j=0;j<k;++j)a.data[(size_t)i*a.stride+j]=next(&seed,i,j,mode);
        for(int i=0;i<k;++i)for(int j=0;j<n;++j) {
            float x=next(&seed,i,j,mode);
            b.data[(size_t)i*b.stride+j]=mode==1 && (i&1)?-x:x;
        }
        readonly(a);readonly(b);
        size_t nc=(size_t)m*n;
        double *ref=calloc(nc,sizeof(*ref));if(!ref)abort();
        for(int i=0;i<m;++i)for(int p=0;p<k;++p) {
            double av=a.data[(size_t)i*a.stride+p];
            for(int j=0;j<n;++j)ref[(size_t)i*n+j]+=av*b.data[(size_t)p*b.stride+j];
        }
        for(int i=0;i<m;++i) {
            float *row=c.data+(size_t)i*c.stride;
            for(int j=0;j<n;++j)row[j]=NAN;
            for(float *q=(float*)(c.map+(size_t)i*(c.band+c.pg));q<row;++q)*q=-227.5f;
        }
        uint64_t old=seen?seen():0;
        run(m,n,k,a.data,a.stride,b.data,b.stride,c.data,c.stride);
        int eligible=m>=24&&m<=55&&n>=1024&&n<=4096&&k>=96&&k<=384;
        int nseen=seen?(int)(seen()-old):-1;
        if(seen && nseen!=eligible)return 3;
        double error=0;
        for(int i=0;i<m;++i) {
            float *row=c.data+(size_t)i*c.stride;
            for(int j=0;j<n;++j) {
                double d=fabs(row[j]-ref[(size_t)i*n+j]);
                if(!isfinite(d)||d>.002){fprintf(stderr,"FP64 %d %d %d %d %d %d %.12g\n",m,n,k,mode,i,j,d);return 4;}
                error=fmax(error,d);
            }
            for(float *q=(float*)(c.map+(size_t)i*(c.band+c.pg));q<row;++q)if(*q!=-227.5f)return 5;
        }
        printf("%d,%d,%d,%s,%zu,%.12g,%d,%d,1,1\n",m,n,k,mode==0?"random":mode==1?"cancellation":"mixed-scale",nc,error,eligible,nseen);
        free(ref);munmap(a.map,a.bytes);munmap(b.map,b.bytes);munmap(c.map,c.bytes);
    }
    if(seen) {
        enum {m=24,k=96,n=1024};size_t size=(size_t)k*n;
        float *a=calloc(size,sizeof(float)),*c=calloc(size,sizeof(float));if(!a||!c)abort();
        uint64_t before=seen();run(m,n,k,a,k,a,n,c,n);
        if(seen()!=before+1)return 6;
        before=seen();run(m,n,k,a,k,a,n,a,n);if(seen()!=before)return 7;
        run(m,n,k,a,k,c,n,c,n);if(seen()!=before)return 8;
        free(a);free(c);
        fprintf(stderr,"shared A/B selected; C/A and C/B overlap bypassed\n");
    }
    return 0;
}
