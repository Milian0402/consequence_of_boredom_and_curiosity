/* Every logical entry of A and B is a signed power of two or zero. All
 * expected dots reduce to a signed integer K (or a signed sum of ones)
 * times an explicit power of two and are exactly representable in FP32. */
#include "cob_gemm.h"
#ifdef WITH_OPENBLAS
#include <cblas.h>
#endif
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum axis { OUTER, INSIDE, SCATTER, SPIKE, ZERO_AXES, CANCEL, GLOBAL, TINY };
static const char *names[]={"outer","inside","scatter","spike","zero-axes","cancel","global","subnormal"};
struct sample { enum axis pattern; int shift; };
static const struct sample samples[]={
    {OUTER,0},{OUTER,4},{OUTER,5},{OUTER,6},{OUTER,7},{OUTER,8},
    {OUTER,12},{OUTER,16},{OUTER,24},{INSIDE,8},{INSIDE,16},{INSIDE,24},
    {SCATTER,8},{SCATTER,16},{SPIKE,12},{ZERO_AXES,12},{CANCEL,10},
    {GLOBAL,127},{TINY,0}
};

static float row_factor(enum axis p,int shift,int i,int m)
{
    if(p==GLOBAL) return ldexpf(1,shift);
    if(p==TINY) return ldexpf(1,-140);
    if(p==INSIDE) return 1;
    if(p==ZERO_AXES && i%23==0) return 0;
    int small=i<m/2;
    if(p==SCATTER || p==ZERO_AXES || p==CANCEL) small=((unsigned)i*37%11)<5;
    if(p==SPIKE) small=i==m/2+3;
    return ldexpf(p==CANCEL && i%5==0 ? -1.f:1.f,small?-shift:0);
}
static float col_factor(enum axis p,int shift,int j,int n)
{
    if(p==GLOBAL) return ldexpf(1,-shift);
    if(p==TINY) return ldexpf(1,125);
    if(p==INSIDE) return 1;
    if(p==ZERO_AXES && j%29==0) return 0;
    int small=j<n/3;
    if(p==SCATTER || p==ZERO_AXES || p==CANCEL) small=((unsigned)j*31%13)<7;
    if(p==SPIKE) small=j==n-3;
    return ldexpf(p==CANCEL && j%3==1 ? -1.f:1.f,small?-shift:0);
}
static int reciprocal_depth(enum axis p){return p==INSIDE || p==SCATTER;}
static int depth_group(enum axis p,int l,int k)
{
    return p==SCATTER ? ((unsigned)l*17%7)<3 : l<k/3;
}

static double wall(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec + t.tv_nsec*1e-9;
}
int main(int argc,char **argv)
{
    if(argc!=5) return 2;
    const int m=atoi(argv[1]),n=atoi(argv[2]),k=atoi(argv[3]);
    const int la=k+9,lb=n+15,lc=n+13;
    if(m<2 || n<3 || k<3 || !(k%2)) return 2;
    const size_t al=(size_t)m*la+6,bl=(size_t)k*lb+9,cl=(size_t)m*lc+12;
    float *am=malloc(al*4), *bm=malloc(bl*4), *cm=malloc(cl*4);
    float *ar=malloc(m*4),*bc=malloc(n*4),*ak=malloc(k*4),*bk=malloc(k*4);
    if(!am || !bm || !cm || !ar || !bc || !ak || !bk) return 3;
    const int ao=3,bo=5,co=7;
    float *a=am+ao,*b=bm+bo,*c=cm+co;
    for(size_t q=0;q<al;q++) am[q]=-455.75f;
    for(size_t q=0;q<bl;q++) bm[q]=745.f;
    for(size_t q=0;q<cl;q++) cm[q]=147.25f;
#ifdef WITH_OPENBLAS
    if(openblas_get_num_threads()!=1) return 4;
#endif
    puts("method,pattern,shift,m,n,k,cells,wrong,zeros_lost,negative_from_positive,nonfinite,over_0_002,gt_1pct,max_abs,max_rel,expected,observed,row,col,wall_s");
    for(size_t s=0;s<sizeof samples/sizeof *samples;s++) {
        const enum axis p=samples[s].pattern; int shift=samples[s].shift;
        for(int i=0;i<m;i++) ar[i]=row_factor(p,shift,i,m);
        for(int j=0;j<n;j++) bc[j]=col_factor(p,shift,j,n);
        double dot=0;
        for(int l=0;l<k;l++) {
            int small=reciprocal_depth(p) && depth_group(p,l,k);
            int other=reciprocal_depth(p) && !small;
            ak[l]=ldexpf(p==CANCEL && (l&1)?-1.f:1.f,small?-shift:0);
            bk[l]=ldexpf(1,other?-shift:0);
            dot+=(double)ak[l]*bk[l];
        }
        for(int i=0;i<m;i++) for(int l=0;l<k;l++) a[(size_t)i*la+l]=ar[i]*ak[l];
        for(int l=0;l<k;l++) for(int j=0;j<n;j++) b[(size_t)l*lb+j]=bk[l]*bc[j];
        for(int i=0;i<m;i++) for(int j=0;j<n;j++) c[(size_t)i*lc+j]=NAN;
        double start=wall();
#ifdef WITH_OPENBLAS
        cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,n,k,1.f,a,la,b,lb,0.f,c,lc);
#else
        cob_sgemm_rowmajor(m,n,k,a,la,b,lb,c,lc);
#endif
        double elapsed=wall()-start;
        int wrong=0,zero=0,neg=0,bad=0,over=0,relative=0,wi=0,wj=0;
        double worst=0,wrel=0,wref=0,wgot=0;
        for(int i=0;i<m;i++) for(int j=0;j<n;j++) {
            const double expected=(double)ar[i]*bc[j]*dot;
            if((double)(float)expected!=expected || !isfinite(expected)) return 8;
            const double value=c[(size_t)i*lc+j];
            const double error=isfinite(value)?fabs(value-expected):INFINITY;
            const double rel=expected?error/fabs(expected):error?INFINITY:0;
            wrong+=value!=expected;zero+=(value==0 && expected!=0);
            neg+=(value<0 && expected>0);bad+=!isfinite(value);
            over+=error>.002;relative+=rel>.01;worst=fmax(worst,error);
            if(rel>wrel){wi=i;wj=j;wref=expected;wgot=value;wrel=rel;}
        }
        for(size_t q=0;q<cl;q++) {
            if(q<(size_t)co || q>=(size_t)co+(size_t)m*lc || (q-co)%lc>=(size_t)n)
                if(cm[q]!=147.25f) return 5;
        }
        for(int i=0;i<m;i++) for(int l=0;l<la;l++)
            if(a[(size_t)i*la+l]!=(l<k ? ar[i]*ak[l] : -455.75f))return 6;
        for(int l=0;l<k;l++) for(int j=0;j<lb;j++)
            if(b[(size_t)l*lb+j]!=(j<n ? bk[l]*bc[j] : 745.f))return 7;
        printf("%s,%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%.12g,%.12g,%.12g,%.12g,%d,%d,%.9g\n",
               argv[4],names[p],shift,m,n,k,m*n,wrong,zero,neg,bad,over,relative,worst,wrel,wref,wgot,wi,wj,elapsed);
        fflush(stdout);
    }
    free(am);free(bm);free(cm);free(ar);free(bc);free(ak);free(bk);return 0;
}
