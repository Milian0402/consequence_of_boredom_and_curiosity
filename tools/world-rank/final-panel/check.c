#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern int final_panel_run(int,int,int,const float*,int,const float*,int,float*,int);
extern int prior_panel_run(int,int,int,const float*,int,const float*,int,float*,int);

static float draw(uint32_t* state)
{
    *state=*state*1664525u+1013904223u;
    return (float)((int)((*state>>8)%2001u)-1000)/997.0f;
}

static int check(int m,int n,int k,int pad,int pattern)
{
    const int lda=k+pad,ldb=n+2*pad,ldc=n+3*pad;
    const size_t as=(size_t)m*lda,bs=(size_t)k*ldb,cs=(size_t)m*ldc;
    float* x=malloc((as+16)*4),*y=malloc((bs+16)*4);
    float* d=malloc((cs+16)*4),*e=malloc((cs+16)*4);
    if (!x||!y||!d||!e) return 1;
    float* a=x+8,*b=y+8,*dest=d+8,*ref=e+8;
    for(size_t z=0;z<as+16;++z)x[z]=NAN;
    for(size_t z=0;z<bs+16;++z)y[z]=NAN;
    for(size_t z=0;z<cs+16;++z)d[z]=e[z]=130712.5f;
    uint32_t seed=0x19af76c1;
    for(int i=0;i<m;++i)for(int p=0;p<k;++p){
        float v=draw(&seed);
        if(pattern==1){int h=m/2;int r=i%h,c=p%h;float base=(float)((r*17+c*29)%509-254)/257.f,eps=((r+c)&1)?7.e-4f:-9.e-4f;v=i<h?(p<h?base:-base+2*eps):(p<h?base+eps:-base+eps);}
        if(pattern==2)v=(i+p)%2 ? 0.f:-0.f;
        a[(size_t)i*lda+p]=v;
    }
    for(int p=0;p<k;++p)for(int j=0;j<n;++j){
        float v=draw(&seed);
        if(pattern==1){int h=m/2;int r=p%h,c=j%h;float base=(float)((r*31+c*13)%503-251)/263.f,eps=((r+2*c)&1)?8.e-4f:-6.e-4f;v=p<h?(j<h?base:-base+2*eps):(j<h?base+eps:-base+eps);}
        b[(size_t)p*ldb+j]=v;
    }
    if(!final_panel_run(m,n,k,a,lda,b,ldb,dest,ldc)||!prior_panel_run(m,n,k,a,lda,b,ldb,ref,ldc))return 2;
    if(memcmp(d,e,(cs+16)*4)){fprintf(stderr,"bit mismatch %d,%d,%d\n",m,n,k);return 3;}
    double max_err=0; int ok=1;
    for(int i=0;i<m;++i)for(int j=0;j<ldc;++j) {
        if(j>=n){if(dest[(size_t)i*ldc+j]!=130712.5f)ok=0;continue;}
        double dot=0;for(int p=0;p<k;++p)dot+=(double)a[(size_t)i*lda+p]*b[(size_t)p*ldb+j];
        double err=fabs(dot-dest[(size_t)i*ldc+j]);if(!isfinite(err))ok=0;
        max_err=fmax(max_err,err);
    }
    for(int i=0;i<8;++i)if(d[i]!=130712.5f||d[cs+8+i]!=130712.5f)ok=0;
    printf("%d,%d,%d,%d,%d,%zu,%.9g,%s\n",m,n,k,pad,pattern,(size_t)m*n,max_err,ok&&max_err<=.002?"pass":"fail");
    free(x);free(y);free(d);free(e);return !ok||max_err>.002;
}
int main(void)
{
    const int cases[][5]={{32,32,32,0,0},{48,64,40,3,0},{49,67,63,5,0},{73,97,81,7,0},
      {149,157,512,2,0},{153,159,513,3,0},{143,173,514,5,0},{241,251,1025,7,0},
      {144,144,144,9,1},{145,145,145,5,2}};
    puts("m,n,k,pad,pattern,fp64_cells,max_abs,bitwise_guards_fp64");
    for(size_t i=0;i<sizeof cases/sizeof *cases;++i)if(check(cases[i][0],cases[i][1],cases[i][2],cases[i][3],cases[i][4]))return 1;
    return 0;
}
