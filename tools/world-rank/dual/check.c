#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
void rank_vendor_sgemm(int,int,int,const float*,int,const float*,int,float*,int);
int rank_vendor_init(void);

static unsigned state=170113;
static float next(void) {
    state=1664525*state+1013904223;
    return (float)(state>>8)*(1.f/8388608.f)-1.f;
}
int main(void) {
    if(rank_vendor_init()!=0) return 2;
    const int shapes[][4]={{1,3,0,3},{3,1,7,1},{11,9,17,3},{31,47,13,5},
        {48,96,27,0},{49,401,39,7},{257,259,16,3},{119,483,65,3}};
    double worst=0;
    size_t checked=0;
    for(size_t s=0;s<sizeof(shapes)/sizeof(shapes[0]);s++) {
        const int* shape=shapes[s];
        const int m=shape[0],n=shape[1],k=shape[2],pad=shape[3];
        const int lda=k+pad+2,ldb=n+pad+3,ldc=n+pad+1;
        size_t na=(size_t)m*lda,nb=(size_t)(k?k:1)*ldb,nc=(size_t)m*ldc;
        float *a=malloc(na*sizeof(float)),*b=malloc(nb*sizeof(float));
        float *mem=malloc((nc+2)*sizeof(float)),*c=mem?mem+1:NULL;
        if(!a||!b||!mem) return 2;
        mem[0]=mem[nc+1]=82.125f;
        for(int i=0;i<m;i++) for(int p=0;p<lda;p++) a[i*lda+p]=
            p<k ? (s==7 && (p&1)?a[i*lda+p-1]:next()) : NAN;
        for(int p=0;p<k;p++) for(int j=0;j<ldb;j++)
            b[p*ldb+j]=j<n?(s==7 && (p&1)?-b[(p-1)*ldb+j]:next()):NAN;
        for(int i=0;i<m;i++) for(int j=0;j<ldc;j++) c[i*ldc+j]=j<n?NAN:82.125f;
        rank_vendor_sgemm(m,n,k,a,lda,b,ldb,c,ldc);
        if(mem[0]!=82.125f||mem[nc+1]!=82.125f) return 1;
        for(int i=0;i<m;i++) for(int j=0;j<ldc;j++) {
            if(j>=n) { if(c[i*ldc+j]!=82.125f) return 1; continue; }
            double d=0; float f=0;
            for(int p=0;p<k;p++){
                d+=(double)a[i*lda+p]*b[p*ldb+j];
                f=fmaf(a[i*lda+p],b[p*ldb+j],f);
            }
            const float actual=c[i*ldc+j];
            if(!isfinite(actual)||!isfinite(d)||fabs((double)actual-d)>.002||memcmp(&actual,&f,sizeof(f))) {
                fprintf(stderr,"mismatch %d %d %d, cell %d %d: %.9g %.15g\n",m,n,k,i,j,actual,d);
                return 1;
            }
            worst=fmax(worst,fabs((double)actual-d));checked++;
        }
        free(a);free(b);free(mem);
    }
    printf("%zu FP64 and bitwise-FMA cells with guarded C padding, max absolute error %.9g\n",checked,worst);
    return 0;
}
