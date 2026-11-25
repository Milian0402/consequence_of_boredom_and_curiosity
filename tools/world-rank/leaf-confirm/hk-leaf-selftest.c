/* Independent low-cost scalar provider and FP64 reference for serial guard transitions. */
#include "cob_gemm.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cblas.h>

static int mode,calls,checks;
int openblas_get_parallel(void) {return 0;}
char *openblas_get_config(void) {return "fixture-only scalar SGEMM";}
int openblas_get_num_threads(void) {++checks;return mode==1 || (mode==2 && calls>=5) ? 2 : 1;}
void cblas_sgemm(enum CBLAS_ORDER order,enum CBLAS_TRANSPOSE ta,enum CBLAS_TRANSPOSE tb,
    int m,int n,int k,float alpha,const float *a,int lda,const float *b,int ldb,float beta,float *c,int ldc)
{
    if(order!=CblasRowMajor||ta!=CblasNoTrans||tb!=CblasNoTrans||alpha!=1.0f||beta!=0.0f||
        mode==1||(mode==2 && calls>=5)) abort();
    ++calls;
    for(int i=0;i<m;++i)for(int j=0;j<n;++j){
        float s=0.0f;for(int p=0;p<k;++p)s+=a[(size_t)i*lda+p]*b[(size_t)p*ldb+j];
        c[(size_t)i*ldc+j]=s;
    }
}
int main(void)
{
    unsigned long entries=0,guarded=0;int geometries=0;
    for(mode=0;mode<3;++mode)for(int mr=0;mr<2;++mr)for(int nr=0;nr<3;++nr)for(int kr=0;kr<3;++kr) {
        int m=18+mr,n=27+nr,k=27+kr,lda=k+13,ldb=n+17,ldc=n+21;
        size_t sa=(m+4)*lda,sb=(k+4)*ldb,sc=(m+4)*ldc;
        float *A=malloc(sa*4),*B=malloc(sb*4),*C=malloc(sc*4),*A0=malloc(sa*4),*B0=malloc(sb*4);
        if(!A||!B||!C||!A0||!B0)return 1;
        for(size_t i=0;i<sa;++i)A[i]=NAN;
        for(size_t i=0;i<sb;++i)B[i]=NAN;
        for(size_t i=0;i<sc;++i)C[i]=551.75f;
        float *a=A+lda+2,*b=B+ldb+5,*c=C+ldc+11;
        for(int i=0;i<m;++i)for(int p=0;p<k;++p)a[(size_t)i*lda+p]=(5*i+3*p)%9-4;
        for(int p=0;p<k;++p)for(int j=0;j<n;++j)b[(size_t)p*ldb+j]=(5*j+11*p)%7-3;
        memcpy(A0,A,sa*4);memcpy(B0,B,sb*4);calls=checks=0;
        cob_sgemm_rowmajor(m,n,k,a,lda,b,ldb,c,ldc);
        int expected=0;
#if COB_ENABLE_HK15_CBLAS_LEAF
        expected=mode==0?15:(mode==2?5:0);
#endif
        if (calls!=expected) {fprintf(stderr,"serial mode %d expected %d calls, observed %d\n",mode,expected,calls);return 2;}
        if (memcmp(A,A0,sa*4)||memcmp(B,B0,sb*4))return 3;
        for(size_t t=0;t<sc;++t) {
            int i=(int)(t/ldc)-1,j=(int)(t%ldc)-11;
            if(i>=0 && i<m && j>=0 && j<n) {
                double ref=0;
                for(int p=0;p<k;++p)ref+=(double)a[(size_t)i*lda+p]*b[(size_t)p*ldb+j];
                if(C[t]!=ref)return 4;
                ++entries;
            } else {if(C[t]!=551.75f)return 5;++guarded;}
        }
        ++geometries;free(A);free(B);free(C);free(A0);free(B0);
    }
    printf("variant=%d geometries=%d checked=%lu guards=%lu max_error=0 inputs_unchanged=1\n",
        COB_ENABLE_HK15_CBLAS_LEAF,geometries,entries,guarded);
    return 0;
}
