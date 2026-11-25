#include "precision.h"
#include <stdio.h>
#include <stdlib.h>
#if defined(PRECISION_FUSED)
extern int amx_fused6_init(void);
#endif
#ifndef PRECISION_MODE
#error missing registered precision mode
#endif
int rank_vendor_init(void){
#if defined(PRECISION_FUSED)
    return amx_fused6_init();
#else
    return 0;
#endif
}
void rank_vendor_sgemm(int m,int n,int k,const float *a,int lda,const float *b,int ldb,float *c,int ldc)
{
    int rc=precision_sgemm(PRECISION_MODE,m,n,k,a,lda,b,ldb,c,ldc);
    if(rc){fprintf(stderr,"precision method %d failed: %s (%d)\n",PRECISION_MODE,precision_last_reason(),rc);abort();}
}
