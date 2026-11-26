#include "cob_gemm.h"
int rank_vendor_init(void) { return 0; }
void rank_vendor_sgemm(int m,int n,int k,const float*a,int la,const float*b,int lb,float*c,int lc)
{ cob_sgemm_rowmajor(m,n,k,a,la,b,lb,c,lc); }
