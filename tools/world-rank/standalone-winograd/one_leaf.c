/* The original non-HK public dispatcher is file-local; call it here with
   recursive Winograd disabled. This leaves at most one fused/stored Strassen
   level. HK is compiled to retain its private base/trace ABI but is bypassed. */
#include "gemm.c"

void cob_winograd_one_public(int m,int n,int k,const float* a,int lda,
    const float* b,int ldb,float* c,int ldc)
{
#ifdef COB_WINOGRAD_TRACE
    cob_hk15_children=0;
    cob_hk15_child_fused=0;
    cob_hk15_other_fused=0;
    cob_hk15_trace_inside=0;
    cob_hk15_lower_children=0;
#endif
    cob_sgemm_rowmajor_base(m,n,k,a,lda,b,ldb,c,ldc);
}
