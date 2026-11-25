#include <stdio.h>
extern int stream_trace_old_medium,stream_trace_blocked;
#if STREAM_CANDIDATE
extern int cob_stream_b_count(int);
#endif
extern void __real_cob_sgemm_rowmajor(int,int,int,const float*,int,const float*,int,float*,int);
void __wrap_cob_sgemm_rowmajor(int m,int n,int k,const float *a,int lda,const float *b,int ldb,float *c,int ldc)
{
    int counts[5]={0};
    __real_cob_sgemm_rowmajor(m,n,k,a,lda,b,ldb,c,ldc);
#if STREAM_CANDIDATE
    for (int i=0;i<5;i++) counts[i]=cob_stream_b_count(i);
#endif
    printf("STREAM,selected=%d,mr=%d,nr=%d,packs=%d,tiles=%d,old_medium=%d,blocked=%d\n",
        counts[0],counts[1],counts[2],counts[3],counts[4],stream_trace_old_medium,stream_trace_blocked);
}
