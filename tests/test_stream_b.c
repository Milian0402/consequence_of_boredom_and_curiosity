#define _DEFAULT_SOURCE 1
#define _DARWIN_C_SOURCE 1
#define _POSIX_C_SOURCE 200112L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

/* The test can hide real CPU features but it never grants unavailable ones. */
static int hide_all, hide_avx512, observed_width, observations;
#define COB_STREAM_B_SUPPORTS(name) (!hide_all && \
    !(hide_avx512 && strcmp(name, "avx512f") == 0) && __builtin_cpu_supports(name))
#define COB_STREAM_B_OBSERVE(width) do { observed_width=(width); ++observations; } while (0)
static int next_failure, requests;
static int test_memalign(void **mem,size_t align,size_t bytes)
{
    ++requests;
    if (next_failure) { next_failure=0; return 12; }
    return posix_memalign(mem,align,bytes);
}
#define posix_memalign test_memalign
#include "../src/gemm.c"
#undef posix_memalign

static uint32_t state=0xa4217625;
static float draw(void) {
    state=state*1664525u+1013904223u;
    return (int)((state>>8)&65535)/32768.f-1.f;
}
static void insist(int ok,const char *why) {
    if (!ok) { fprintf(stderr,"Failed: %s\n",why); exit(1); }
}
static size_t checked;
static double worst;

static void full(int m,int n,int k,int skew,int fail,int accepted) {
    int la=k+skew,lb=n+2*skew,lc=n+3*skew,guard=17;
    size_t na=(size_t)m*la+2*guard, nb=(size_t)k*lb+2*guard, nc=(size_t)m*lc+2*guard;
    float *raw_a=malloc(na*sizeof(float)), *raw_b=malloc(nb*sizeof(float)), *raw_c=malloc(nc*sizeof(float));
    float *saved_a=malloc(na*sizeof(float)), *saved_b=malloc(nb*sizeof(float));
    insist(raw_a && raw_b && raw_c && saved_a && saved_b,"allocation");
    float *a=raw_a+guard, *b=raw_b+guard, *c=raw_c+guard;
    for (size_t i=0;i<na;++i) raw_a[i]=NAN;
    for (size_t i=0;i<nb;++i) raw_b[i]=NAN;
    for (size_t i=0;i<nc;++i) raw_c[i]=719.25f;
    for(int i=0;i<m;++i) for(int p=0;p<k;++p) a[(size_t)i*la+p]=draw();
    for(int p=0;p<k;++p) for(int j=0;j<n;++j) b[(size_t)p*lb+j]=draw();
    for(int i=0;i<m;++i) for(int j=0;j<n;++j) c[(size_t)i*lc+j]=NAN;
    memcpy(saved_a,raw_a,na*sizeof(float));memcpy(saved_b,raw_b,nb*sizeof(float));
    requests=0; next_failure=fail;
    observed_width=0; observations=0;
    int expected_width=accepted ? cob_stream_b_tile_width() : 0;
    cob_sgemm_rowmajor(m,n,k,a,la,b,lb,c,lc);
    insist(observations==(expected_width!=0) && observed_width==expected_width,"actual public ISA selection");
    if (expected_width) insist(requests==1,"one B panel allocation");
    if (fail) insist(!next_failure,"allocation refusal exercised");
    for(size_t x=0;x<nc;++x) {
        if(x<(size_t)guard || x>=nc-guard || (x-guard)%lc>=(size_t)n) {
            insist(raw_c[x]==719.25f,"output guard");continue;
        }
        int i=(x-guard)/lc,j=(x-guard)%lc; double ref=0;
        for(int p=0;p<k;++p) ref+=(double)a[(size_t)i*la+p]*b[(size_t)p*lb+j];
        double error=fabs(raw_c[x]-ref);
        insist(isfinite(error) && error<=0.002,"FP64 error cap");
        if(error>worst)worst=error;
        ++checked;
    }
    insist(!memcmp(raw_a,saved_a,na*sizeof(float))&&!memcmp(raw_b,saved_b,nb*sizeof(float)),"inputs changed");
    /* Refusals must occur without allocation or writes, including overlapping subranges. */
    requests=0;
    insist(!cob_try_stream_b(m,n,k,a,la,b,lb,a+1,lc),"C overlaps A");
    insist(!cob_try_stream_b(m,n,k,a,la,b,lb,b+1,lc),"C overlaps B");
    insist(!cob_try_stream_b(m,n,k,a,k-1,b,lb,c,lc),"short A stride");
    insist(!cob_try_stream_b(m,n,k,a,la,b,n-1,c,lc),"short B stride");
    insist(!cob_try_stream_b(m,n,k,a,la,b,lb,c,n-1),"short C stride");
    insist(!cob_try_stream_b(m,n,k,NULL,la,b,lb,c,lc),"null A");
    insist(!requests&&!memcmp(raw_a,saved_a,na*sizeof(float))&&!memcmp(raw_b,saved_b,nb*sizeof(float)),"guard mutated input");
    free(saved_b);free(saved_a);free(raw_c);free(raw_b);free(raw_a);
}
struct guarded { float *data; void *mapping; size_t bytes; int ld; };
static struct guarded guarded_rows(int rows, int cols, int input) {
    const size_t page=(size_t)sysconf(_SC_PAGESIZE);
    size_t valid=((size_t)cols*sizeof(float)+page-1)/page*page, stride=valid+page;
    const size_t bytes=(size_t)rows*stride;
    char *base=mmap(NULL,bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    insist(base!=MAP_FAILED,"mmap");
    float *data=(float *)(base+valid-cols*sizeof(float));
    for(int i=0;i<rows;++i) {
        float *row=(float *)((char *)data+(size_t)i*stride);
        for(int j=0;j<cols;++j) row[j]=input?((i+2*j)%7-3):NAN;
        if(input) insist(mprotect(base+(size_t)i*stride,valid,PROT_READ)==0,"read-only input");
        insist(mprotect(base+(size_t)i*stride+valid,page,PROT_NONE)==0,"guard page");
    }
    return (struct guarded){data,base,bytes,(int)(stride/sizeof(float))};
}
static void pages(void) {
    const int m=29,n=1025,k=107;
    struct guarded a=guarded_rows(m,k,1),b=guarded_rows(k,n,1),c=guarded_rows(m,n,0);
    observed_width=observations=0;
    int expected_width=cob_stream_b_tile_width();
    cob_sgemm_rowmajor(m,n,k,a.data,a.ld,b.data,b.ld,c.data,c.ld);
    insist(observed_width==expected_width && observations==(expected_width!=0),"protected public route");
    for(int i=0;i<m;++i) for(int j=0;j<n;++j) {
        double ref=0;
        for(int p=0;p<k;++p) ref+=(double)a.data[(size_t)i*a.ld+p]*b.data[(size_t)p*b.ld+j];
        insist(c.data[(size_t)i*c.ld+j]==ref,"protected exact output");
        ++checked;
    }
    munmap(a.mapping,a.bytes);munmap(b.mapping,b.bytes);munmap(c.mapping,c.bytes);
}
int main(void) {
    const int native_width=cob_stream_b_tile_width();
    insist(native_width==0 || native_width==32 || native_width==64,"unexpected tile width");
    pages();
    for (int rem=0;rem<6;++rem) full(24+rem,1024+rem,96+rem,3,0,1);
    full(49,2861,157,11,0,1);
    if (native_width) full(29,1029,101,3,1,0);
    full(23,1024,96,3,0,0);
    full(24,1023,96,3,0,0);
    full(24,1024,95,3,0,0);
    hide_avx512=1;
    const int avx2_width=cob_stream_b_tile_width();
    insist(avx2_width==0 || avx2_width==32,"AVX-512 masking");
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
        insist(avx2_width==32,"AVX2 kernel remains available");
        for (int rem=0;rem<3;++rem) full(24+rem,1088+rem,108+rem,2,0,1);
        pages();
    }
    hide_all=1;
    insist(cob_stream_b_tile_width()==0,"CPU without SIMD");
    full(29,1029,101,3,0,0);
    printf("PASS native_width=%d AVX2_width=%d checked=%zu max_abs_error=%.10g\n",native_width,avx2_width,checked,worst);
    return 0;
}
