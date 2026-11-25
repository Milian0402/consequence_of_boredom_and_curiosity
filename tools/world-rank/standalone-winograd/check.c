/* Verify full matrices at real execution sizes using dense repeated input
   families. Each FP64 reference dot product uses the entire original K. */
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef void (*gemm_fn)(int,int,int,const float*,int,const float*,int,float*,int);
typedef int (*count_fn)(int);
enum { RA=17, CB=19 };
static uint32_t state;
static float sample(void)
{
    state=1664525u*state+1013904223u;
    return (float)(state>>8)*(1.f/8388608.f)-1.f;
}
static void* load(const char* path)
{
    void* lib=dlopen(path,RTLD_NOW|RTLD_LOCAL);
    if(!lib){fprintf(stderr,"%s\n",dlerror());exit(2);}
    int (*init)(void)=(int (*)(void))dlsym(lib,"rank_vendor_init");
    if(!init||init()!=0)exit(2);
    return lib;
}
int main(int argc,char** argv)
{
    if(argc!=10 && argc!=11) {
        fprintf(stderr,"check TIMED TRACE M N K PAD EXPECTED_DEPTH MODE SEED [PREVIOUS_TRACE] (modes random, cancel, scaled)\n");
        return 2;
    }
    void* plain=load(argv[1]); void* trace=load(argv[2]);
    void* prior=argc==11?load(argv[10]):NULL;
    gemm_fn fn[3]={(gemm_fn)dlsym(plain,"rank_vendor_sgemm"),(gemm_fn)dlsym(trace,"rank_vendor_sgemm"),
        prior?(gemm_fn)dlsym(prior,"rank_vendor_sgemm"):NULL};
    count_fn prior_count=prior?(count_fn)dlsym(prior,"rank_winograd_leaf_count"):NULL;
    if(prior&&(!fn[2]||!prior_count))return 2;
    count_fn count=(count_fn)dlsym(trace,"rank_winograd_leaf_count");
    if(!fn[0]||!fn[1]||!count||dlsym(plain,"rank_winograd_leaf_count"))return 2;
    int m=atoi(argv[3]),n=atoi(argv[4]),k=atoi(argv[5]),pad=atoi(argv[6]),depth=atoi(argv[7]);
    const char* mode=argv[8];state=(uint32_t)strtoul(argv[9],NULL,10);
    int cancel=!strcmp(mode,"cancel"), scaled=!strcmp(mode,"scaled");
    if(!cancel&&!scaled&&strcmp(mode,"random"))return 2;
    if(m<1||n<1||k<1||m>12000||n>12000||k>12000||pad<0||pad>64||depth<0||depth>2)return 2;
    const int lda=k+pad,ldb=n+(pad?pad+2:0),ldc=n+(pad?pad+4:0);
    const size_t span=(size_t)m*ldc+1;
    float *a=malloc((size_t)m*lda*sizeof(float)),*b=malloc((size_t)k*ldb*sizeof(float));
    float *c=malloc(span*sizeof(float)),*old=malloc(span*sizeof(float));
    float *at=malloc((size_t)RA*k*sizeof(float)), *bt=malloc((size_t)k*CB*sizeof(float));
    if(!a||!b||!c||!old||!at||!bt)return 2;
    for(int r=0;r<RA;r++)for(int t=0;t<k;t++)
        at[(size_t)r*k+t]=(cancel&&t%2)?at[(size_t)r*k+t-1]:(scaled&&t>=k/2?0.001953125f:1.f)*sample();
    for(int t=0;t<k;t++)for(int j=0;j<CB;j++)
        bt[(size_t)t*CB+j]=(cancel&&t%2)?-bt[(size_t)(t-1)*CB+j]+.0009765625f*sample():
            (scaled&&t<k/2?0.001953125f:1.f)*sample();
    double ref[RA][CB];
    for(int r=0;r<RA;r++)for(int j=0;j<CB;j++) {
        double x=0;
        for(int t=0;t<k;t++)x+=(double)at[(size_t)r*k+t]*bt[(size_t)t*CB+j];
        ref[r][j]=x;
    }
    for(int i=0;i<m;i++) {
        memcpy(a+(size_t)i*lda,at+(size_t)(i%RA)*k,(size_t)k*sizeof(float));
        for(int t=k;t<lda;t++)a[(size_t)i*lda+t]=NAN;
    }
    for(int t=0;t<k;t++)for(int j=0;j<ldb;j++)b[(size_t)t*ldb+j]=j<n?bt[(size_t)t*CB+j%CB]:NAN;
    double worst=0;
    for(int implementation=0;implementation<(prior?3:2);implementation++) {
        for(int i=0;i<m;i++)for(int j=0;j<ldc;j++)c[(size_t)i*ldc+j]=j<n?NAN:713.25f;
        c[span-1]=713.25f;
        fn[implementation](m,n,k,a,lda,b,ldb,c,ldc);
        for(int i=0;i<m;i++){
            for(int j=0;j<n;j++){
                double err=fabs((double)c[(size_t)i*ldc+j]-ref[i%RA][j%CB]);
                if(!isfinite(err)) { fprintf(stderr,"nonfinite at %d,%d\n",i,j);return 1; }
                if(err>worst)worst=err;
            }
            for(int j=n;j<ldc;j++)if(c[(size_t)i*ldc+j]!=713.25f)return 1;
        }
        if(c[span-1]!=713.25f)return 1;
        if(implementation==0)memcpy(old,c,span*sizeof(float));
        else if(memcmp(old,c,span*sizeof(float))) {fputs("plain/trace mismatch\n",stderr);return 1;}
    }
    for(int i=0;i<m;i++){
        if(memcmp(a+(size_t)i*lda,at+(size_t)(i%RA)*k,(size_t)k*sizeof(float)))return 1;
        for(int t=k;t<lda;t++)if(!isnan(a[(size_t)i*lda+t]))return 1;
    }
    for(int t=0;t<k;t++){
        for(int j=0;j<n;j++)if(b[(size_t)t*ldb+j]!=bt[(size_t)t*CB+j%CB])return 1;
        for(int j=n;j<ldb;j++)if(!isnan(b[(size_t)t*ldb+j]))return 1;
    }
    const int tile=1<<depth,core=depth==2?49:depth==1?7:0;
    const int edge=depth?((m%tile)!=0)+((n%tile)!=0):0,direct=depth==0;
    const int expected[]={core+edge+direct,core,edge,depth,direct};
    for(int i=0;i<5;i++)if(count(i)!=expected[i]){
        fprintf(stderr,"counter %d got %d want %d\n",i,count(i),expected[i]);return 1;
    }
    if(prior) {
        for(int id=0;id<=8;id++)if(count(id)!=prior_count(id)) {
            fprintf(stderr,"budget changed a route: id %d, new %d, previous %d\n",id,count(id),prior_count(id));
            return 1;
        }
    }
    printf("previous_identical=%d shape=%dx%dx%d mode=%s depth=%d core=%d edge=%d hk=%d fused=%d hk_fused=%d hk_children=%d checked=%zu max_abs=%.9g\n",
        prior!=NULL,m,n,k,mode,depth,core,edge,count(5),count(6),count(7),count(8),(size_t)m*n,worst);
    const float sent=713.25f;c[0]=sent;
    fn[1](m,n,0,a,lda,b,ldb,c,ldc);
    if(c[0]!=sent||count(0)!=0)return 1;
    fn[1](m,n,k,a,k-1,b,ldb,c,ldc);
    if(c[0]!=sent||count(0)!=0)return 1;
    free(a);free(b);free(c);free(old);free(at);free(bt);if(prior)dlclose(prior);dlclose(plain);dlclose(trace);
    return worst<=0.002?0:1;
}
