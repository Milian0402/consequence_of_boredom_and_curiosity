#define _GNU_SOURCE
#include "exact_fixture.h"
#include <dlfcn.h>
#include <dirent.h>
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <openssl/evp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#ifdef __linux__
#include <sched.h>
#else
#define MAP_ANONYMOUS MAP_ANON
#endif

typedef void (*sgemm_fn)(int,int,int,const float*,int,const float*,int,float*,int);
typedef struct {float *start,*v; size_t count; int rows,cols,ld,row,col;} view;
static void fail(const char *reason){fprintf(stderr,"exact worker: %s\n",reason);exit(2);}
static int number(const char *s){
    char *end;
    errno=0; long n=strtol(s,&end,10);
    if(errno || *s<'0' || *s>'9' || *end || n<0 || n>1000000000) fail("bad numeric argument");
    return (int)n;
}
static int native_cpu(void){
#ifdef __linux__
    cpu_set_t cpus;
    if (sched_getaffinity(0,sizeof(cpus),&cpus) || CPU_COUNT(&cpus)!=1) fail("non-singleton affinity");
    DIR *dir=opendir("/proc/self/task");
    if(!dir) fail("task inventory");
    struct dirent *e; int n=0;
    while((e=readdir(dir))) if(e->d_name[0]>='0' && e->d_name[0]<='9') ++n;
    closedir(dir);
    int cpu=sched_getcpu();
    if(n!=1 || cpu<0 || !CPU_ISSET(cpu,&cpus)) fail("worker CPU or thread count changed");
    return cpu;
#else
    /* Local harness-only smoke. Recorded Linux rankings require exact affinity. */
    return -1;
#endif
}
static void bits(float *p,uint32_t n){memcpy(p,&n,sizeof(n));}
static view parent(int rows,int cols,int ld,int row,int col,uint32_t guard){
    if(rows<1 || cols<1 || rows>16384 || cols>16384 || ld<cols+col || ld>32768 || row>1024 || col>1024)
        fail("unsupported parent view");
    size_t page=(size_t)sysconf(_SC_PAGESIZE);
    size_t bytes=(((size_t)rows+row+2)*ld*sizeof(float)+page-1)/page*page;
    char *map=mmap(NULL,bytes+2*page,PROT_NONE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    if(map==MAP_FAILED || mprotect(map+page,bytes,PROT_READ|PROT_WRITE)) fail("guarded parent allocation");
    float *start=(float*)(map+page);
    for(size_t i=0;i<bytes/4;++i) bits(start+i,guard);
    return (view){start,start+(size_t)row*ld+col,bytes/4,rows,cols,ld,row,col};
}
static void protect(view v){if(mprotect(v.start,v.count*4,PROT_READ))fail("input readonly");}
static void poison(view c){
    for(int i=0;i<c.rows;++i) for(int j=0;j<c.cols;++j) bits(c.v+(size_t)i*c.ld+j,0x7fc11443u);
}
static void digest(view x,char out[65]){
    EVP_MD_CTX* ctx=EVP_MD_CTX_new(); unsigned char bytes[32]; unsigned int size=0;
    if(!ctx || !EVP_DigestInit_ex(ctx,EVP_sha256(),NULL))fail("SHA initialization");
    for(int i=0;i<x.rows;++i)
        if(!EVP_DigestUpdate(ctx,x.v+(size_t)i*x.ld,(size_t)x.cols*sizeof(float)))fail("SHA input");
    if(!EVP_DigestFinal_ex(ctx,bytes,&size)||size!=32)fail("SHA final");
    EVP_MD_CTX_free(ctx);
    for(unsigned int i=0;i<32;++i) snprintf(out+2*i,3,"%02x",bytes[i]);
}
static void *lookup(void *lib,const char *name){void *p=dlsym(lib,name);if(!p)fail(name);return p;}
static int getint(void *lib,const char *s){int (*fn)(void)=lookup(lib,s);return fn();}
static const char *getstr(void *lib,const char *s){const char *(*fn)(void)=lookup(lib,s); const char *v=fn();return v?v:"";}
static double timer(clockid_t which){struct timespec t;if(clock_gettime(which,&t))fail("clock");return t.tv_sec+t.tv_nsec*1e-9;}
static void transfer(const char *path,double *data,size_t n,int write){
    FILE *f=fopen(path,write?"wbx":"rb");if(!f)fail("reference file open");
    int ok=write?fwrite(data,sizeof(double),n,f)==n:fread(data,sizeof(double),n,f)==n && fgetc(f)==EOF && !ferror(f);
    if(fclose(f)||!ok)fail("reference file length or I/O");
}
static void receipt(void *lib,const char *stage){
    if(dlsym(lib,"exact_int8_checked_last_route")){
        unsigned long long (*inspected)(void)=lookup(lib,"exact_int8_checked_last_inspected");
        printf("S8_ROUTE\t%s\t%d\t%s\t%llu\t%d\t%d\t%d\t%d\t%d\t%d\t%s\n",
               stage,getint(lib,"exact_int8_checked_last_route"),getstr(lib,"exact_int8_checked_last_reason"),inspected(),
               getint(lib,"exact_int8_checked_last_amax"),getint(lib,"exact_int8_checked_last_bmax"),
               getint(lib,"exact_int8_checked_last_plans"),getint(lib,"exact_int8_checked_last_products"),
               getint(lib,"exact_int8_checked_last_fallback_calls"),getint(lib,"exact_int8_checked_last_native"),
               getstr(lib,"exact_int8_checked_last_impl"));
    }
    if(dlsym(lib,"exact_last_route")){
        unsigned long long (*inspected)(void)=lookup(lib,"exact_last_inspected");
        printf("ROUTE\t%s\t%d\t%s\t%llu\t%d\t%d\t%d\t%d\t%d\t%d\t%s\n",
            stage,getint(lib,"exact_last_route"),getstr(lib,"exact_last_reason"),inspected(),
            getint(lib,"exact_last_amax"),getint(lib,"exact_last_bmax"),getint(lib,"exact_last_fast_calls"),
            getint(lib,"exact_last_fallback_calls"),getint(lib,"exact_last_products"),getint(lib,"exact_last_native"),getstr(lib,"exact_last_impl"));
    }
    if(dlsym(lib,"precision_last_route")){
        int plans=getint(lib,"precision_last_plan_count");
        int (*native)(int)=lookup(lib,"precision_last_plan_native"); const char *(*impl)(int)=lookup(lib,"precision_last_plan_impl");
        printf("BF16\t%s\t%d\t%d\t%d\t%d\t%s\n",stage,getint(lib,"precision_last_route"),
               getint(lib,"precision_last_products"),plans,plans==1?native(0):0,plans==1?impl(0):"-");
    }
    if(dlsym(lib,"rank_onednn_implementation")){
        unsigned long (*plans)(void)=lookup(lib,"rank_onednn_plans");
        printf("DNN\t%s\t%s\t%lu\t%d\n",stage,getstr(lib,"rank_onednn_implementation"),plans(),getint(lib,"rank_onednn_cached_plan"));
    }
    if(dlsym(lib,"exact_int8_implementation"))
        printf("INT8\t%s\t%d\t%s\n",stage,getint(lib,"exact_int8_last_terms"),getstr(lib,"exact_int8_implementation"));
    if(dlsym(lib,"exact_binary_last_words"))
        printf("BINARY\t%s\t%d\n",stage,getint(lib,"exact_binary_last_words"));
}
static void validate(view c,int k,const double *truth,const double *weight,const char *stage,int cpu,double elapsed,double cpu_s){
    size_t badguard=0,guards=0,nonfinite=0,overabs=0,overgamma=0,inexact=0;
    double max_abs=0,max_rho=0,gamma=(double)k/(16777216.-k);
    for(size_t i=0;i<c.count;++i){
        size_t r=i/c.ld,col=i%c.ld;
        if(r>=(size_t)c.row && r<(size_t)(c.row+c.rows) && col>=(size_t)c.col && col<(size_t)(c.col+c.cols))continue;
        uint32_t v;memcpy(&v,c.start+i,sizeof(v));badguard+=v!=0xc7428000u;++guards;
    }
    for(int i=0;i<c.rows;i++) for(int j=0;j<c.cols;j++){
        const size_t ij=(size_t)i*c.cols+j;double output=c.v[(size_t)i*c.ld+j],w=weight[ij],ref=truth[ij];
        if(!isfinite(w)||w<0||!isfinite(ref))fail("invalid FP64 oracle");
        if(!isfinite(output)){++nonfinite;++inexact;continue;}
        double d=fabs(output-ref),rho=w>0?d/w:(d==0?0:INFINITY);
        if(d>max_abs)max_abs=d;
        if(rho>max_rho)max_rho=rho;
        overabs+=d>0.002; overgamma+=d>gamma*w; inexact+=output!=ref;
    }
    char hash[65];digest(c,hash);
    printf("CHECK\t%s\t%.17g\t%.17g\t%d\t%d\t%zu\t%zu\t%zu\t%zu\t%zu\t%zu\t%zu\t%.17g\t%.17g\t%s\n",
          stage,elapsed,cpu_s,cpu,1,(size_t)c.rows*c.cols,guards,badguard,nonfinite,overabs,overgamma,inexact,max_abs,max_rho,hash);
}
int main(int argc,char **argv){
    if(argc!=21)fail("usage: gold|measure plugin spec corpus seed signed weight m n k lda ldb ldc ar ac br bc cr cc n_calls");
    const int cpu=native_cpu(),m=number(argv[8]),n=number(argv[9]),k=number(argv[10]);
    view a=parent(m,k,number(argv[11]),number(argv[14]),number(argv[15]),0x7fc12345u);
    view b=parent(k,n,number(argv[12]),number(argv[16]),number(argv[17]),0x7fc12345u);
    view c=parent(m,n,number(argv[13]),number(argv[18]),number(argv[19]),0xc7428000u);
    char *end=NULL; errno=0; unsigned long long seed=strtoull(argv[5],&end,10);
    if(errno||*end||argv[5][0]<'0'||argv[5][0]>'9')fail("seed");
    if(exact_fixture_fill(argv[3],argv[4],seed,m,n,k,a.v,a.ld,b.v,b.ld))fail(exact_fixture_error());
    protect(a);protect(b);char ah[65],bh[65];digest(a,ah);digest(b,bh);
    printf("INPUT\t%s\t%s\t%d\t%zu\t%zu\t%zu\n",ah,bh,cpu,(size_t)((uintptr_t)a.v%64),(size_t)((uintptr_t)b.v%64),(size_t)((uintptr_t)c.v%64));
    size_t cells=(size_t)m*n;
    double *truth=malloc(cells*sizeof(double)),*weight=malloc(cells*sizeof(double));
    if(!truth||!weight)fail("oracle allocation");
    if(!strcmp(argv[1],"gold")){
        if(exact_fixture_golden(argv[2],m,n,k,a.v,a.ld,b.v,b.ld,truth,weight))fail(exact_fixture_error());
        if(native_cpu()!=cpu)fail("oracle CPU changed");
        transfer(argv[6],truth,cells,1);transfer(argv[7],weight,cells,1);
        printf("GOLD\t%zu\t%d\t%d\n",cells,cpu,exact_fixture_spots());return 0;
    }
    int calls=number(argv[20]);if(strcmp(argv[1],"measure") || (calls!=1 && calls!=2))fail("invalid operation");
    transfer(argv[6],truth,cells,0);transfer(argv[7],weight,cells,0);
    void *lib=dlopen(argv[2],RTLD_NOW|RTLD_LOCAL);if(!lib)fail(dlerror());
    int (*init)(void)=lookup(lib,"rank_vendor_init");sgemm_fn product=lookup(lib,"rank_vendor_sgemm");
    if(init()||native_cpu()!=cpu)fail("plugin initialization or affinity");
    for(int i=0;i<calls;++i){
        const char *stage=i?"WARM":"FIRST";poison(c);
        double cpu_t=timer(CLOCK_THREAD_CPUTIME_ID),t=timer(CLOCK_MONOTONIC);
        product(m,n,k,a.v,a.ld,b.v,b.ld,c.v,c.ld);
        t=timer(CLOCK_MONOTONIC)-t;cpu_t=timer(CLOCK_THREAD_CPUTIME_ID)-cpu_t;
        if(native_cpu()!=cpu)fail("CPU changed during multiplication");
        receipt(lib,stage);validate(c,k,truth,weight,stage,cpu,t,cpu_t);
    }
    return 0;
}
