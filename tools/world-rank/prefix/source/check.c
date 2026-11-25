#define _GNU_SOURCE
#include "prefix.h"
#include <cpuid.h>
#include <dlfcn.h>
#include <dirent.h>
#include <math.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

static void require(int yes,const char *message)
{
    if (!yes) { fprintf(stderr,"prefix witness: %s\n",message); exit(1); }
}
static void *symbol(void *h,const char *name)
{
    void *p=dlsym(h,name);
    require(p!=NULL,name);
    return p;
}
static void *openlib(const char *path)
{
    void *h=dlopen(path,RTLD_NOW|RTLD_LOCAL);
    if (!h) { fprintf(stderr,"%s\n",dlerror()); exit(1); }
    return h;
}
static void one_thread(void)
{
    DIR *dir=opendir("/proc/self/task");
    require(dir!=NULL,"thread directory");
    int n=0;
    struct dirent *item;
    while ((item=readdir(dir))) if (item->d_name[0]>='0' && item->d_name[0]<='9') ++n;
    closedir(dir);
    require(n==1,"multiple process threads");
    cpu_set_t set;
    require(!sched_getaffinity(0,sizeof set,&set) && CPU_COUNT(&set)==1,"CPU affinity");
}

struct view {
    float *base,*v;
    size_t bytes,count;
    int r,c,rows,cols,ld;
};
static const uint32_t poison=0x7fc12345;
static struct view make(int rows,int cols,int ld,int r,int c)
{
    require(rows>0 && cols>0 && r>=0 && c>=0 && ld>=cols+c,"invalid view");
    size_t page=(size_t)sysconf(_SC_PAGESIZE);
    size_t bytes=(((size_t)(rows+r+2)*ld*sizeof(float)+page-1)/page)*page;
    unsigned char *p=mmap(NULL,bytes+2*page,PROT_NONE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    require(p!=MAP_FAILED && !mprotect(p+page,bytes,PROT_READ|PROT_WRITE),"protected parent");
    float *base=(float *)(p+page);
    for (size_t i=0;i<bytes/4;++i) memcpy(base+i,&poison,4);
    return (struct view){base,base+(size_t)r*ld+c,bytes,bytes/4,r,c,rows,cols,ld};
}
static size_t inspect(struct view out,const double *expected,double *worst)
{
    size_t guards=0;
    *worst=0;
    for(size_t t=0;t<out.count;++t) {
        int i=(int)(t/out.ld)-out.r, j=(int)(t%out.ld)-out.c;
        if (i<0 || i>=out.rows || j<0 || j>=out.cols) {
            uint32_t word;
            memcpy(&word,out.base+t,4);
            require(word==poison,"output parent changed outside logical view");
            ++guards;
        } else {
            double error=fabs(out.base[t]-expected[(size_t)i*out.cols+j]);
            require(isfinite(error) && error<=0.002,"FP64 result");
            if (error>*worst) *worst=error;
        }
    }
    return guards;
}
static int same(struct call p,struct call q)
{
    return p.m==q.m && p.n==q.n && p.k==q.k && p.lda==q.lda && p.ldb==q.ldb && p.ldc==q.ldc &&
           p.a==q.a && p.b==q.b && p.c==q.c;
}
static void part(const char *role,int id,struct call p,struct call orig)
{
    printf("PART,%s,%d,%d,%d,%d,%d,%d,%d,%zu,%zu,%zu\n",role,id,p.m,p.n,p.k,p.lda,p.ldb,p.ldc,
        (size_t)(p.a-orig.a)/4,(size_t)(p.b-orig.b)/4,(size_t)(p.c-orig.c)/4);
}
int main(int argc,char **argv)
{
    require(argc==18,"expected variant ordinary trace observer actual M N K lda ldb ldc ar ac br bc cr cc");
    int variant=atoi(argv[1]),m=atoi(argv[6]),n=atoi(argv[7]),k=atoi(argv[8]);
    int lda=atoi(argv[9]),ldb=atoi(argv[10]),ldc=atoi(argv[11]);
    require(variant>=0 && variant<=2 && m>0 && m<=1024 && n>0 && n<=8192 && k>0 && k<=2048,"dimensions");
    struct view A=make(m,k,lda,atoi(argv[12]),atoi(argv[13]));
    struct view B=make(k,n,ldb,atoi(argv[14]),atoi(argv[15]));
    struct view C=make(m,n,ldc,atoi(argv[16]),atoi(argv[17]));
    for(int i=0;i<m;++i) for(int p=0;p<k;++p) A.v[(size_t)i*lda+p]=(float)(((i*61+p*19+7)%43)-21)/37.0f;
    for(int p=0;p<k;++p) for(int j=0;j<n;++j) B.v[(size_t)p*ldb+j]=(float)(((p*11+j*29+3)%47)-23)/43.0f;
    require(!mprotect(A.base,A.bytes,PROT_READ) && !mprotect(B.base,B.bytes,PROT_READ),"read-only operands");
    size_t points=(size_t)m*n;
    double *expected=calloc(points,sizeof *expected);
    require(expected!=NULL,"FP64 oracle allocation");
    for (int i=0;i<m;++i) for(int p=0;p<k;++p) {
        double value=(double)A.v[(size_t)i*lda+p];
        for(int j=0;j<n;++j) expected[(size_t)i*n+j]+=value*(double)B.v[(size_t)p*ldb+j];
    }
    one_thread();
    require(!setenv("RANK_PREFIX_MKL",argv[5],1),"set real provider");
    void *ordinary=openlib(argv[2]);
    int (*initialize)(void)=(int (*)(void))symbol(ordinary,"rank_vendor_init");
    gemm_fn normal=(gemm_fn)symbol(ordinary,"rank_vendor_sgemm");
    require(!initialize(),"ordinary provider initialization");
    normal(m,n,k,A.v,lda,B.v,ldb,C.v,ldc);
    one_thread();
    double ordinary_error,trace_error;
    size_t guards=inspect(C,expected,&ordinary_error);
    for(size_t i=0;i<C.count;++i) memcpy(C.base+i,&poison,4);

    require(!setenv("RANK_PREFIX_MKL",argv[4],1) && !setenv("RANK_PREFIX_REAL_MKL",argv[5],1),"set traced provider");
    void *observer=openlib(argv[4]),*traced=openlib(argv[3]);
    initialize=(int (*)(void))symbol(traced,"rank_vendor_init");
    gemm_fn traced_call=(gemm_fn)symbol(traced,"rank_vendor_sgemm");
    int (*observed)(int,struct call *)=(int (*)(int,struct call *))symbol(observer,"rank_observed");
    void (*snapshot)(struct receipt *)=(void (*)(struct receipt *))symbol(traced,"rank_prefix_snapshot");
    require(!initialize(),"traced provider initialization");
    traced_call(m,n,k,A.v,lda,B.v,ldb,C.v,ldc);
    one_thread();
    require(inspect(C,expected,&trace_error)==guards,"trace guard count");
    unsigned a,b,e,d;
    require(__get_cpuid(0,&a,&b,&e,&d),"x86 CPU identification");
    char vendor[13]={0};
    memcpy(vendor,&b,4); memcpy(vendor+4,&d,4); memcpy(vendor+8,&e,4);
    __builtin_cpu_init();
    int capable=!strcmp(vendor,"GenuineIntel") && __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    int eligible=capable && m>=24 && m<=160 && n>=1024 && n<=4096 && k>=96 && k<=384 &&
        m*8<=n && m*k<=49152 && !((uintptr_t)B.v%64);
    struct receipt t;
    snapshot(&t);
    struct call orig=actual(m,n,k,A.v,lda,B.v,ldb,C.v,ldc);
    struct call left=orig,body=orig;
    left.n=8; body.n-=8; body.b+=8*sizeof(float);body.c+=8*sizeof(float);
    int active=eligible && variant!=0;
    require(t.intel_avx2==capable && t.eligible==eligible && same(t.incoming,orig),"observed incoming request");
    require(t.prefix_calls==(active && variant==2),"custom prefix call count");
    if(t.prefix_calls) { require(same(t.prefix,left),"custom prefix pointers, shape and strides"); part("vector-prefix",0,t.prefix,orig); }
    struct call provider[2];
    int count=observed(-1,NULL);
    require(count==1+(active && variant==1),"actual provider call count");
    for(int z=0;z<count;++z) {
        observed(z,&provider[z]);
        struct call want=!active ? orig : (variant==1 && z==0 ? left : body);
        require(same(provider[z],want),"actual provider pointer, shape or stride");
        part("provider",z,provider[z],orig);
    }
    printf("RESULT,%s,%d,%d,%s,%d,%d,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%d,%.10g,%.10g\n",vendor,capable,eligible,
        active ? (variant==1 ? "provider-split" : "vector-prefix") : "forward",
        t.prefix_calls,count,points,guards,orig.a%64,orig.b%64,orig.c%64,
        (size_t)(A.v-A.base),(size_t)(B.v-B.base),(size_t)(C.v-C.base),sched_getcpu(),ordinary_error,trace_error);
    return 0;
}
