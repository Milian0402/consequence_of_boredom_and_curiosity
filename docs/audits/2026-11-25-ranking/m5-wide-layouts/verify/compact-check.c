#define _DARWIN_C_SOURCE
#include <dlfcn.h>
#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

typedef void (*gemm_fn)(int, int, int, const float *, int, const float *, int, float *, int);
typedef uint64_t (*counter_fn)(void);
typedef struct {
    float *data, *area;
    void *map;
    size_t len, area_bytes, total_bytes;
} guarded;
typedef struct { int m,n,k,pa,pb,pc; const char *name; } shape;
typedef struct { double max_error; size_t over, nonfinite, outside_bound; } stats;
enum fixture { NORMAL, DYADIC, CANCEL, RECIPROCAL, CORR16, CORR32, SHARED };
static const char *names[]={"normalized","exact-dyadic","cancel","reciprocal80","correlated16","correlated32","shared-readonly"};
static const float sentinel = -21875.25f;

static void die(const char *message) { fprintf(stderr,"%s\n",message); exit(2); }
static guarded buffer(size_t n, int tail)
{
    size_t pg=(size_t)sysconf(_SC_PAGESIZE), bytes=n*sizeof(float);
    size_t rounded=(bytes+pg-1)/pg*pg, total=2*pg+rounded;
    char *map=mmap(NULL,total,PROT_NONE,MAP_PRIVATE|MAP_ANON,-1,0);
    if(map==MAP_FAILED || mprotect(map+pg,rounded,PROT_READ|PROT_WRITE)!=0) die("guard allocation");
    float *area=(float *)(map+pg);
    for(size_t i=0;i<rounded/sizeof(float);++i) area[i]=sentinel;
    return (guarded){area+(tail?(rounded-bytes)/sizeof(float):0), area, map, n, rounded,total};
}
static void readonly(guarded g)
{
    if(mprotect(g.area,g.area_bytes,PROT_READ)!=0) die("input protection");
}
static int guard_ok(guarded g)
{
    for(const float *p=g.area;p<g.data;++p) if(*p!=sentinel)return 0;
    for(const float *p=g.data+g.len;p<g.area+g.area_bytes/sizeof(float);++p) if(*p!=sentinel)return 0;
    return 1;
}
static void release(guarded g) { if(munmap(g.map,g.total_bytes))die("unmap"); }
static uint32_t rnd(uint32_t *s) { uint32_t x=*s; x^=x<<13; x^=x>>17; x^=x<<5;return *s=x; }
static float draw(uint32_t *state, enum fixture mode, int p, int left)
{
    uint32_t r=rnd(state);
    float x=ldexpf((float)((int)(r>>8)-8388608),-23);
    switch(mode){
      case DYADIC:return (float)((int)(r%31)-15)*0x1p-3f;
      case CANCEL:return (1.0f+x*0x1p-14f)*((left && (p&1))?-1.0f:1.0f);
      case RECIPROCAL:return ldexpf(x,(left?1:-1)*(p*47%161-80));
      case CORR16:return 16.0f*x;
      case CORR32:return 32.0f*x;
      default:return x;
    }
}
static void *lookup(void *lib,const char *symbol)
{
    dlerror();
    void *p=dlsym(lib,symbol);
    const char *error=dlerror();
    if(error){fprintf(stderr,"%s: %s\n",symbol,error);exit(2);}
    return p;
}
static void *library(const char *path, gemm_fn *fn)
{
    void *lib=dlopen(path, RTLD_NOW|RTLD_LOCAL|RTLD_FIRST);
    if(!lib)die(dlerror());
    void *p=lookup(lib,"cob_sgemm_rowmajor");memcpy(fn,&p,sizeof(*fn));return lib;
}
static int eligible(const shape s)
{
    return s.m>=24 && s.m<=55 && s.n>=1024 && s.n<=4096 && s.k>=96 && s.k<=384 &&
           s.pa==0 && s.pb==0 && s.pc==0;
}
static void record(stats *s,double error,double limit)
{
    if(!isfinite(error))++s->nonfinite;
    else{
        if(error>s->max_error)s->max_error=error;
        if(error>.002)++s->over;
        if(error>limit)++s->outside_bound;
    }
}
static int check(shape g, enum fixture mode, int tail, uint32_t seed,
                 const gemm_fn *fn, size_t nimpl, counter_fn counter)
{
    const int m=g.m,n=g.n,k=g.k,lda=k+g.pa,ldb=n+g.pb,ldc=n+g.pc;
    const size_t cells=(size_t)m*n;
    if(mode==SHARED && ((size_t)m*lda>(size_t)k*ldb))die("shared backing");
    guarded a=buffer((size_t)m*lda,tail), b=buffer((size_t)k*ldb,tail), c[3];
    for(size_t t=0;t<nimpl;++t)c[t]=buffer((size_t)m*ldc,tail);
    uint32_t state=seed;
    for(int i=0;i<m;++i)for(int p=0;p<k;++p)a.data[(size_t)i*lda+p]=draw(&state,mode,p,1);
    for(int p=0;p<k;++p)for(int j=0;j<n;++j)
        b.data[(size_t)p*ldb+j]=mode==CORR16 || mode==CORR32?
            a.data[(size_t)(j%m)*lda+p]:draw(&state,mode,p,0);
    const float *aa=mode==SHARED?b.data:a.data;
    readonly(a);readonly(b);
    double *reference=calloc(cells,sizeof(*reference)),*magnitude=calloc(cells,sizeof(*magnitude));
    if(!reference || !magnitude)die("oracle allocation");
    for(int i=0;i<m;++i)for(int p=0;p<k;++p){
        const double av=aa[(size_t)i*lda+p];
        for(int j=0;j<n;++j){
            const double term=av*(double)b.data[(size_t)p*ldb+j];
            const size_t at=(size_t)i*n+j;
            reference[at]+=term;magnitude[at]+=fabs(term);
        }
    }
    int route=-1,canaries=1;
    for(size_t t=0;t<nimpl;++t){
        for(int i=0;i<m;++i)for(int j=0;j<n;++j)c[t].data[(size_t)i*ldc+j]=NAN;
        uint64_t before=counter?counter():0;
        fn[t](m,n,k,aa,lda,b.data,ldb,c[t].data,ldc);
        if(t==2 && counter)route=(int)(counter()-before);
        canaries&=guard_ok(c[t]);
        for(int i=0;i<m;++i)for(int j=n;j<ldc;++j)canaries&=(c[t].data[(size_t)i*ldc+j]==sentinel);
    }
    const double theta=2.0*k*(FLT_EPSILON/2.0);
    /* FP64 absolute-product sum and reference each have up to k additions of error. */
    const double multiplier=(theta/(1.0-theta)+2.0*k*DBL_EPSILON)/(1.0-k*DBL_EPSILON);
    stats acc[2]={{0}};
    size_t different=0,trace_diff=0,over_floor=0,new_over=0,removed_over=0;
    double max_floor=0;
    for(int i=0;i<m;++i)for(int j=0;j<n;++j){
        size_t d=(size_t)i*n+j,o=(size_t)i*ldc+j;
        const double v=reference[d],limit=multiplier*magnitude[d];
        const double ex=fabs((double)c[0].data[o]-v),ey=fabs((double)c[1].data[o]-v);
        double floor=fabs((double)(float)v-v);
        record(&acc[0],ex,limit);record(&acc[1],ey,limit);
        if(floor>max_floor)max_floor=floor;
        over_floor+=(floor>.002);
        new_over+=(ey>.002&&ex<=.002);removed_over+=(ey<=.002&&ex>.002);
        different+=(memcmp(c[0].data+o,c[1].data+o,sizeof(float))!=0);
        if(nimpl==3)trace_diff+=(memcmp(c[1].data+o,c[2].data+o,sizeof(float))!=0);
    }
    int ok=canaries && !acc[0].nonfinite && !acc[1].nonfinite &&
        !acc[0].outside_bound&&!acc[1].outside_bound && !trace_diff &&
        (eligible(g)||different==0) && (!counter||route==eligible(g));
    if(mode==DYADIC)ok&=(acc[0].max_error==0 && acc[1].max_error==0);
    if(mode!=CORR16 && mode!=CORR32)ok&=(acc[0].over==0 && acc[1].over==0);
    printf("%s,%d,%d,%d,%d,%d,%d,%s,%s,%"PRIu32",%zu,%d,%d,%.12g,%.12g,%.12g,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%d,%d\n",
        g.name,m,n,k,lda,ldb,ldc,names[mode],tail?"trailing":"leading",seed,cells,eligible(g),route,
        acc[0].max_error,acc[1].max_error,max_floor,acc[0].over,acc[1].over,over_floor,new_over,removed_over,
        acc[0].outside_bound+acc[1].outside_bound,different,trace_diff,canaries,ok);
    free(reference);free(magnitude);release(a);release(b);
    for(size_t t=0;t<nimpl;++t)release(c[t]);
    return ok;
}
int main(int argc,char **argv)
{
    if(argc!=3&&argc!=5){fprintf(stderr,"usage: %s BASE.dylib CANDIDATE.dylib [TRACE.dylib COUNTER]\n",argv[0]);return 2;}
    gemm_fn fn[3];library(argv[1],&fn[0]);library(argv[2],&fn[1]);counter_fn counter=NULL;
    size_t nimpl=2;
    if(argc==5){
        void *lib=library(argv[3],&fn[2]),*p=lookup(lib,argv[4]);
        memcpy(&counter,&p,sizeof(counter));nimpl=3;
    }
    const shape multiple[]={
        {24,1024,96,0,0,0,"lower-corner"}, {31,1087,127,0,0,0,"three-fringes"},
        {39,2304,160,0,0,0,"historical-compact"}, {48,1088,192,0,0,0,"complete-row-tiles"},
        {49,1153,193,0,0,0,"partial-final-row-tile"}, {55,4096,384,0,0,0,"upper-corner"}};
    const shape control[]={
        {23,1057,97,0,0,0,"below-m"}, {56,1057,97,0,0,0,"above-m"},
        {29,1023,97,0,0,0,"below-n"}, {29,4097,97,0,0,0,"above-n"},
        {29,1057,95,0,0,0,"below-k"}, {29,1057,385,0,0,0,"above-k"},
        {39,2304,160,3,0,0,"padded-a"}, {39,2304,160,0,5,0,"padded-b"},
        {39,2304,160,0,0,7,"padded-c"}, {41,1123,137,1,3,5,"three-strides"}};
    puts("layout,m,n,k,lda,ldb,ldc,fixture,boundary,seed,cells,eligible,trace_delta,max_abs_base,max_abs_candidate,max_abs_nearest,over_base,over_candidate,over_nearest,new_over_candidate,removed_over_candidate,over_forward_bound,nonidentical_to_base,nonidentical_to_trace,canaries_ok,pass");
    int ok=1;
    for(size_t i=0;i<sizeof(multiple)/sizeof(*multiple);++i){
        for(int t=0;t<2;++t)for(int q=NORMAL;q<=CORR32;++q){
            uint32_t seed=143460677u+(uint32_t)i*9397u+(uint32_t)q*2399u+t*572245342u;
            ok&=check(multiple[i],q,t,seed,fn,nimpl,counter);
        }
    }
    for(size_t i=0;i<sizeof(control)/sizeof(*control);++i)for(int t=0;t<2;++t)
        ok&=check(control[i],NORMAL,t,74571713u+6719u*(uint32_t)i+233449361u*t,fn,nimpl,counter);
    for(int t=0;t<2;++t)ok&=check(multiple[2],SHARED,t,56127783u+738698443u*t,fn,nimpl,counter);
    return ok?0:1;
}
