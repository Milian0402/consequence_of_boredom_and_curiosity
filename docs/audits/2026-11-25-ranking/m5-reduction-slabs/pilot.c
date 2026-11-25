#define main ignore_main
#include "common-bench.c"
#undef main
#include <sys/mman.h>
#include <unistd.h>
#include <sys/resource.h>
#include <mach/mach.h>
#include <pthread/qos.h>
#include <libxsmm.h>
enum { methods=3 };
typedef struct { float *mem; char *vec; size_t bytes,pages,stride,logical; } arena;
typedef struct { const char *name; gemm_fn fn; libxsmm_gemmfunction jit; double maximum; } method;
static const float canary=-29073.0f;
static size_t multiple(size_t x,size_t unit){return (x+unit-1)/unit*unit;}
static arena alloc_pool(size_t logical,int slots,size_t pg) {
    arena z={0};z.logical=logical;z.stride=multiple(logical+64,32);
    z.bytes=multiple(z.stride*(size_t)slots*sizeof(float),pg);z.pages=z.bytes/pg;
    z.mem=mmap(NULL,z.bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANON,-1,0);
    z.vec=calloc(z.pages,1);if(z.mem==MAP_FAILED||!z.vec)fail("arena allocation");
    for(size_t q=0;q<z.bytes/sizeof(float);++q)z.mem[q]=canary;
    return z;
}
static size_t resident(arena z) {
    if(mincore(z.mem,z.bytes,z.vec))fail("mincore");
    size_t total=0;for(size_t p=0;p<z.pages;++p)total+=(z.vec[p]&1)!=0;return total;
}
static uint32_t next(uint32_t *state) { *state=*state*1664525u+1013904223u;return *state; }
static uint64_t initialize(float *v,size_t len,uint32_t *state) {
    uint64_t hash=UINT64_C(14695981039346656037);
    for(size_t i=0;i<len;++i){v[i]=(float)((int)(next(state)>>8)-8388608)/8388608.0f;
        uint32_t bits;memcpy(&bits,v+i,sizeof(bits));hash^=bits;hash*=UINT64_C(1099511628211);}
    return hash;
}
static void unique(const uint64_t *s,int slots) {
    for(int i=0;i<slots;++i)for(int j=0;j<i;++j)if(s[i]==s[j]) fail("duplicate input content");
}
static void guards(arena z,int slot) {
    const float *v=z.mem+(size_t)slot*z.stride;
    for(size_t q=z.logical;q<z.stride;++q)if(v[q]!=canary)fail("slot guard");
}
static unsigned thread_count(void) {
    thread_act_array_t ls;mach_msg_type_number_t n;
    if(task_threads(mach_task_self(),&ls,&n)!=KERN_SUCCESS)fail("task_threads");
    for(unsigned j=0;j<n;++j)mach_port_deallocate(mach_task_self(),ls[j]);
    vm_deallocate(mach_task_self(),(vm_address_t)ls,n*sizeof(*ls));return n;
}
static uint64_t clock_ns(clockid_t clock) {
    struct timespec ts;if(clock_gettime(clock,&ts))fail("clock");
    return (uint64_t)ts.tv_sec*1000000000+ts.tv_nsec;
}
static int gcd(int a,int b){while(b){int c=a%b;a=b;b=c;}return a;}
static FILE *output(const char *root,const char *name){
    char p[4096];if(snprintf(p,sizeof(p),"%s/%s",root,name)>=(int)sizeof(p))fail("output path");
    FILE *f=fopen(p,"a");if(!f)fail("output file");return f;
}
int main(int argc,char **argv) {
    if(argc!=10+2*methods)fail("family M N K pad seed block ordinal outdir [method dylib]...");
    if(pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED,0))fail("QoS");
    const char *family=argv[1],*root=argv[9];const int m=number(argv[2],1,64),n=number(argv[3],1,8192),k=number(argv[4],1,512),pad=number(argv[5],0,0);
    const uint32_t seed=(uint32_t)number(argv[6],1,UINT32_MAX);uint32_t rng=seed;
    const int block=number(argv[7],1,2),ordinal=number(argv[8],0,8);
    const size_t ma=(size_t)m*k,mb=(size_t)k*n,mc=(size_t)m*n;
    const size_t slot_bytes=sizeof(float)*(ma+mb+mc),pg=(size_t)sysconf(_SC_PAGESIZE);
    const int slots=(int)(((size_t)134217728+slot_bytes-1)/slot_bytes);
    arena A=alloc_pool(ma,slots,pg),B=alloc_pool(mb,slots,pg),C=alloc_pool(mc,slots,pg);
    if(mlock(A.mem,A.bytes)||mlock(B.mem,B.bytes)||mlock(C.mem,C.bytes)){
        perror("unable to wire matrix pool");return 4;
    }
    uint64_t *ha=calloc(slots,sizeof(*ha)),*hb=calloc(slots,sizeof(*hb));
    libxsmm_gemm_param *params=calloc(slots,sizeof(*params));
    double *ref=calloc(mc,sizeof(*ref));int *order=calloc(slots,sizeof(*order));
    enum { probes=8 };size_t *probe_index=calloc((size_t)slots*probes,sizeof(*probe_index));double *probe_value=calloc((size_t)slots*probes,sizeof(*probe_value));
    if(!ha||!hb||!params||!ref||!order||!probe_index||!probe_value)fail("allocation");
    for(int s=0;s<slots;++s){
        float *a=A.mem+s*A.stride,*b=B.mem+s*B.stride,*c=C.mem+s*C.stride;
        ha[s]=initialize(a,ma,&rng);hb[s]=initialize(b,mb,&rng);
        for(size_t q=0;q<mc;++q)c[q]=NAN;
        params[s].a.primary=b;params[s].b.primary=a;params[s].c.primary=c;
    }
    unique(ha,slots);unique(hb,slots);
    if(mprotect(A.mem,A.bytes,PROT_READ)||mprotect(B.mem,B.bytes,PROT_READ))fail("read-only operands");
    method impl[methods]={0};const mtmm_config cfg={.num_threads=1};
    for(int i=0;i<methods;++i) {
        method *e=impl+i;e->name=argv[10+2*i];const char *lib=argv[11+2*i];
        if(!strcmp(lib,"prepared")){
            const libxsmm_datatype fp=LIBXSMM_DATATYPE_F32;libxsmm_init();
            libxsmm_xmmfunction u;libxsmm_kernel_info info;
            u.gemm=libxsmm_dispatch_gemm(libxsmm_create_gemm_shape(n,m,k,n,k,n,fp,fp,fp,fp),LIBXSMM_GEMM_FLAG_BETA_0,LIBXSMM_GEMM_PREFETCH_NONE);
            if(!u.gemm||libxsmm_get_kernel_info(u.ptr_const,&info)||!info.code_size||info.is_reference_kernel||info.kind!=LIBXSMM_KERNEL_KIND_MATMUL)fail("prepared JIT");
            e->jit=u.gemm;fprintf(stderr,"jit %s %zu\n",family,info.code_size);
        }else e->fn=load(lib);
    }
    if(!impl[0].fn||impl[0].fn!=impl[1].fn||impl[0].fn==impl[2].fn)fail("identical alias witness");
#define RUN(e,s) ((e)->jit?(e)->jit(&params[s]):(e)->fn(m,n,k,A.mem+(s)*A.stride,k,B.mem+(s)*B.stride,n,C.mem+(s)*C.stride,n,&cfg))
    FILE *acc=output(root,"accuracy.csv"),*tim=output(root,"samples.csv"),*meta=output(root,"pool.csv");
    float *prior_fp32=calloc(mc,sizeof(*prior_fp32));if(!prior_fp32)fail("temporary FP32 result");
    for(int s=0;s<slots;++s) {
        const float *a=A.mem+s*A.stride,*b=B.mem+s*B.stride;float *c=C.mem+s*C.stride;
        memset(ref,0,mc*sizeof(*ref));
        for(int i=0;i<m;++i)for(int j=0;j<k;++j) {
            double left=a[(size_t)i*k+j];double *out=ref+(size_t)i*n;
            for(int q=0;q<n;++q)out[q]+=left*(double)b[(size_t)j*n+q];
        }
        for(int q=0;q<probes;++q){size_t id=(size_t)s*probes+q;probe_index[id]=(size_t)next(&rng)%mc;probe_value[id]=ref[probe_index[id]];}
        for(int i=0;i<methods;++i){
            for(size_t q=0;q<mc;++q)c[q]=NAN;
            method *e=impl+i;RUN(e,s);double maximum=0;
            for(size_t q=0;q<mc;++q){double d=fabs(ref[q]-c[q]);if(!isfinite(d)||d>.002){fprintf(stderr,"FP64 %s/%s slot %d cell %zu err %.12g\n",family,e->name,s,q,d);return 3;}maximum=fmax(maximum,d);}
            if(i==0)memcpy(prior_fp32,c,mc*sizeof(*prior_fp32));
            else if(memcmp(prior_fp32,c,mc*sizeof(*c)))fail("nonidentical ordered FP32 result");
            e->maximum=fmax(e->maximum,maximum);guards(C,s);
        }
        guards(A,s);guards(B,s);
    }
    for(int i=0;i<methods;++i)fprintf(acc,"%s,%d,%d,%d,%d,%u,%d,%s,%d,%zu,%.12g,1\n",family,m,n,k,pad,seed,block,impl[i].name,slots,mc*(size_t)slots,impl[i].maximum);
    fclose(acc);
    const size_t pages=A.pages+B.pages+C.pages;
#define RESIDENT() (resident(A)+resident(B)+resident(C))
    const size_t before=RESIDENT();size_t lowest=before;const unsigned nt=thread_count();if(nt!=1)fail("worker count");
    for(int mode_id=0;mode_id<2;++mode_id) {
        const int rotate=(mode_id+ordinal+block-1)%2;
        struct rusage prior,after_epoch;
        if(getrusage(RUSAGE_SELF,&prior))fail("usage before");
        for(int round=0;round<18;++round){
            const int start=(seed%(unsigned)slots+17*round)%slots;int step=0;
            if(rotate){step=(seed%(unsigned)slots+2*round+1)%slots;while(!step||gcd(step,slots)!=1)step=(step+1)%slots;}
            int slot=start;for(int q=0;q<slots;++q){order[q]=slot;slot=(slot+step)%slots;}
            for(int pos=0;pos<methods;++pos) {
                const int x=((round/methods+block-1)%2==0)?pos:methods-1-pos;const int delta=x==0?0:x%2?(x+1)/2:methods-x/2;
                method *e=impl+(delta+round)%methods;
                const unsigned nt0=thread_count();
                uint64_t cpu=clock_ns(CLOCK_THREAD_CPUTIME_ID),wall=now_ns();
                for(int q=0;q<slots;++q){int s=order[q];RUN(e,s);}
                wall=now_ns()-wall;cpu=clock_ns(CLOCK_THREAD_CPUTIME_ID)-cpu;
                const unsigned nt1=thread_count();if(nt0!=1||nt1!=1)fail("thread count changed");
                for(int q=0;q<(rotate?slots:1);++q){int s=order[q];float *c=C.mem+s*C.stride;size_t id=(size_t)s*probes+round%probes;double diff=fabs((double)c[probe_index[id]]-probe_value[id]);if(!isfinite(diff)||diff>.002)fail("post-timed FP64");guards(C,s);}
                fprintf(tim,"%s,%d,%d,%d,%d,%u,%d,%s,%d,%d,%s,%d,%d,%d,%d,%llu,%llu,%u,%u,1\n",family,m,n,k,pad,seed,block,rotate?"rotate":"hot",round+1,pos+1,e->name,slots,slots,start,step,(unsigned long long)wall,(unsigned long long)cpu,nt0,nt1);
            }
            size_t cur=RESIDENT();if(cur<lowest)lowest=cur;
        }
        if(getrusage(RUSAGE_SELF,&after_epoch))fail("usage after");
        FILE *faults=output(root,"faults.csv");
        fprintf(faults,"%s,%u,%d,%s,%ld,%ld,1\n",family,seed,block,rotate?"rotate":"hot",after_epoch.ru_majflt-prior.ru_majflt,after_epoch.ru_minflt-prior.ru_minflt);
        fclose(faults);
    }
    const size_t after=RESIDENT();if(after<lowest)lowest=after;
    fprintf(meta,"%s,%d,%d,%d,%d,%u,%d,%d,%zu,%zu,%zu,%zu,%zu,%zu,%d,%d,%d,1,1,1,%u,%zu\n",family,m,n,k,pad,seed,block,slots,slot_bytes*slots,A.bytes+B.bytes+C.bytes,pg,pages,before,after,slots,slots,slots,nt,lowest);
    if(lowest!=pages)fail("wired pages missing");
    fclose(meta);fclose(tim);
    return 0;
}
