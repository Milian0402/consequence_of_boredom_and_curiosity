#define main pair_cli
#include "common-bench.c"
#undef main
#include <libxsmm.h>
#include <pthread/qos.h>
#include <mach/mach.h>
typedef struct {
    const char *name;
    gemm_fn call;
    libxsmm_gemmfunction jit;
    libxsmm_gemm_param args;
    float *out;
    uint64_t first_ns, setup_ns;
    size_t code_size;
    double error;
} entry;
static uint64_t ticks(clockid_t id) {
    struct timespec t; if (clock_gettime(id,&t)) fail("CPU clock");
    return (uint64_t)t.tv_sec*1000000000+t.tv_nsec;
}
static unsigned threads(void) {
    thread_act_array_t t; mach_msg_type_number_t n;
    if (task_threads(mach_task_self(),&t,&n) != KERN_SUCCESS) fail("task_threads");
    for (unsigned i=0;i<n;++i) mach_port_deallocate(mach_task_self(),t[i]);
    vm_deallocate(mach_task_self(),(vm_address_t)t,n*sizeof(*t)); return n;
}
static double verify(int m,int n,int ldc,const float *c,const double *ref) {
    check_output_padding(c,m,n,ldc); double largest=0;
    for(int i=0;i<m;++i) for(int j=0;j<n;++j) {
        double d=fabs(c[(size_t)i*ldc+j]-ref[(size_t)i*n+j]);
        if (!isfinite(d) || d>tolerance) {fprintf(stderr,"FP64 error %d,%d %.12g\n",i,j,d);exit(1);}
        largest=fmax(largest,d);
    } return largest;
}
int main(int argc,char **argv) {
    if (pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED,0)) fail("QoS");
    if (argc<11 || (argc-7)%2) fail("M N K pad seed block [NAME DYLIB]...");
    int m=number(argv[1],1,512),n=number(argv[2],1,8192),k=number(argv[3],1,1024);
    int pad=number(argv[4],0,64),block=number(argv[6],1,2),lda=k+pad,ldb=n+pad,ldc=n+pad;
    uint32_t seed=number(argv[5],1,UINT32_MAX),state=seed;
    size_t sa=(size_t)m*lda*sizeof(float),sb=(size_t)k*ldb*sizeof(float);
    float *a=matrix(m,lda),*b=matrix(k,ldb),*acopy=matrix(m,lda),*bcopy=matrix(k,ldb);
    fill(a,m,k,lda,&state);fill(b,k,n,ldb,&state);memcpy(acopy,a,sa);memcpy(bcopy,b,sb);
    double *ref=calloc((size_t)m*n,sizeof(*ref)); if(!ref)fail("reference allocation");
    for(int i=0;i<m;++i) for(int p=0;p<k;++p) {
        double v=a[(size_t)i*lda+p];
        for(int j=0;j<n;++j) ref[(size_t)i*n+j]+=v*(double)b[(size_t)p*ldb+j];
    }
    entry methods[16]={0}; int count=0;
    const mtmm_config cfg={.num_threads=1};
    for (int id=7;id<argc;id+=2) {
        if(count>=16) fail("too many participants");
        entry *e=methods+count; e->name=argv[id];
        if(strcmp(argv[id+1],"prepared")==0) {
            libxsmm_init();
            const libxsmm_datatype f=LIBXSMM_DATATYPE_F32;
            libxsmm_xmmfunction u; libxsmm_kernel_info info;
            uint64_t start=now_ns();
            u.gemm=libxsmm_dispatch_gemm(libxsmm_create_gemm_shape(n,m,k,ldb,lda,ldc,f,f,f,f),LIBXSMM_GEMM_FLAG_BETA_0,LIBXSMM_GEMM_PREFETCH_NONE);
            e->setup_ns=now_ns()-start;
            if(!u.gemm || libxsmm_get_kernel_info(u.ptr_const,&info) || !info.code_size || info.is_reference_kernel || info.kind!=LIBXSMM_KERNEL_KIND_MATMUL) {
                fprintf(stderr,"JIT unavailable %d %d %d %d %u\n",m,n,k,pad,seed);
                memset(e,0,sizeof(*e));continue;
            } e->jit=u.gemm;e->code_size=info.code_size;
        } else e->call=load(argv[id+1]);
        e->out=count?methods[0].out:matrix(m,ldc);e->args.a.primary=b;e->args.b.primary=a;e->args.c.primary=e->out;
        count++;
    }
    if(count!=3||strcmp(methods[0].name,"A")||strcmp(methods[1].name,"B")||strcmp(methods[2].name,"D"))fail("3 aliases A/B/D expected");
    if(!methods[0].call || methods[0].call!=methods[1].call || methods[0].call==methods[2].call)fail("function identity check failed");
    fprintf(stderr,"A=B; D distinct, %d %d %d %d %u\n",m,n,k,pad,seed);
    int calls=(int)(14000000u/((size_t)m*n*k)); if(calls<1)calls=1;if(calls>8)calls=8;
#define INVOKE(e) ((e)->jit ? (e)->jit(&(e)->args) : (e)->call(m,n,k,a,lda,b,ldb,(e)->out,ldc,&cfg))
    for(int pos=0;pos<count;++pos) {
        entry *e=methods+(block==1?pos:count-1-pos);
        prepare_output(e->out,m,n,ldc);
        uint64_t start=now_ns(); INVOKE(e); e->first_ns=now_ns()-start;
        e->error=verify(m,n,ldc,e->out,ref);
        for(int j=0;j<2;++j)INVOKE(e);
        if(memcmp(acopy,a,sa)||memcmp(bcopy,b,sb))fail("input modified");
        check_guard(a,(size_t)m*lda);check_guard(b,(size_t)k*ldb);
    }
    static const int orders[6][3]={{0,1,2},{1,2,0},{2,0,1},{2,1,0},{1,0,2},{0,2,1}};
    if(methods[0].out!=methods[1].out || methods[1].out!=methods[2].out)fail("shared output required");
    unsigned before=threads();
    if(before!=1)fail("single caller required");
    for(int round=0;round<60;++round) for(int pos=0;pos<count;++pos) {
        int slot=orders[block==1?(round+round/6)%6:5-(round+round/6)%6][pos];
        entry *e=methods+slot;
        unsigned nt0=before; uint64_t proc=ticks(CLOCK_PROCESS_CPUTIME_ID),thr=ticks(CLOCK_THREAD_CPUTIME_ID);
        uint64_t start=now_ns();for(int j=0;j<calls;++j)INVOKE(e);uint64_t wall=now_ns()-start;
        thr=ticks(CLOCK_THREAD_CPUTIME_ID)-thr;proc=ticks(CLOCK_PROCESS_CPUTIME_ID)-proc;
        unsigned nt1=before;
        if(round>=54) {
            e->error=fmax(e->error,verify(m,n,ldc,e->out,ref));
            if(memcmp(acopy,a,sa)||memcmp(bcopy,b,sb))fail("input modified");
            check_guard(a,(size_t)m*lda);check_guard(b,(size_t)k*ldb);
        } else {
            int i=(round*7+pos*11)%m,j=(round*151+pos*97)%n;
            double d=fabs(e->out[(size_t)i*ldc+j]-ref[(size_t)i*n+j]);
            if(!isfinite(d)||d>tolerance)fail("spot check");
        }
        printf("%d,%d,%d,%d,%u,%d,%d,%d,%s,%d,%.12g,%.12g,%.12g,%u,%u,%.8f,%zu\n",m,n,k,pad,seed,block,round+1,pos+1,e->name,calls,wall*1e-9/calls,thr*1e-9/calls,e->error,nt0,nt1,(double)proc/thr,(size_t)(round>=54?m*n:1));
    }
    if(threads()!=1)fail("thread count changed");
    return 0;
}
