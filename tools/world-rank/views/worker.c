/* Reuse the frozen worker's RNG, input distributions, plugin initialization,
   one-task check and clock. Unused previous entry points are removed at link. */
#define main previous_study_main
#include "../worker.c"
#undef main
#include <sys/mman.h>

struct view {
    float *start, *data;
    size_t count, bytes;
    int rows, cols, ld, row, col;
};

static struct view parent(int rows,int cols,int ld,int row,int col,uint32_t bits)
{
    size_t page=(size_t)sysconf(_SC_PAGESIZE);
    if (ld<cols+col || cols<1 || rows<1 || row<0 || col<0) fatal("invalid parent layout");
    size_t bytes=((size_t)(row+rows+2)*ld*sizeof(float)+page-1)/page*page;
    char *map=mmap(NULL,bytes+2*page,PROT_NONE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    if (map==MAP_FAILED || mprotect(map+page,bytes,PROT_READ|PROT_WRITE)) fatal("parent mapping");
    float *start=(float*)(map+page);
    for (size_t i=0;i<bytes/sizeof(float);++i) set_bits(start+i,bits);
    struct view v={start,start+(size_t)row*ld+col,bytes/sizeof(float),bytes,rows,cols,ld,row,col};
    return v;
}

#ifdef RANK_VIEW_REFERENCE
#include "../ragged-hk/golden.h"
#else
static double verify(struct view a,struct view b,struct view c,size_t *guards)
{
    double *row=calloc((size_t)c.cols,sizeof(*row));
    if (!row) fatal("reference allocation");
    double worst=0;
    for (int i=0;i<c.rows;++i) {
        memset(row,0,(size_t)c.cols*sizeof(*row));
        for (int p=0;p<a.cols;++p) {
            const double av=a.data[(size_t)i*a.ld+p];
            const float *bp=b.data+(size_t)p*b.ld;
            for (int j=0;j<c.cols;++j) row[j]+=av*(double)bp[j];
        }
        for (int j=0;j<c.cols;++j) {
            double error=fabs(c.data[(size_t)i*c.ld+j]-row[j]);
            if (!isfinite(error) || error>0.002) fatal("FP64 mismatch");
            if (error>worst) worst=error;
        }
    }
    free(row);
    *guards=0;
    for (size_t i=0;i<c.count;++i) {
        size_t r=i/c.ld, col=i%c.ld;
        if (r>=(size_t)c.row && r<(size_t)(c.row+c.rows) && col>=(size_t)c.col && col<(size_t)(c.col+c.cols)) continue;
        if (get_bits(c.start+i)!=output_guard) fatal("outside C view changed");
        ++*guards;
    }
    return worst;
}

#endif

#ifdef VIEW_AUDIT_HEADER
#include VIEW_AUDIT_HEADER
#endif

int main(int argc,char **argv)
{
    if (argc!=17) fatal("usage: plugin-or-public distribution seed calls M N K lda ldb ldc ar ac br bc cr cc");
    if (!pin_cpu()) fatal("pin or initialization thread");
    rank_sgemm_fn product=cob_sgemm_rowmajor;
#ifdef RANK_VIEW_RECORD_LD
    void *trace_lib=NULL;
    if (strcmp(argv[1],"public")) { trace_lib=plugin(argv[1]); product=selected_sgemm; }
#else
    if (strcmp(argv[1],"public")) { plugin(argv[1]); product=selected_sgemm; }
#endif
    const char *mode=argv[2]; uint32_t seed=number(argv[3]);
    const int calls=number(argv[4]),m=number(argv[5]),n=number(argv[6]),k=number(argv[7]);
#ifdef RANK_VIEW_REFERENCE
    const int k_limit=3840;
#else
    const int k_limit=2048;
#endif
    if (m<=0 || m>4096 || n<=0 || n>8192 || k<=0 || k>k_limit || calls>1024) fatal("invalid dimensions");
    struct view a=parent(m,k,number(argv[8]),number(argv[11]),number(argv[12]),input_guard);
    struct view b=parent(k,n,number(argv[9]),number(argv[13]),number(argv[14]),input_guard);
    struct view c=parent(m,n,number(argv[10]),number(argv[15]),number(argv[16]),output_guard);
    if (!strcmp(mode,"cancel")) {
        cancellation_input(a.data,m,k,a.ld,1); cancellation_input(b.data,k,n,b.ld,0);
    } else if (!strcmp(mode,"random") || !strcmp(mode,"scaled")) {
        for(int i=0;i<m;++i) fill(a.data+(size_t)i*a.ld,k,&seed);
        for(int p=0;p<k;++p) fill(b.data+(size_t)p*b.ld,n,&seed);
        if (!strcmp(mode,"scaled")) {
            for(int i=0;i<m;++i) for(int p=k/2;p<k;++p) a.data[(size_t)i*a.ld+p]*=0x1p-8f;
            for(int p=0;p<k/2;++p) for(int j=0;j<n;++j) b.data[(size_t)p*b.ld+j]*=0x1p-8f;
        }
    } else fatal("unknown distribution");
    if (mprotect(a.start,a.bytes,PROT_READ) || mprotect(b.start,b.bytes,PROT_READ)) fatal("read-only parents");
    for(int i=0;i<m;++i) for(int j=0;j<n;++j) c.data[(size_t)i*c.ld+j]=NAN;
#ifdef RANK_VIEW_REFERENCE
    if (getenv("RAGGED_WRITE_REFERENCE")) { generate_reference(a,b,c,trace_lib); return 0; }
    double cpu_start=thread_clock();
#endif
    double start=seconds();
    product(m,n,k,a.data,a.ld,b.data,b.ld,c.data,c.ld);
    double first=seconds()-start;
#ifdef RANK_VIEW_REFERENCE
    double cpu_first=thread_clock()-cpu_start;
#endif
    if (!single_task()) fatal("thread count after first call");
#ifdef RANK_VIEW_REFERENCE
    cpu_start=thread_clock();
#endif
    start=seconds();
    for (int r=0;r<calls;++r) product(m,n,k,a.data,a.ld,b.data,b.ld,c.data,c.ld);
    double elapsed=seconds()-start;
#ifdef RANK_VIEW_REFERENCE
    double cpu_elapsed=thread_clock()-cpu_start;
    report_hashes(a,b,c,cpu_first,calls ? cpu_elapsed/calls : 0,trace_lib);
#endif
    if (!single_task()) fatal("thread count after batch");
    size_t guards;
    double error=verify(a,b,c,&guards);
#ifdef VIEW_AUDIT_HEADER
    report_audit(argv[1], calls, a, b, c);
#endif
    printf("VIEW,%zu,%zu,%u,%u,%u,%d,1,%.12g,%.12g,%.12g",(size_t)m*n,guards,
           (unsigned)((uintptr_t)a.data%64),(unsigned)((uintptr_t)b.data%64),(unsigned)((uintptr_t)c.data%64),
           sched_getcpu(),error,first,calls ? elapsed/calls : 0);
#ifdef RANK_VIEW_RECORD_LD
    printf(",%d,%d,%d",a.ld,b.ld,c.ld);
#endif
    putchar('\n');
#ifdef RANK_VIEW_RECORD_LD
    if (trace_lib) {
        const char *(*implementation)(void)=NULL;
        void *ptr=dlsym(trace_lib,"rank_onednn_implementation");
        memcpy(&implementation,&ptr,sizeof implementation);
        if (implementation) {
            unsigned long (*plans)(void)=NULL;
            int (*cached)(void)=NULL;
            ptr=symbol(trace_lib,"rank_onednn_plans"); memcpy(&plans,&ptr,sizeof plans);
            ptr=symbol(trace_lib,"rank_onednn_cached_plan"); memcpy(&cached,&ptr,sizeof cached);
            const char *kind=implementation();
            if (!kind || !*kind || !plans()) fatal("oneDNN did not execute a measured primitive");
            printf("DNN,%s,%lu,%d\n",kind,plans(),cached());
        }
    }
    const char *dump=getenv("RANK_VIEW_DIGEST_PATH");
    if (dump && *dump) {
        FILE *f=fopen(dump,"wbx");
        if (!f) fatal("cannot create unique view result");
        for(int i=0;i<m;++i)
            if(fwrite(c.data+(size_t)i*c.ld,sizeof(float),(size_t)n,f)!=(size_t)n) fatal("view result write");
        if(fclose(f)) fatal("view result close");
    }
#endif
    return 0;
}
