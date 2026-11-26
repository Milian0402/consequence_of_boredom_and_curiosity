/* Use identical readonly cropped parents and affinity checks as the timed workers. */
#define main exact_observation_worker_main
#include "../driver/worker.c"
#undef main

int main(int argc, char **argv)
{
    if (argc != 15) fail("usage: raw|checked module M N K LDA LDB LDC AR AC BR BC CR CC");
    const int checked = strcmp(argv[1], "checked") == 0;
    const int cpu = native_cpu(), m=number(argv[3]), n=number(argv[4]), k=number(argv[5]);
    view a=parent(m,k,number(argv[6]),number(argv[9]),number(argv[10]),0x7fc12345u);
    view b=parent(k,n,number(argv[7]),number(argv[11]),number(argv[12]),0x7fc12345u);
    view c=parent(m,n,number(argv[8]),number(argv[13]),number(argv[14]),0xc7428000u);
    for(int i=0;i<m;i++) for(int p=0;p<k;p++) a.v[(size_t)i*a.ld+p]=(float)((i%5-2)*(p%2?2:-2));
    for(int p=0;p<k;p++) for(int j=0;j<n;j++) b.v[(size_t)p*b.ld+j]=(float)((j%7-3)*(p%2?1:-1));
    protect(a); protect(b); poison(c);
    void *lib=dlopen(argv[2],RTLD_NOW|RTLD_LOCAL);
    if(!lib) fail(dlerror());
    int (*init)(void)=lookup(lib,"rank_vendor_init");
    sgemm_fn call=lookup(lib,"rank_vendor_sgemm");
    if(init() || native_cpu()!=cpu) fail("engine initialization or affinity");
    call(m,n,k,a.v,a.ld,b.v,b.ld,c.v,c.ld);
    if(native_cpu()!=cpu) fail("changed CPU or threads");
    size_t seen=0, guards=0;
    for(size_t i=0;i<c.count;i++){
        size_t r=i/c.ld,col=i%c.ld;
        if(r>=(size_t)c.row && r<(size_t)(c.row+m) && col>=(size_t)c.col && col<(size_t)(c.col+n)){
            int row=(int)r-c.row,column=(int)col-c.col;
            if(c.start[i]!=(float)(2*(row%5-2)*(column%7-3)*k)) fail("wrong SGEMM result");
            seen++;
        }else{
            uint32_t sentinel;memcpy(&sentinel,c.start+i,4);
            if(sentinel!=0xc7428000u) fail("overwritten output guard");
            guards++;
        }
    }
    receipt(lib,"PROBE");
    printf("PROOF\t%s\t%zu\t%zu\t%d\t1\treadonly\t%zu\t%zu\t%zu\n",checked?"checked":"raw",seen,guards,cpu,
           (size_t)((uintptr_t)a.v%64),(size_t)((uintptr_t)b.v%64),(size_t)((uintptr_t)c.v%64));
    return 0;
}
