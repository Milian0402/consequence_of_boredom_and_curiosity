/* Untimed macOS qualification of the identical dynamic-library entry used by the benchmark. */
#include <dlfcn.h>
#include <math.h>
#include <mach/mach.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/sysctl.h>
#include <unistd.h>

struct mtmm_config { int num_threads; };
typedef void (*product)(int,int,int,const float*,int,const float*,int,float*,int,const struct mtmm_config*);
struct parent { char *map; float *base; size_t bytes, total; };
static const uint32_t guard = 0x7fc12345;
static const struct mtmm_config serial = {1};
static const size_t offsets[3] = {3,5,11};
static uint32_t state = 919412061;
static float sample(void) {
    state = state * 1664525u + 1013904223u;
    return (float)(state >> 8) * 0x1p-24f - .5f;
}
static uint64_t svcr(void) { uint64_t v; __asm__ volatile("mrs %0, SVCR" : "=r"(v)); return v; }
static unsigned threads(void) {
    thread_act_array_t names; mach_msg_type_number_t count;
    if (task_threads(mach_task_self(), &names, &count) != KERN_SUCCESS) abort();
    for (unsigned i=0;i<count;++i) mach_port_deallocate(mach_task_self(),names[i]);
    vm_deallocate(mach_task_self(),(vm_address_t)names,count*sizeof(*names));
    return count;
}
static struct parent allocate(size_t needed) {
    size_t page=(size_t)sysconf(_SC_PAGESIZE), bytes=(needed*4+page-1)/page*page;
    size_t total=bytes+2*page;
    char *p=mmap(NULL,total,PROT_NONE,MAP_PRIVATE|MAP_ANON,-1,0);
    if (p==MAP_FAILED || mprotect(p+page,bytes,PROT_READ|PROT_WRITE)) abort();
    for(size_t i=0;i<bytes/4;++i) memcpy(p+page+i*4,&guard,4);
    return (struct parent){p,(float*)(p+page),bytes,total};
}
static void release(struct parent p) { if(munmap(p.map,p.total)) abort(); }
static int feature(const char *key) {
    int on=0; size_t size=sizeof(on); return sysctlbyname(key,&on,&size,NULL,0)==0 && on;
}
int main(int argc,char **argv) {
    if(argc!=2 || !feature("hw.optional.arm.FEAT_SME2") || !feature("hw.optional.arm.SME_F32F32")) return 77;
    void *lib=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL); if(!lib) {fprintf(stderr,"%s\n",dlerror());return 2;}
    product gemm=(product)dlsym(lib,"mtmm_sgemm_rowmajor"); size_t (*length)(void)=dlsym(lib,"kleidi_streaming_vector_bytes");
    if(!gemm||!length||svcr()!=0||threads()!=1) return 3;
    const int dims[][3]={{1,1,1},{3,31,2},{16,17,15},{17,65,13},{24,127,95},{25,79,97},
                         {31,97,33},{32,64,64},{33,33,37},{48,49,41},{65,17,19},{24,1027,97},{3,17,0}};
    size_t checked=0,guards=0,methods=0; double worst=0;
    for(size_t s=0;s<sizeof(dims)/sizeof(*dims);++s) for(int mode=0;mode<3;++mode) {
        int m=dims[s][0],n=dims[s][1],k=dims[s][2],lda=k+7,ldb=n+17,ldc=n+11;
        struct parent ap=allocate(offsets[0]+(size_t)m*lda), bp=allocate(offsets[1]+(size_t)(k+1)*ldb), cp=allocate(offsets[2]+(size_t)m*ldc);
        float *a=ap.base+offsets[0],*b=bp.base+offsets[1],*c=cp.base+offsets[2];
        for(int i=0;i<m;++i) for(int q=0;q<k;++q) {
            float x=sample(); a[(size_t)i*lda+q]=mode==0?x:mode==1?(q%2?1.f:-1.f)*(1.f+x*0x1p-12f):ldexpf(x,(q%9)-4);
        }
        for(int q=0;q<k;++q) for(int j=0;j<n;++j) {
            float x=sample(); b[(size_t)q*ldb+j]=mode==0?x:mode==1?1.f+x*0x1p-12f:ldexpf(x,4-(q%9));
        }
        if(mprotect(ap.base,ap.bytes,PROT_READ)||mprotect(bp.base,bp.bytes,PROT_READ)) return 4;
        gemm(m,n,k,a,lda,b,ldb,c,ldc,&serial);
        if(svcr()!=0||threads()!=1) return 5;
        for(size_t z=0;z<cp.bytes/4;++z) {
            size_t rel=z-offsets[2],i=rel/(size_t)ldc,j=rel%(size_t)ldc;
            if(z>=offsets[2]&&i<(size_t)m&&j<(size_t)n) {
                double ref=0;for(int q=0;q<k;++q) ref+=(double)a[i*lda+q]*(double)b[(size_t)q*ldb+j];
                double got=cp.base[z],err=fabs(ref-got);if(!isfinite(got)||err>0.002) {fprintf(stderr,"m,n,k %d,%d,%d at %zu,%zu %.9g %.9g\n",m,n,k,i,j,got,ref);return 6;}
                if(err>worst)worst=err;++checked;
            }else{uint32_t bits;memcpy(&bits,cp.base+z,4);if(bits!=guard) return 7;++guards;}
        }
        ++methods;release(ap);release(bp);release(cp);
    }
    gemm(0,7,11,NULL,11,NULL,7,NULL,7,&serial);gemm(5,0,11,NULL,11,NULL,0,NULL,0,&serial);
    printf("cases=%zu,fp64_cells=%zu,c_guard_cells=%zu,max_abs_error=%.9g,svl_bytes=%zu,svcr_after=%llu,threads_before_after=1\n",methods,checked,guards,worst,length(),(unsigned long long)svcr());
    dlclose(lib);return 0;
}
