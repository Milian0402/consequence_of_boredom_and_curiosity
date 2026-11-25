#include <oneapi/dnnl/dnnl.h>
#include <dlfcn.h>
#include <dirent.h>
#include <sched.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cfenv>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

static void require(bool ok, const char *why) {
    if (!ok) { std::fprintf(stderr,"%s\n",why); std::exit(1); }
}
static int threads() {
    DIR *d=opendir("/proc/self/task"); require(d,"cannot inspect threads");
    int n=0; while (auto *e=readdir(d)) { if (e->d_name[0]!='.') ++n; }
    closedir(d); return n;
}
template<class F> static F symbol(void *h,const char *name) {
    auto addr=dlsym(h,name); require(addr,name); return reinterpret_cast<F>(addr);
}
static uint32_t state;
static float sample() {
    state=state*1664525u+1013904223u;
    return (int32_t((state>>8) & 65535)-32768)/32768.0f;
}
int main(int argc,char **argv) {
    require(argc==2,"usage: check /path/adapter.so");
    cpu_set_t set; CPU_ZERO(&set);
    require(sched_getaffinity(0,sizeof(set),&set)==0 && CPU_COUNT(&set)==1,"pin process to one CPU");
    void *h=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL); if(!h) { std::fprintf(stderr,"%s\n",dlerror()); return 1; }
    auto init=symbol<int(*)()>(h,"rank_vendor_init");
    using fn=void(*)(int,int,int,const float*,int,const float*,int,float*,int);
    auto mul=symbol<fn>(h,"rank_vendor_sgemm");
    auto impl=symbol<const char*(*)()>(h,"rank_onednn_implementation");
    auto plans=symbol<unsigned long(*)()>(h,"rank_onednn_plans");
    auto cached=symbol<int(*)()>(h,"rank_onednn_cached_plan");
    auto version=symbol<const dnnl_version_t*(*)()>(h,"rank_onednn_version");
    require(threads()==1 && init()==0,"initialization failed or extra threads before init");
    auto v=version();
    Dl_info mapping{}; require(dladdr(v,&mapping)!=0,"cannot resolve mapped libdnnl");
    std::printf("runtime=%d.%d.%d cpu_runtime=%u gpu_runtime=%u cached_plan=%d affinity=%d lib=%s\n",
        v->major,v->minor,v->patch,v->cpu_runtime,v->gpu_runtime,cached(),CPU_COUNT(&set),mapping.dli_fname);
    std::array<std::array<int,6>,5> shapes{{{1,17,9,11,23,25},{16,16,16,16,16,16},
        {67,131,93,100,136,142},{67,131,93,101,139,143},{97,193,159,163,199,202}}};
    constexpr int guards=19;
    const float poison=std::numeric_limits<float>::quiet_NaN();
    size_t total=0;
    for(const auto &s:shapes) {
        auto [m,n,k,la,lb,lc]=s;
        // Different addresses and contents on each invocation of a repeated plan.
        std::vector<float> aa[2],bb[2],cc[2];
        for(int run=0;run<2;++run) {
            auto &a=aa[run],&b=bb[run],&c=cc[run];
            a.assign(guards+m*la+guards,poison); b.assign(guards+k*lb+guards,poison);
            c.assign(guards+m*lc+guards,12345.5f);
            state=0x3b3456e3u+run+7*k+la;
            for(int i=0;i<m;++i) for(int j=0;j<k;++j) a[guards+i*la+j]=sample();
            for(int i=0;i<k;++i) for(int j=0;j<n;++j) b[guards+i*lb+j]=sample();
            for(int i=0;i<m;++i) std::fill_n(c.data()+guards+i*lc,n,poison);
            const auto a0=a,b0=b;
            auto start=plans();
            mul(m,n,k,a.data()+guards,la,b.data()+guards,lb,c.data()+guards,lc);
            require(plans()-start==static_cast<unsigned long>(!cached()||!run),"wrong primitive-cache behavior");
            require(!std::memcmp(a.data(),a0.data(),a.size()*sizeof(float)) &&
                    !std::memcmp(b.data(),b0.data(),b.size()*sizeof(float)),"inputs modified");
            double worst=0;
            for(size_t id=0;id<c.size();++id) {
                if(id<guards || id>=guards+(size_t)m*lc || (id-guards)%lc>=(size_t)n) {
                    require(c[id]==12345.5f,"C padding/guard modified"); continue;
                }
                int i=(id-guards)/lc,j=(id-guards)%lc;
                double sum=0; for(int p=0;p<k;++p) sum+=double(a[guards+i*la+p])*b[guards+p*lb+j];
                double err=std::abs(c[id]-sum); require(std::isfinite(err) && err<=0.002,"FP64 error failed");
                worst=std::max(worst,err); ++total;
            }
            require(threads()==1,"extra OS threads after execution");
            std::printf("M=%d N=%d K=%d lda=%d ldb=%d ldc=%d trial=%d checked=%d max_abs=%.10g threads=1 impl=%s\n",
                m,n,k,la,lb,lc,run,m*n,worst,impl());
        }
    }
    // A BF16/FP16/TF32 conversion loses the residual in the first row.
    // Flushing FP32 subnormals changes the second row.
    require(std::fegetround()==FE_TONEAREST,"requires round-to-nearest");
    const float tiny=std::numeric_limits<float>::denorm_min();
    const float small_a[]={std::nextafter(1.0f,2.0f),-1.0f,tiny,0};
    const float small_b[]={1,1}; float small_c[]={poison,poison};
    mul(2,1,2,small_a,2,small_b,1,small_c,1);
    require(small_c[0]==std::numeric_limits<float>::epsilon() && small_c[1]==tiny,
            "IEEE FP32 residual or gradual-underflow check failed");
    std::printf("strict_fp32_residual=%a subnormal=%a\n",small_c[0],small_c[1]);
    float z[]={7,7,7,7}; mul(1,2,0,nullptr,0,nullptr,0,z,4);
    require(z[0]==0 && z[1]==0 && z[2]==7 && z[3]==7,"zero-K overwrite");
    mul(1,3,1,z,1,z,3,z,2); require(z[2]==7,"invalid leading dimension wrote C");
    std::printf("PASS full_fp64_outputs=%zu created_plans=%lu\n",total,plans());
    return 0;
}
