// A single-call FP32 entry to a checked signed oneDNN s8/s8/s32 matrix product.
#include "checked-int8.h"
#include <oneapi/dnnl/dnnl.h>
#include <oneapi/dnnl/dnnl_version.h>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <limits>
#include <memory>
#include <new>
#include <sys/stat.h>
#ifdef __x86_64__
#include <cpuid.h>
#include <sys/syscall.h>
#include <unistd.h>
#elif defined(__aarch64__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif
#ifndef EXACT_INT8_DNNL_PATH
#define EXACT_INT8_DNNL_PATH "libdnnl.so.3"
#endif

using gemm_fn=exact_int8_fallback;
namespace {
static_assert(sizeof(float)==4 && std::numeric_limits<float>::is_iec559 && CHAR_BIT==8,"IEEE binary32 and eight-bit bytes required");
struct observation {
    int route=2,amax=-1,bmax=-1,products=0,plans=0,fallback_calls=0,native=0,status=0;
    unsigned long long inspected=0;
    const char *reason="empty-or-invalid-view"; char impl[192]{};
};
thread_local observation last;
const bool disabled=[](){const char *v=std::getenv("EXACT_INT8_NATIVE_ENABLED");return v && std::strcmp(v,"1");}();

bool owned(const char *path,void *symbol) {
    Dl_info d{};struct stat a{},b{};
    return symbol && dladdr(symbol,&d) && d.dli_fname && !stat(path,&a) && !stat(d.dli_fname,&b) && a.st_dev==b.st_dev && a.st_ino==b.st_ino;
}
gemm_fn backing() {
    static gemm_fn selected=[]() -> gemm_fn {
        const char*path=std::getenv("EXACT_FALLBACK_PLUGIN");
        if(!path || !*path || owned(path,reinterpret_cast<void*>(&backing))) return nullptr;
        void *lib=dlopen(path,RTLD_NOW|RTLD_LOCAL);
        if(!lib)return nullptr;
        auto init=reinterpret_cast<int(*)()>(dlsym(lib,"rank_vendor_init"));
        auto mul=reinterpret_cast<gemm_fn>(dlsym(lib,"rank_vendor_sgemm"));
        if(!owned(path,reinterpret_cast<void*>(init)) || !owned(path,reinterpret_cast<void*>(mul)) || init()!=0) { dlclose(lib);return nullptr; }
        return mul;
    }();
    return selected;
}

bool hardware() {
#ifdef __x86_64__
    unsigned a,b,c,d;
    if(!__get_cpuid_count(7,0,&a,&b,&c,&d) || (d&(3u<<24))!=(3u<<24))return false;
    unsigned long mask=0;
    if(syscall(SYS_arch_prctl,0x1022,&mask)!=0)return false;
    if(!(mask&(1ul<<18)) && syscall(SYS_arch_prctl,0x1023,18ul)!=0)return false;
    return syscall(SYS_arch_prctl,0x1022,&mask)==0 && (mask&(1ul<<18));
#elif defined(__aarch64__)
    return (getauxval(AT_HWCAP)&HWCAP_SVE) && (getauxval(AT_HWCAP2)&HWCAP2_SVEI8MM);
#else
    return false;
#endif
}
bool exact_impl(const char *name) {
    if(!name)return false;
#ifdef __x86_64__
    return !std::strncmp(name,"brg_matmul:",11) && std::strstr(name,"amx");
#elif defined(__aarch64__)
    return !std::strcmp(name,"jit:int8");
#else
    return false;
#endif
}

// Typed C API imports keep the ordinary FP32 entry usable if the optional library is absent.
#define DNNL_CALLS(X) X(version) X(get_primitive_cache_capacity) X(engine_create) X(engine_destroy) \
    X(stream_create) X(stream_destroy) X(stream_wait) X(memory_desc_create_with_strides) X(memory_desc_destroy) \
    X(memory_create) X(memory_destroy) X(matmul_primitive_desc_create) X(primitive_desc_query) \
    X(primitive_desc_destroy) X(primitive_create) X(primitive_destroy) X(primitive_execute) \
    X(primitive_attr_create) X(primitive_attr_destroy) X(primitive_attr_set_fpmath_mode)
struct api {
#define FIELD(name) decltype(&dnnl_##name) name=nullptr;
    DNNL_CALLS(FIELD)
#undef FIELD
    bool available=false;
    api(){
        const char *path=std::getenv("EXACT_INT8_DNNL_PATH");
        void *lib=dlopen(path?path:EXACT_INT8_DNNL_PATH,RTLD_NOW|RTLD_LOCAL);
        if(!lib)return;
#define LOAD(name) name=reinterpret_cast<decltype(name)>(dlsym(lib,"dnnl_" #name)); if(!name)return;
        DNNL_CALLS(LOAD)
#undef LOAD
        int capacity=-1;const auto*v=version();
        available=v && v->major==3 && v->minor==14 && v->patch==1 && v->cpu_runtime==DNNL_RUNTIME_SEQ &&
                v->gpu_runtime==DNNL_RUNTIME_NONE && get_primitive_cache_capacity(&capacity)==dnnl_success && capacity==0;
    }
};
#undef DNNL_CALLS
api &driver(){static api lib;return lib;}
struct engine {
    dnnl_engine_t value=nullptr;
    ~engine(){if(value)driver().engine_destroy(value);}
};
thread_local engine cpu;
struct operation {
    api &lib;dnnl_primitive_attr_t attr=nullptr;dnnl_primitive_desc_t desc=nullptr;
    dnnl_primitive_t primitive=nullptr;dnnl_stream_t stream=nullptr;
    dnnl_memory_desc_t md[3]{};dnnl_memory_t memory[3]{};
    explicit operation(api &d):lib(d){}
    ~operation(){
        for(auto &a:memory)if(a)lib.memory_destroy(a);
        if(primitive)lib.primitive_destroy(primitive);
        if(desc)lib.primitive_desc_destroy(desc);
        if(attr)lib.primitive_attr_destroy(attr);
        for(auto &a:md)if(a)lib.memory_desc_destroy(a);
        if(stream)lib.stream_destroy(stream);
    }
};
#ifdef EXACT_INT8_TEST_FAULTS
bool fault(const char *reason){const char *v=std::getenv("EXACT_INT8_TEST_FAULT");return v && !std::strcmp(reason,v);}
#else
constexpr bool fault(const char *){return false;}
#endif

// Validate and produce each packed byte in the same loop, entirely with integer operations.
// Two's-complement bytes in the temporary buffer are consumed as dnnl_s8 only after validation.
int convert(const float *src,int rows,int cols,int ld,uint8_t *dst) {
    uint32_t largest=0;
    for(int i=0;i<rows;++i){
        auto row=src+size_t(i)*ld;auto out=dst+size_t(i)*cols;
        for(int first=0;first<cols;){
            const int count=cols-first<64?cols-first:64;
            uint32_t invalid=0,max_chunk=0;
            for(int j=0;j<count;++j){
                uint32_t u;std::memcpy(&u,row+first+j,sizeof(u));
                const uint32_t mag=u&0x7fffffffu,neg=u>>31;
                const uint32_t shift=(150u-(mag>>23))&31u;
                const uint32_t low=(1u<<shift)-1u;
                invalid|=(mag!=0)&((mag<0x3f800000u)|(mag>(0x42fe0000u+(neg<<17)))|((mag&low)!=0));
                const uint32_t n=(((mag&0x7fffffu)|0x800000u)>>shift)&(0u-uint32_t(mag!=0));
                out[first+j]=static_cast<uint8_t>((n^(0u-neg))+neg);
                if(n>max_chunk)max_chunk=n;
            }
            first+=count;
            if(invalid){last.inspected+=static_cast<unsigned long long>(i)*cols+first;return -1;}
            if(max_chunk>largest)largest=max_chunk;
        }
    }
    last.inspected+=static_cast<unsigned long long>(rows)*cols;
    return static_cast<int>(largest);
}

bool failed(dnnl_status_t status,const char *stage){
    if(status==dnnl_success)return false;
    last.status=status;last.reason=status==dnnl_out_of_memory?"workspace":stage;
    return true;
}

bool attempt(int m,int n,int k,const float *a,int la,const float *b,int lb,float*c,int lc) {
    if(m<=0 || n<=0 || k<=0 || !a || !b || !c || la<k || lb<n || lc<n)return false;
    if(k<=128){last.reason="depth-shallow";return false;}
    if(k>4096){last.reason="depth-outside";return false;}
    last.route=3;
    if(disabled){last.reason="native-disabled";return false;}
    last.reason="backend-unavailable";
    static thread_local const bool usable=hardware();
    if(!usable || fault("capability"))return false;
    auto &d=driver();
    if(!d.available)return false;
    if(!cpu.value && failed(d.engine_create(&cpu.value,dnnl_cpu,0),"backend-unavailable"))return false;
    if(fault("allocate-a"))throw std::bad_alloc();
    std::unique_ptr<uint8_t[]> x(new uint8_t[size_t(m)*k]);
    last.amax=convert(a,m,k,la,x.get());
    if(last.amax<0){last.route=2;last.reason="integer-domain";return false;}
    if(fault("allocate-b"))throw std::bad_alloc();
    std::unique_ptr<uint8_t[]> y(new uint8_t[size_t(k)*n]);
    last.bmax=convert(b,k,n,lb,y.get());
    if(last.bmax<0){last.route=2;last.reason="integer-domain";return false;}
    if(uint64_t(k)*last.amax*last.bmax>(1u<<24)){last.route=2;last.reason="accumulator-bound";return false;}
    if(fault("allocate-c"))throw std::bad_alloc();
    std::unique_ptr<int32_t[]> z(new int32_t[size_t(m)*n]);
    operation fresh(d);dnnl_dim_t sizes[][2]={{m,k},{k,n},{m,n}};dnnl_dim_t strides[][2]={{k,1},{n,1},{n,1}};
    for(int i=0;i<3;++i)
        if(failed(d.memory_desc_create_with_strides(&fresh.md[i],2,sizes[i],i==2?dnnl_s32:dnnl_s8,strides[i]),"backend-unavailable"))return false;
    if(failed(d.primitive_attr_create(&fresh.attr),"backend-unavailable") ||
       failed(d.primitive_attr_set_fpmath_mode(fresh.attr,dnnl_fpmath_mode_strict),"backend-unavailable"))return false;
    if(fault("descriptor") || failed(d.matmul_primitive_desc_create(&fresh.desc,cpu.value,fresh.md[0],fresh.md[1],nullptr,fresh.md[2],fresh.attr),"backend-unavailable"))return false;
    const char *name=nullptr;
    if(failed(d.primitive_desc_query(fresh.desc,dnnl_query_impl_info_str,0,&name),"backend-unavailable"))return false;
    std::snprintf(last.impl,sizeof(last.impl),"%s",name?name:"");
    if(!exact_impl(name) || fault("non-native"))return false;
    if(failed(d.primitive_create(&fresh.primitive,fresh.desc),"backend-unavailable"))return false;
    last.plans=1;last.reason="backend-execution";
    if(failed(d.stream_create(&fresh.stream,cpu.value,dnnl_stream_default_flags),"backend-execution"))return false;
    void *buffers[]={x.get(),y.get(),z.get()};
    for(int i=0;i<3;++i)
        if(failed(d.memory_create(&fresh.memory[i],fresh.md[i],cpu.value,buffers[i]),"backend-execution"))return false;
    dnnl_exec_arg_t args[]={{DNNL_ARG_SRC,fresh.memory[0]},{DNNL_ARG_WEIGHTS,fresh.memory[1]},{DNNL_ARG_DST,fresh.memory[2]}};
    if(fault("execute") || failed(d.primitive_execute(fresh.primitive,fresh.stream,3,args),"backend-execution") ||
       failed(d.stream_wait(fresh.stream),"backend-execution") || fault("after-wait"))return false;
    for(int i=0;i<m;++i)for(int j=0;j<n;++j)c[size_t(i)*lc+j]=static_cast<float>(z[size_t(i)*n+j]);
    last.route=1;last.reason="native-exact";last.native=last.products=1;
    return true;
}
}

extern "C" int exact_int8_checked_sgemm(int m,int n,int k,const float*a,int la,const float*b,int lb,float*c,int lc,gemm_fn fallback) {
    last=observation{};
    if(!fallback)return -1;
    try {if(attempt(m,n,k,a,la,b,lb,c,lc))return last.route;}
    catch(const std::bad_alloc&){last.route=3;last.reason="workspace";}
    catch(...){last.route=3;last.reason="backend-unavailable";}
    last.fallback_calls=1;fallback(m,n,k,a,la,b,lb,c,lc);return last.route;
}
extern "C" int rank_vendor_init(){return backing()?0:1;}
extern "C" void rank_vendor_sgemm(int m,int n,int k,const float*a,int la,const float*b,int lb,float*c,int lc) {
    if(exact_int8_checked_sgemm(m,n,k,a,la,b,lb,c,lc,backing())<0){std::fputs("checked INT8: ordinary SGEMM provider unavailable\n",stderr);std::abort();}
}
extern "C" int exact_int8_checked_last_route(){return last.route;}
extern "C" const char *exact_int8_checked_last_reason(){return last.reason;}
extern "C" unsigned long long exact_int8_checked_last_inspected(){return last.inspected;}
extern "C" int exact_int8_checked_last_amax(){return last.amax;}
extern "C" int exact_int8_checked_last_bmax(){return last.bmax;}
extern "C" int exact_int8_checked_last_native(){return last.native;}
extern "C" int exact_int8_checked_last_plans(){return last.plans;}
extern "C" int exact_int8_checked_last_products(){return last.products;}
extern "C" int exact_int8_checked_last_fallback_calls(){return last.fallback_calls;}
extern "C" int exact_int8_checked_last_status(){return last.status;}
extern "C" const char *exact_int8_checked_last_impl(){return last.impl;}
