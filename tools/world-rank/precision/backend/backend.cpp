#include "backend.h"
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_version.h>
#include <cstdio>
#include <cstring>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#ifdef __x86_64__
#include <cpuid.h>
#include <sys/syscall.h>
#include <unistd.h>
#elif defined(__aarch64__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif

namespace {
using namespace dnnl;
struct ledger {
    uint64_t created=0, executed=0, destroyed=0;
    dnnl_status_t status=dnnl_success;
    char message[192]{};
};
thread_local ledger history;

int report(dnnl_status_t status, const char *message) noexcept {
    history.status=status;
    std::snprintf(history.message,sizeof history.message,"%s",message ? message : "");
    return static_cast<int>(status);
}

engine &cpu_engine() {
    static thread_local engine cpu=[] {
        const auto *v=dnnl_version();
        if (v->major!=3 || v->minor!=14 || v->patch!=1 ||
                v->cpu_runtime!=DNNL_RUNTIME_SEQ || v->gpu_runtime!=DNNL_RUNTIME_NONE)
            throw std::runtime_error("expected pinned oneDNN v3.14.2 (library 3.14.1), sequential CPU only");
        int capacity=-1;
        auto status=dnnl_get_primitive_cache_capacity(&capacity);
        if (status!=dnnl_success) throw error(status,"cannot read primitive cache capacity");
        if (capacity!=0) throw std::runtime_error("primitive cache must be disabled");
        return engine(engine::kind::cpu,0);
    }();
    return cpu;
}

primitive_attr attributes() {
    primitive_attr attr;
    attr.set_fpmath_mode(fpmath_mode::strict);
    attr.set_accumulation_mode(accumulation_mode::strict);
    return attr;
}

// Only a caller explicitly requesting metadata pays for CPUID/OS queries.
// bf16 is the actual primitive input datatype. ISA labels alone never reclassify an f32 primitive.
const char *classify(const std::string &impl) {
#ifdef __x86_64__
    unsigned a=0,b=0,c=0,d=0;
    const bool cpuid_ok=__get_cpuid_count(7,0,&a,&b,&c,&d)!=0;
    const bool amx=cpuid_ok && (d&(1u<<22)) && (d&(1u<<24));
    // Linux ARCH_GET_XCOMP_PERM. oneDNN itself requests XTILEDATA if dispatch requires it.
    unsigned long mask=0;
    if (impl.rfind("brg_matmul:",0)==0 && impl.find("amx")!=std::string::npos) {
        if (amx && syscall(SYS_arch_prctl,0x1022,&mask)==0 && (mask&(1ul<<18)))
            return "x86-amx-bf16";
        return "amx-feature-or-os-permission-unverified";
    }
    if (impl.rfind("brg_matmul:",0)==0 && impl.find("bf16")!=std::string::npos && impl.find("avx512")!=std::string::npos) {
        const bool base=cpuid_ok && (b&(1u<<16));
        if(base && __get_cpuid_count(7,1,&a,&b,&c,&d) && (a&(1u<<5))) return "x86-avx512-bf16";
    }
#elif defined(__aarch64__)
    // gemm_bf16bf16f32 dispatches to ref_gemm_bf16bf16f32 on AArch64 in the pinned release.
    if (impl=="gemm:jit:bf16") return "arm-reference-bf16";
    if (impl.rfind("brg:sve_",0)==0 || impl=="jit:bf16") {
        if ((getauxval(AT_HWCAP)&HWCAP_SVE) && (getauxval(AT_HWCAP2)&HWCAP2_SVEBF16))
            return "arm-sve-bf16";
        return "sve-bf16-feature-unverified";
    }
#endif
    if (impl.find("ref")!=std::string::npos) return "reference";
    return "other-implementation-unverified";
}

struct plan {
    engine &cpu;
    stream queue;
    memory::desc a,b,c;
    matmul::primitive_desc desc;
    matmul product;
    std::string implementation;
    uint64_t executions=0;

    plan(int m,int n,int k,int lda,int ldb,int ldc)
        : cpu(cpu_engine()), queue(cpu),
          a({m,k},memory::data_type::bf16,{lda,1}),
          b({k,n},memory::data_type::bf16,{ldb,1}),
          c({m,n},memory::data_type::f32,{ldc,1}),
          desc(cpu,a,b,c,attributes()),product(desc),implementation(desc.impl_info_str()) {}
};
}

extern "C" void *bf16_plan_create_strided(int m,int n,int k,int lda,int ldb,int ldc) {
    if (m<=0 || n<=0 || k<=0 || lda<k || ldb<n || ldc<n) {
        report(dnnl_invalid_arguments,"invalid dimensions or row strides");
        return nullptr;
    }
    try {
        auto fresh=std::make_unique<plan>(m,n,k,lda,ldb,ldc);
        ++history.created;
        report(dnnl_success,"");
        return fresh.release();
    } catch (const dnnl::error &e) { report(e.status,e.what()); }
      catch (const std::bad_alloc &e) { report(dnnl_out_of_memory,e.what()); }
      catch (const std::exception &e) { report(dnnl_runtime_error,e.what()); }
    return nullptr;
}

extern "C" void *bf16_plan_create(int m,int n,int k) {
    return bf16_plan_create_strided(m,n,k,k,n,n);
}

extern "C" int bf16_plan_exec(void *opaque,const uint16_t *a,const uint16_t *b,float *c) {
    if (!opaque || !a || !b || !c) return report(dnnl_invalid_arguments,"null buffer or plan");
    auto &p=*static_cast<plan*>(opaque);
    try {
        dnnl::memory x(p.a,p.cpu,const_cast<uint16_t*>(a));
        dnnl::memory y(p.b,p.cpu,const_cast<uint16_t*>(b));
        dnnl::memory z(p.c,p.cpu,c);
        p.product.execute(p.queue,{{DNNL_ARG_SRC,x},{DNNL_ARG_WEIGHTS,y},{DNNL_ARG_DST,z}});
        p.queue.wait();
        ++p.executions;
        ++history.executed;
        return report(dnnl_success,"");
    } catch (const dnnl::error &e) { return report(e.status,e.what()); }
      catch (const std::bad_alloc &e) { return report(dnnl_out_of_memory,e.what()); }
      catch (const std::exception &e) { return report(dnnl_runtime_error,e.what()); }
}

extern "C" void bf16_plan_destroy(void *p) {
    if (p) { delete static_cast<plan*>(p); ++history.destroyed; }
}
extern "C" const char *bf16_plan_implementation(const void *p) {
    return p ? static_cast<const plan*>(p)->implementation.c_str() : "";
}
extern "C" const char *bf16_plan_route(const void *p) {
    return p ? classify(static_cast<const plan*>(p)->implementation) : "no-descriptor";
}
extern "C" int bf16_plan_native(const void *p) {
    const char *r=bf16_plan_route(p);
    return !std::strcmp(r,"x86-amx-bf16") || !std::strcmp(r,"arm-sve-bf16") || !std::strcmp(r,"x86-avx512-bf16");
}
extern "C" uint64_t bf16_plan_executions(const void *p) { return p ? static_cast<const plan*>(p)->executions : 0; }
extern "C" int bf16_plan_status() { return static_cast<int>(history.status); }
extern "C" const char *bf16_plan_error() { return history.message; }
extern "C" void bf16_plan_counts(uint64_t *a,uint64_t *b,uint64_t *c) {
    if(a)*a=history.created;
    if(b)*b=history.executed;
    if(c)*c=history.destroyed;
}
extern "C" void bf16_plan_reset_counts() { history.created=history.executed=history.destroyed=0; }
