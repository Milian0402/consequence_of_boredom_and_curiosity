#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_version.h>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>

#ifndef ONEDNN_CACHED_PLAN
#define ONEDNN_CACHED_PLAN 0
#endif

using namespace dnnl;
using key = std::array<int, 6>;

struct plan {
    memory::desc a, b, c;
    matmul::primitive_desc descriptor;
    matmul operation;

    static primitive_attr strict_attributes() {
        primitive_attr attr;
        attr.set_fpmath_mode(fpmath_mode::strict);
#ifndef ONEDNN_DISTRO_DIAGNOSTIC
        attr.set_accumulation_mode(accumulation_mode::strict);
#endif
        return attr;
    }
    plan(const engine &eng, const key &s)
        : a({s[0],s[2]}, memory::data_type::f32, {s[3],1}),
          b({s[2],s[1]}, memory::data_type::f32, {s[4],1}),
          c({s[0],s[1]}, memory::data_type::f32, {s[5],1}),
          descriptor(eng,a,b,c,strict_attributes()), operation(descriptor) {}
};

struct state {
    engine eng{engine::kind::cpu,0};
    stream queue{eng};
    std::unique_ptr<plan> cached;
    key previous{};
    std::string implementation;
    unsigned long created = 0;
};
static thread_local std::unique_ptr<state> worker;

extern "C" int rank_vendor_init() {
    if (worker) return 0;
    try {
        auto v=dnnl_version();
#ifndef ONEDNN_DISTRO_DIAGNOSTIC
        // The official v3.14.2 tag carries PROJECT_VERSION=3.14.1.
        if (v->major!=3 || v->minor!=14 || v->patch!=1 ||
                v->cpu_runtime!=DNNL_RUNTIME_SEQ || v->gpu_runtime!=DNNL_RUNTIME_NONE)
            throw std::runtime_error("requires the pinned sequential CPU build");
#else
        if (v->major != DNNL_VERSION_MAJOR || v->minor != DNNL_VERSION_MINOR ||
                v->patch != DNNL_VERSION_PATCH)
            throw std::runtime_error("distro headers and runtime version disagree");
#endif
        if (dnnl_set_default_fpmath_mode(dnnl_fpmath_mode_strict)!=dnnl_success)
            throw std::runtime_error("cannot select strict fpmath");
#ifdef ONEDNN_DISTRO_DIAGNOSTIC
        if (dnnl_set_primitive_cache_capacity(0)!=dnnl_success)
            throw std::runtime_error("cannot disable distro primitive cache");
#endif
        int capacity = -1;
        if (dnnl_get_primitive_cache_capacity(&capacity)!=dnnl_success || capacity!=0)
            throw std::runtime_error("primitive cache is active");
        worker=std::make_unique<state>();
        return 0;
    } catch(const std::exception &e) {
        std::fprintf(stderr,"oneDNN init failed: %s\n",e.what()); return 1;
    }
}

extern "C" void rank_vendor_sgemm(int m,int n,int k,const float *a,int lda,
        const float *b,int ldb,float *c,int ldc) {
    if (m<=0 || n<=0 || k<0 || ldc<n || !c) return;
    if (k==0) {
        for(int i=0;i<m;++i) for(int j=0;j<n;++j) c[(size_t)i*ldc+j]=0.0f;
        return;
    }
    if (!a || !b || lda<k || ldb<n) return;
    try {
        if (!worker && rank_vendor_init()) std::abort();
        auto &w=*worker;
        key shape{m,n,k,lda,ldb,ldc};
        std::unique_ptr<plan> fresh;
        auto &chosen=ONEDNN_CACHED_PLAN ? w.cached : fresh;
        if (!chosen || w.previous!=shape) {
            chosen=std::make_unique<plan>(w.eng,shape);
            w.previous=shape;
            ++w.created;
            w.implementation=chosen->descriptor.impl_info_str();
        }
        memory x(chosen->a,w.eng,const_cast<float*>(a));
        memory y(chosen->b,w.eng,const_cast<float*>(b));
        memory z(chosen->c,w.eng,c);
        chosen->operation.execute(w.queue,{{DNNL_ARG_SRC,x},{DNNL_ARG_WEIGHTS,y},{DNNL_ARG_DST,z}});
        w.queue.wait();
    } catch(const std::exception &e) {
        std::fprintf(stderr,"oneDNN matmul failed: %s\n",e.what());
        std::abort();
    }
}
extern "C" const char *rank_onednn_implementation() { return worker ? worker->implementation.c_str() : ""; }
extern "C" unsigned long rank_onednn_plans() { return worker ? worker->created : 0; }
extern "C" int rank_onednn_cached_plan() { return ONEDNN_CACHED_PLAN; }
extern "C" const dnnl_version_t *rank_onednn_version() { return dnnl_version(); }
