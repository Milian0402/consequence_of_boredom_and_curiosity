/* FP32 entry to the sequential signed-int8 oneDNN primitive. No persistent packed operands/plan. */
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_version.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

static thread_local std::unique_ptr<dnnl::engine> cpu;
static thread_local char implementation[192];
static thread_local int executed;
extern "C" const char* exact_int8_implementation() { return implementation; }
extern "C" int exact_int8_last_terms() { return executed; }
extern "C" int rank_vendor_init() {
    const auto *v=dnnl_version();int capacity=-1;
    if(v->major!=3 || v->minor!=14 || v->patch!=1 || v->cpu_runtime!=DNNL_RUNTIME_SEQ ||
       dnnl_get_primitive_cache_capacity(&capacity)!=dnnl_success || capacity!=0) return 1;
    try { if(!cpu) cpu=std::make_unique<dnnl::engine>(dnnl::engine::kind::cpu,0); return 0; }
    catch(const std::exception& e) { std::fprintf(stderr,"%s\n",e.what()); return 1; }
}
static unsigned convert(const float *src,int rows,int cols,int ld,std::vector<int8_t>& dst) {
    unsigned maximum=0;
    for(int i=0;i<rows;++i) for(int j=0;j<cols;++j) {
        float f=src[size_t(i)*ld+j];
        if(!(f>=-128.f && f<=127.f) || f!=std::trunc(f)) std::abort();
        int x=static_cast<int>(f);maximum=std::max(maximum,static_cast<unsigned>(std::abs(x)));
        dst[size_t(i)*cols+j]=static_cast<int8_t>(x);
    }
    return maximum;
}
extern "C" void rank_vendor_sgemm(int m,int n,int k,const float*a,int la,const float*b,int lb,float*c,int lc) {
    using namespace dnnl;
    if(!cpu || m<=0 || n<=0 || k<=0 || k>4096 || la<k || lb<n || lc<n) std::abort();
    executed=0;implementation[0]=0;
    try {
        std::vector<int8_t> x(size_t(m)*k),y(size_t(k)*n);
        std::vector<int32_t> z(size_t(m)*n);
        auto am=convert(a,m,k,la,x),bm=convert(b,k,n,lb,y);
        if(uint64_t(k)*am*bm>(1u<<24)) std::abort();
        memory::desc ax({m,k},memory::data_type::s8,{k,1}),by({k,n},memory::data_type::s8,{n,1}),cz({m,n},memory::data_type::s32,{n,1});
        primitive_attr attr;attr.set_fpmath_mode(fpmath_mode::strict);
        matmul::primitive_desc desc(*cpu,ax,by,cz,attr);
        matmul op(desc);stream s(*cpu);
        memory aa(ax,*cpu,x.data()),bb(by,*cpu,y.data()),cc(cz,*cpu,z.data());
        op.execute(s,{{DNNL_ARG_SRC,aa},{DNNL_ARG_WEIGHTS,bb},{DNNL_ARG_DST,cc}});s.wait();
        std::snprintf(implementation,sizeof implementation,"%s",desc.impl_info_str());
        for(int i=0;i<m;++i) for(int j=0;j<n;++j)c[size_t(i)*lc+j]=static_cast<float>(z[size_t(i)*n+j]);
        executed=1;
    } catch(const std::exception &e){ std::fprintf(stderr,"integer SGEMM unavailable: %s\n",e.what());std::abort(); }
}
