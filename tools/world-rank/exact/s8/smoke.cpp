// Correctness and failure atomicity only. Every fallback checks the entire untouched dirty C first.
#include "checked-int8.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
using api_fn=decltype(&exact_int8_checked_sgemm);
api_fn product; int (*route)(),(*terms)(),(*plans)(),(*back)(),(*native)(),(*get_amax)(),(*get_bmax)();
unsigned long long (*scanned)(); const char* (*reason)(); const char* (*implementation)();
std::vector<float> *live=nullptr, initial;
int calls=0;
void need(bool condition,const std::string& why){if(!condition)throw std::runtime_error(why);}
void scalar(int m,int n,int k,const float*a,int la,const float*b,int lb,float*c,int lc){
    for(int i=0;i<m;++i)for(int j=0;j<n;++j){double s=0;for(int p=0;p<k;++p)s+=double(a[size_t(i)*la+p])*b[size_t(p)*lb+j];c[size_t(i)*lc+j]=float(s);}
}
void ordinary(int m,int n,int k,const float*a,int la,const float*b,int lb,float*c,int lc){
    need(!std::memcmp(initial.data(),live->data(),initial.size()*4),"caller C written before successful execution or before fallback");
    ++calls;scalar(m,n,k,a,la,b,lb,c,lc);
}
template<class T> void load(void*h,const char*name,T&dest){dest=reinterpret_cast<T>(dlsym(h,name));need(dest!=nullptr,name);}
float word(uint32_t u){float f;std::memcpy(&f,&u,4);return f;}
void enable_ftz(){
#if defined(__aarch64__)
    unsigned long fpcr=0;asm volatile("mrs %0, fpcr" : "=r"(fpcr));fpcr|=(1ul<<24);asm volatile("msr fpcr, %0" :: "r"(fpcr):"memory");
#elif defined(__x86_64__)
    unsigned mxcsr=0;asm volatile("stmxcsr %0":"=m"(mxcsr));mxcsr|=(1u<<15)|(1u<<6);asm volatile("ldmxcsr %0"::"m"(mxcsr):"memory");
#endif
}
int count=0;long long cells=0,guarded=0;
void run(const std::string&label,int k,int expected,const char *why,int poison=0,const char *fault=nullptr){
    constexpr int m=19,n=27;int la=k+7,lb=n+9,lc=n+11;size_t ao=3,bo=5,co=7;
    std::vector<float>a(ao+size_t(m)*la+19,1.5f),b(bo+size_t(k)*lb+21,2.5f),c(co+size_t(m)*lc+23,-15591.25f);
    for(int i=0;i<m;++i)for(int p=0;p<k;++p)a[ao+size_t(i)*la+p]=float((i*19+p*17)%256-128);
    for(int p=0;p<k;++p)for(int j=0;j<n;++j)b[bo+size_t(p)*lb+j]=float((p*21+j*113)%256-128);
    if(poison==8 || poison==9){
        for(int i=0;i<m;++i)for(int p=0;p<k;++p)a[ao+size_t(i)*la+p]=(poison==9 && p==1024)?1.f:-128.f;
        for(int p=0;p<k;++p)for(int j=0;j<n;++j)b[bo+size_t(p)*lb+j]=(poison==9 && p==1024)?1.f:-128.f;
    }
    if(poison==1)a[ao]=word(1u);
    if(poison==2)b[bo+size_t(k-1)*lb+n-1]=127.5f;
    if(poison==3)a[ao]=word(0x7f800001u);
    if(poison==4)a[ao]=128.f;
    if(poison==5)a[ao]=-129.f;
    if(poison==6)a[ao]=-0.f;
    if(poison==7)b[bo+lb+8]=word(0x7f800000u);
    for(int i=0;i<m;++i)for(int j=0;j<n;++j)c[co+size_t(i)*lc+j]=word(0x7fc12345u);
    auto prev_a=a,prev_b=b,oracle=c;initial=c;live=&c;calls=0;
    scalar(m,n,k,a.data()+ao,la,b.data()+bo,lb,oracle.data()+co,lc);
    if(fault)setenv("EXACT_INT8_TEST_FAULT",fault,1);else unsetenv("EXACT_INT8_TEST_FAULT");
    int r=product(m,n,k,a.data()+ao,la,b.data()+bo,lb,c.data()+co,lc,ordinary);
    if(expected<0){expected=r;why=reason();need(r==1 || r==3,"auto forward or native");}
    need(r==expected && route()==expected && !std::strcmp(reason(),why),label+" unexpected route="+std::to_string(r)+" reason="+reason()+" impl="+implementation());
    need(calls==int(r!=1) && back()==calls && terms()==(r==1) && native()==(r==1),label+" telemetry");
    need(!std::memcmp(c.data(),oracle.data(),c.size()*4) && !std::memcmp(a.data(),prev_a.data(),a.size()*4) && !std::memcmp(b.data(),prev_b.data(),b.size()*4),label+" exact result or guards");
    if(poison==8 || poison==9){
        need(c[co]==16777216.f,label+" exact limit / RNE result mismatch");
        const int64_t exact=int64_t(1024)*128*128+(poison==9);
        need((double(c[co])==double(exact))==(poison==8),label+" integer equality at output boundary");
    }
    if(r==1){need(plans()==1 && scanned()==static_cast<unsigned long long>((m+n)*k) && get_amax()==128 && get_bmax()==128 && *implementation(),label+" native certificate");}
    if(!std::strcmp(why,"integer-domain")){
        unsigned long long expected_scan=poison==2?(m+n)*k:poison==7?m*k+2*n:std::min(k,64);
        need(scanned()==expected_scan,label+" poison scan");
    }
    if(fault && (!std::strcmp(fault,"execute") || !std::strcmp(fault,"after-wait")))need(plans()==1,label+" missing created plan");
    ++count;cells+=m*n;guarded+=c.size()-size_t(m)*n;
    std::printf("case=%s route=%d reason=%s checked=%d inspected=%llu plans=%d products=%d fallback=%d impl=%s\n",label.c_str(),r,reason(),m*n,scanned(),plans(),terms(),calls,implementation());
}
}
int main(int argc,char**argv){
    try{
        need(argc==3,"usage: check library auto|blocked|positive|faults|missing");
        void*h=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);if(!h)throw std::runtime_error(dlerror());
        load(h,"exact_int8_checked_sgemm",product);
#define GET(f,v) load(h,"exact_int8_checked_last_" #f,v)
        GET(route,route);GET(reason,reason);GET(products,terms);GET(plans,plans);GET(fallback_calls,back);GET(native,native);GET(inspected,scanned);GET(impl,implementation);GET(amax,get_amax);GET(bmax,get_bmax);
#undef GET
        std::string mode=argv[2];enable_ftz();
        run("zero",0,2,"empty-or-invalid-view");run("depth-128",128,2,"depth-shallow");run("depth-4097",4097,2,"depth-outside");
        need(product(1,1,1,nullptr,0,nullptr,0,nullptr,0,nullptr)==-1,"null callback");
        auto safe_noop=+[](int,int,int,const float*,int,const float*,int,float*,int){++calls;};
        calls=0;need(product(-1,3,199,nullptr,0,nullptr,0,nullptr,0,safe_noop)==2 && calls==1 && scanned()==0,"invalid view/delegation");
        if(mode=="auto"){run("automatic-eligibility",161,-1,"");mode=(route()==1?"positive":"blocked");}
        if(mode=="missing" || mode=="blocked")run("unavailable",161,3,"backend-unavailable");
        else{
            run("signed-min-max",161,1,"native-exact");run("signed-zero",161,1,"native-exact",6);
            run("exact-two-to-24",1024,1,"native-exact",8);run("ordinary-rounds-two-to-24-plus-one",1025,2,"accumulator-bound",9);
            for(int p:{1,2,3,4,5,7})run("poison-"+std::to_string(p),161,2,"integer-domain",p);
            if(mode=="faults"){
                for(auto p:{"allocate-a","allocate-b","allocate-c"})run(p,161,3,"workspace",0,p);
                for(auto p:{"capability","descriptor","non-native"})run(p,161,3,"backend-unavailable",0,p);
                for(auto p:{"execute","after-wait"})run(p,161,3,"backend-execution",0,p);
                run("successful-retry",161,1,"native-exact");
            }
        }
        std::printf("PASS cases=%d output_cells=%lld guards=%lld mode=%s\n",count,cells,guarded,mode.c_str());return 0;
    }catch(const std::exception&e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
