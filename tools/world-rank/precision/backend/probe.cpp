#include "backend.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <limits>
#include <memory>
#include <sched.h>
#include <stdexcept>
#include <vector>

static int threads() {
    int n=0;
    DIR *d=opendir("/proc/self/task");
    if(!d) throw std::runtime_error("cannot count worker tasks");
    while(auto *x=readdir(d)) if(x->d_name[0]!='.') ++n;
    closedir(d);
    return n;
}
static uint16_t bf16(float x) {
    uint32_t u;
    std::memcpy(&u,&x,sizeof u);
    if((u&65535)!=0) throw std::runtime_error("fixture is not exactly BF16");
    return static_cast<uint16_t>(u>>16);
}
static int number(const char *s) {
    char *end=nullptr;
    long x=std::strtol(s,&end,10);
    if(!end || *end || x<=0 || x>std::numeric_limits<int>::max()) throw std::runtime_error("bad shape");
    return static_cast<int>(x);
}
static float row(int i,int t) { return std::ldexp((i+t)%3 ? 1.f:-1.f,-((i+t)%4)); }
static float col(int j,int t) { return std::ldexp((j+t)%5 ? -1.f:1.f,-((j+2*t)%3)); }
static float left_depth(int k,int t) { return (128+((k+3*t)%7))*0x1p-7f; }
static float right_depth(int k,int t) { return ((k+2*t)%5<3 ? -1.f:1.f); }

int main(int argc,char **argv) {
    if(argc!=8) return 2;
    const int m=number(argv[1]), n=number(argv[2]), k=number(argv[3]);
    const int la=number(argv[4]),lb=number(argv[5]),lc=number(argv[6]);
    const bool diagnostic=!std::strcmp(argv[7],"diagnostic");
    cpu_set_t allowed;
    if(sched_getaffinity(0,sizeof allowed,&allowed)!=0 || CPU_COUNT(&allowed)!=1) return 3;
    // All intermediate integer sums are < 2^24 on the registered domain, so any accumulation order is exact.
    if(la<k || lb<n || lc<n || k>100000) return 2;
    bf16_plan_reset_counts();
    std::unique_ptr<void,decltype(&bf16_plan_destroy)> plan(
        bf16_plan_create_strided(m,n,k,la,lb,lc),bf16_plan_destroy);
    int status=bf16_plan_status();
    std::printf("BF16_PLAN,%d,%s,%s,%d\n",status,bf16_plan_implementation(plan.get()),
                bf16_plan_route(plan.get()),bf16_plan_native(plan.get()));
    if(!plan) std::printf("BF16_ERROR,%s\n",bf16_plan_error());
    const bool execute=plan && (bf16_plan_native(plan.get()) || diagnostic);
    std::vector<uint16_t> retained_a,retained_b;
    std::vector<float> retained_c;
    for(int t=0;t<(execute ? 2:0);++t) {
        // Distinct live buffers and BF16 values for each call, plus disjoint padding and subview origins.
        const size_t ao=3+t,bo=5+2*t,co=7+3*t;
        std::vector<uint16_t>a(ao+size_t(m)*la+11,0x7fc0),b(bo+size_t(k)*lb+13,0x7fc0);
        const float guard=-126.25f;
        std::vector<float> c(co+size_t(m)*lc+17,guard);
        for(int i=0;i<m;++i) {
            for(int l=0;l<k;++l) a[ao+size_t(i)*la+l]=bf16(row(i,t)*left_depth(l,t));
            std::fill_n(c.data()+co+size_t(i)*lc,n,std::numeric_limits<float>::quiet_NaN());
        }
        for(int l=0;l<k;++l)
            for(int j=0;j<n;++j)b[bo+size_t(l)*lb+j]=bf16(right_depth(l,t)*col(j,t));
        const auto original_a=a, original_b=b;
        double inner=0;
        for(int l=0;l<k;++l) inner+=double(left_depth(l,t))*right_depth(l,t);
        status=bf16_plan_exec(plan.get(),a.data()+ao,b.data()+bo,c.data()+co);
        if(status!=0) {
            std::printf("BF16_EXEC_ERROR,%d,%s\n",status,bf16_plan_error());return 4;
        }
        size_t checked=0,guarded=0,wrong=0,changed=0;double worst=0;
        for(size_t q=0;q<c.size();++q) {
            if(q>=co && (q-co)/lc<size_t(m) && (q-co)%lc<size_t(n)) {
                const auto i=int((q-co)/lc),j=int((q-co)%lc);
                const double err=std::fabs(c[q]-row(i,t)*col(j,t)*inner);
                if(!std::isfinite(err) || err!=0) ++wrong;
                worst=std::max(worst,std::isfinite(err)?err:std::numeric_limits<double>::infinity());
                ++checked;
            } else { if(c[q]!=guard) ++changed; ++guarded; }
        }
        bool inputs_ok=(a==original_a && b==original_b);
        std::printf("BF16_EXEC,%d,%d,%zu,%zu,%zu,%zu,%.12g,%d,%d,%d,%llu\n",t,status,checked,guarded,
                    wrong,changed,worst,int(inputs_ok),threads(),sched_getcpu(),
                    static_cast<unsigned long long>(bf16_plan_executions(plan.get())));
        if(wrong || changed || !inputs_ok || checked!=size_t(m)*n) return 5;
        if(t==0) { retained_a.swap(a); retained_b.swap(b); retained_c.swap(c); }
    }
    bool created=bool(plan);
    plan.reset();
    uint64_t a=0,b=0,c=0;bf16_plan_counts(&a,&b,&c);
    std::printf("BF16_LIFE,%llu,%llu,%llu\n",static_cast<unsigned long long>(a),
                static_cast<unsigned long long>(b),static_cast<unsigned long long>(c));
    return (a==static_cast<uint64_t>(created) && c==a && b==(execute ? 2u:0u)) ? 0 : 6;
}
