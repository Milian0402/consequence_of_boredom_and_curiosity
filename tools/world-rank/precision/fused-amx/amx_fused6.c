#define _GNU_SOURCE
#include "amx_fused6.h"
#include "amx_compute.h"
#include <stdlib.h>
#include <string.h>

#if defined(__linux__) && defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#define COB_LINUX_X86 1
#include <cpuid.h>
#include <immintrin.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

extern void precision_range_rescue(int, int, int, const float *, int,
                                   const float *, int, float *, int);

static _Thread_local int tile_permission, route;
static _Thread_local const char *reason;

int amx_fused6_last_route(void) { return route; }
int amx_fused6_last_products(void) { return route == 1 ? 6 : 0; }
const char *amx_fused6_last_reason(void) { return reason ? reason : "no-call"; }

#if COB_LINUX_X86
static uint64_t xcr0(void)
{
    unsigned lo, hi;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return ((uint64_t)hi << 32) | lo;
}

static int vector_pack_available(void)
{
    unsigned a,b,c,d;
    if (!__get_cpuid(1,&a,&b,&c,&d) || !(c & (1u<<27)) ||
        (xcr0() & 0xe6u) != 0xe6u ||
        !__get_cpuid_count(7,0,&a,&b,&c,&d) || a < 1 ||
        (b & ((1u<<16)|(1u<<30))) != ((1u<<16)|(1u<<30))) return 0;
    __cpuid_count(7,1,a,b,c,d);
    return (a & (1u<<5)) != 0;
}
#endif

int amx_fused6_init(void)
{
    if (tile_permission) return 0;
    route=0;
#if COB_LINUX_X86
    unsigned a,b,c,d;
    if (!cob_amx_six_compiled() ||
        !__get_cpuid(1,&a,&b,&c,&d) || !(c & (1u<<27)) ||
        !__get_cpuid_count(7,0,&a,&b,&c,&d) ||
        (d & ((1u<<22)|(1u<<24))) != ((1u<<22)|(1u<<24))) {
        reason="no-amx-bf16-cpu";
        return -2;
    }
    /* Linux allocates tile data on demand after this per-process permission
       request. Check permission in every calling thread before its first use. */
    const unsigned long state=(1ul<<17)|(1ul<<18);
    unsigned long supported=0, permitted=0;
    if (syscall(SYS_arch_prctl,0x1021,&supported) ||
        (supported & state)!=state ||
        syscall(SYS_arch_prctl,0x1023,18ul) ||
        syscall(SYS_arch_prctl,0x1022,&permitted) || (permitted & state)!=state ||
        (xcr0() & state)!=state) {
        reason="no-linux-tile-permission";
        return -2;
    }
    tile_permission=1;
    reason="ready";
    return 0;
#else
    reason="unsupported-platform";
    return -2;
#endif
}

static size_t tiles(int size, int width)
{
    return ((size_t)size + (size_t)width - 1)/(size_t)width;
}

size_t amx_fused6_pack_words(int right,int rows,int cols)
{
    if ((right != 0 && right != 1) || rows <= 0 || cols <= 0) return 0;
    const size_t r=tiles(rows,right ? 32 : 16), c=tiles(cols,right ? 16 : 32);
    const size_t words=3u*512u;
    if (r > SIZE_MAX/c || r*c > SIZE_MAX/(words*sizeof(uint16_t))) return 0;
    return r*c*words;
}

static uint32_t float_bits(float x)
{
    uint32_t u;
    memcpy(&u,&x,sizeof(u));
    return u;
}

static float bf32(uint16_t v)
{
    uint32_t u=(uint32_t)v << 16;
    float x;
    memcpy(&x,&u,sizeof(x));
    return x;
}

static uint16_t bf_rne(float x)
{
    uint32_t u=float_bits(x);
    return (uint16_t)((u+0x7fffu+((u>>16)&1u))>>16);
}

static int pack_scalar(int right,int rows,int cols,const float *x,int ld,
                       uint16_t *dst,size_t plane)
{
    const size_t kt=tiles(right ? rows : cols,32);
    memset(dst,0,3u*plane*sizeof(*dst));
    for (int i=0;i<rows;++i) {
        for (int j=0;j<cols;++j) {
            float r=x[(size_t)i*ld+j];
            const uint32_t mag=float_bits(r)&0x7fffffffu;
            if (mag && (mag<0x30800000u || mag>0x4e800000u)) return 1;
            const size_t t=right ? ((size_t)j/16)*kt+(unsigned)i/32
                                 : ((size_t)i/16)*kt+(unsigned)j/32;
            const size_t cell=right ? ((unsigned)i%32/2)*32+(unsigned)j%16*2+(unsigned)i%2
                                    : ((unsigned)i%16)*32+(unsigned)j%32;
            const size_t p=t*512+cell;
            uint16_t h=bf_rne(r);
            r-=bf32(h);
            uint16_t mid=bf_rne(r);
            dst[p]=h;
            dst[plane+p]=mid;
            dst[2*plane+p]=bf_rne(r-bf32(mid));
        }
    }
    return 0;
}

#if COB_LINUX_X86 && ((!defined(__clang__) && __GNUC__ >= 11) || (defined(__clang__) && __clang_major__ >= 14))
#define BFVEC __attribute__((target("avx512f,avx512bw,avx512bf16")))
typedef struct { __m256i h,m,l; } Parts;

BFVEC static inline Parts split16(__m512 v)
{
    const __m256bh h=_mm512_cvtneps_pbh(v);
    const __m512 r=_mm512_sub_ps(v,_mm512_cvtpbh_ps(h));
    const __m256bh m=_mm512_cvtneps_pbh(r);
    const __m256bh l=_mm512_cvtneps_pbh(_mm512_sub_ps(r,_mm512_cvtpbh_ps(m)));
    return (Parts){(__m256i)h,(__m256i)m,(__m256i)l};
}

BFVEC static inline int unsafe16(__m512 v)
{
    const __m512i zero=_mm512_setzero_si512();
    const __m512i mag=_mm512_and_si512(_mm512_castps_si512(v),_mm512_set1_epi32(0x7fffffff));
    return (_mm512_cmplt_epi32_mask(mag,_mm512_set1_epi32(0x30800000)) &
            _mm512_cmpneq_epi32_mask(mag,zero)) |
            _mm512_cmpgt_epi32_mask(mag,_mm512_set1_epi32(0x4e800000));
}

BFVEC static inline __m512 load16(const float *x,int ld,int r,int c,int rows,int cols)
{
    if (r>=rows || c>=cols) return _mm512_setzero_ps();
    const unsigned count=(unsigned)(cols-c);
    const __mmask16 mask=count>=16 ? 0xffffu : (1u<<count)-1u;
    return _mm512_maskz_loadu_ps(mask,x+(size_t)r*ld+c);
}

BFVEC static inline __m512i zip_pairs(__m256i even,__m256i odd)
{
    return _mm512_or_si512(_mm512_cvtepu16_epi32(even),
                          _mm512_slli_epi32(_mm512_cvtepu16_epi32(odd),16));
}

BFVEC static int pack_vector(int right,int rows,int cols,const float *x,int ld,
                             uint16_t *dst,size_t plane)
{
    const size_t kt=tiles(right ? rows : cols,32), ot=tiles(right ? cols : rows,16);
    for (size_t o=0;o<ot;++o) {
        for (size_t t=0;t<kt;++t) {
            uint16_t *base=dst+(o*kt+t)*512;
            for (int i=0;i<16;++i) {
                /* Right stores 16 interleaved depth pairs; left stores 16 rows. */
                const int r=right ? (int)(32*t)+2*i : (int)(16*o)+i;
                const int c=right ? (int)(16*o) : (int)(32*t);
                __m512 x0=load16(x,ld,r,c,rows,cols);
                __m512 x1=load16(x,ld,r+right,c+(right ? 0 : 16),rows,cols);
                if (unsafe16(x0) | unsafe16(x1)) return 1;
                Parts p=split16(x0),q=split16(x1);
                uint16_t *out=base+i*32;
                if (right) {
                    _mm512_storeu_si512(out,zip_pairs(p.h,q.h));
                    _mm512_storeu_si512(out+plane,zip_pairs(p.m,q.m));
                    _mm512_storeu_si512(out+2*plane,zip_pairs(p.l,q.l));
                } else {
                    _mm256_storeu_si256((__m256i *)out,p.h);
                    _mm256_storeu_si256((__m256i *)(out+16),q.h);
                    _mm256_storeu_si256((__m256i *)(out+plane),p.m);
                    _mm256_storeu_si256((__m256i *)(out+plane+16),q.m);
                    _mm256_storeu_si256((__m256i *)(out+2*plane),p.l);
                    _mm256_storeu_si256((__m256i *)(out+2*plane+16),q.l);
                }
            }
        }
    }
    return 0;
}
#endif

int amx_fused6_pack(int right,int rows,int cols,const float *x,int ld,uint16_t *dst)
{
    const size_t words=amx_fused6_pack_words(right,rows,cols);
    if (!words || !x || !dst || ld<cols) return -1;
#if defined(BFVEC)
    if (vector_pack_available()) return pack_vector(right,rows,cols,x,ld,dst,words/3);
#endif
    return pack_scalar(right,rows,cols,x,ld,dst,words/3);
}

int amx_fused6_sgemm(int m,int n,int k,const float *a,int lda,
                     const float *b,int ldb,float *c,int ldc)
{
    route=0;
    reason="invalid-argument";
    if (m<0 || n<0 || k<0) return -1;
    if (!m || !n) { reason="empty-output"; return 0; }
    if (!c || ldc<n) return -1;
    if (!k) {
        for (int i=0;i<m;++i) memset(c+(size_t)i*ldc,0,(size_t)n*sizeof(*c));
        reason="zero-k";
        return 0;
    }
    if (lda<k || ldb<n || !a || !b) return -1;
    if (amx_fused6_init()) return -2;
    const size_t aw=amx_fused6_pack_words(0,m,k), bw=amx_fused6_pack_words(1,k,n);
    uint16_t *p=NULL;
    if (aw && bw && aw<=SIZE_MAX/sizeof(*p)-bw)
        p=aligned_alloc(64,(aw+bw)*sizeof(*p));
    if (p && !amx_fused6_pack(0,m,k,a,lda,p) && !amx_fused6_pack(1,k,n,b,ldb,p+aw)) {
        int result=cob_amx_six_tiles(m,n,k,p,p+aw,c,ldc);
        free(p);
        if (result) {
            reason="tile-kernel-unavailable";
            return -2;
        }
        route=1;
        reason="amx-tdpbf16ps-six";
        return 0;
    }
    reason="fp64-range-rescue";
    free(p);
    precision_range_rescue(m,n,k,a,lda,b,ldb,c,ldc);
    route=2;
    return 0;
}

#if defined(AMX_FUSED6_BENCH_ABI) && AMX_FUSED6_BENCH_ABI
int precision_sgemm(int mode,int m,int n,int k,const float *a,int lda,
                    const float *b,int ldb,float *c,int ldc)
{
    if (mode==6) return amx_fused6_sgemm(m,n,k,a,lda,b,ldb,c,ldc);
    route=0;
    reason="unsupported-component-count";
    return -1;
}
int precision_last_route(void) { return amx_fused6_last_route(); }
int precision_last_products(void) { return amx_fused6_last_products(); }
const char *precision_last_reason(void) { return amx_fused6_last_reason(); }
int precision_last_plan_count(void) { return 0; }
int precision_last_plan_k(int i) { (void)i; return 0; }
int precision_last_plan_native(int i) { (void)i; return 0; }
const char *precision_last_plan_impl(int i) { (void)i; return ""; }
#endif
