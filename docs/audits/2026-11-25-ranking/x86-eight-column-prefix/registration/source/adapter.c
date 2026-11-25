#define _GNU_SOURCE
#include "prefix.h"
#include <cpuid.h>
#include <dlfcn.h>
#include <immintrin.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#if !defined(PREFIX_VARIANT) || PREFIX_VARIANT < 0 || PREFIX_VARIANT > 2
#error Define PREFIX_VARIANT as 0 (forward), 1 (two vendor calls), or 2 (vector prefix)
#endif

static gemm_fn vendor_product;
static int intel_avx2;
#ifdef PREFIX_TRACE
static _Thread_local struct receipt trace;
void rank_prefix_snapshot(struct receipt *out) { *out = trace; }
#endif

int rank_vendor_init(void)
{
    if (vendor_product) return 0;
    const char *path = getenv("RANK_PREFIX_MKL");
    if (!path || path[0] != '/') return 1;
    void *handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!handle) return 1;
    int (*initialize)(void) = (int (*)(void))dlsym(handle, "rank_vendor_init");
    gemm_fn entry = (gemm_fn)dlsym(handle, "rank_vendor_sgemm");
    Dl_info owner;
    struct stat requested, resolved;
    if (!entry || !initialize || !dladdr((void *)entry, &owner) ||
        stat(path, &requested) || stat(owner.dli_fname, &resolved) ||
        requested.st_dev != resolved.st_dev || requested.st_ino != resolved.st_ino ||
        initialize()) {
        dlclose(handle);
        return 1;
    }
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid(0, &eax, &ebx, &ecx, &edx)) {
        char vendor[13] = {0};
        memcpy(vendor, &ebx, 4);
        memcpy(vendor + 4, &edx, 4);
        memcpy(vendor + 8, &ecx, 4);
        __builtin_cpu_init();
        intel_avx2 = !strcmp(vendor, "GenuineIntel") &&
            __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    }
    vendor_product = entry;
    return 0;
}

#if PREFIX_VARIANT == 2
__attribute__((target("avx2,fma"),noinline))
static void prefix8(GEMM_ARGS)
{
#ifdef PREFIX_TRACE
    ++trace.prefix_calls;
    trace.prefix = actual(GEMM_PASS);
#else
    (void)n;
#endif
    int i = 0;
    for (; i + 4 <= m; i += 4) {
        __m256 c0 = _mm256_setzero_ps(), c1 = _mm256_setzero_ps();
        __m256 c2 = _mm256_setzero_ps(), c3 = _mm256_setzero_ps();
        const float *a0 = a + (size_t)i * lda;
        const float *a1 = a0 + lda, *a2 = a1 + lda, *a3 = a2 + lda;
        for (int p = 0; p < k; ++p) {
            __m256 bv = _mm256_loadu_ps(b + (size_t)p * ldb);
            c0 = _mm256_fmadd_ps(_mm256_broadcast_ss(a0+p),bv,c0);
            c1 = _mm256_fmadd_ps(_mm256_broadcast_ss(a1+p),bv,c1);
            c2 = _mm256_fmadd_ps(_mm256_broadcast_ss(a2+p),bv,c2);
            c3 = _mm256_fmadd_ps(_mm256_broadcast_ss(a3+p),bv,c3);
        }
        float *cp = c + (size_t)i * ldc;
        _mm256_storeu_ps(cp,c0);
        _mm256_storeu_ps(cp+ldc,c1);
        _mm256_storeu_ps(cp+(size_t)2*ldc,c2);
        _mm256_storeu_ps(cp+(size_t)3*ldc,c3);
    }
    for (; i < m; ++i) {
        __m256 v = _mm256_setzero_ps();
        const float *ai = a + (size_t)i * lda;
        for (int p = 0; p < k; ++p)
            v = _mm256_fmadd_ps(_mm256_broadcast_ss(ai+p),_mm256_loadu_ps(b+(size_t)p*ldb),v);
        _mm256_storeu_ps(c+(size_t)i*ldc,v);
    }
}
#endif

void rank_vendor_sgemm(GEMM_ARGS)
{
#if PREFIX_VARIANT != 0 || defined(PREFIX_TRACE)
    const int eligible = intel_avx2 && a && b && c && ((uintptr_t)b & 63u) == 0 &&
        m >= 24 && m <= 160 && n >= 1024 && n <= 4096 && k >= 96 && k <= 384 &&
        n >= 8*m && m*k <= 49152 && lda >= k && ldb >= n && ldc >= n;
#endif
#ifdef PREFIX_TRACE
    trace = (struct receipt){.intel_avx2=intel_avx2,.eligible=eligible,.incoming=actual(GEMM_PASS)};
#endif
#if PREFIX_VARIANT != 0
    if (eligible) {
#if PREFIX_VARIANT == 2
        prefix8(m,8,k,a,lda,b,ldb,c,ldc);
#else
        vendor_product(m,8,k,a,lda,b,ldb,c,ldc);
#endif
        vendor_product(m,n-8,k,a,lda,b+8,ldb,c+8,ldc);
        return;
    }
#endif
    vendor_product(GEMM_PASS);
}
