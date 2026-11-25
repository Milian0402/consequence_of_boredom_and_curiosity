#include "configs/haswell/config.hpp"
#include <cstdio>
#include <cstring>

#if TCI_USE_OPENMP_THREADS || TCI_USE_PTHREADS_THREADS
#error This comparator requires a serial TCI build
#endif

extern "C" int rank_tblis_haswell()
{
    const auto& cfg = tblis::get_default_config();
    const bool matched = std::strcmp(cfg.name, "haswell_d6x8") == 0 &&
        cfg.gemm_ukr._ukr[0] == reinterpret_cast<void(*)()>(bli_sgemm_asm_6x16);
    std::fprintf(stderr, "tblis_backend=%s haswell_fp32_6x16=%d tci_threads=none\n",
                 cfg.name, int(matched));
    return matched;
}

#ifdef TBLIS_TRACE
static unsigned long long tiles;
extern "C" unsigned long long rank_tblis_tiles() { return tiles; }
extern "C" void __real_bli_sgemm_asm_6x16(tblis::stride_type, const float*, const float*,
    const float*, const float*, float*, tblis::stride_type, tblis::stride_type);

extern "C" void __wrap_bli_sgemm_asm_6x16(tblis::stride_type k, const float* alpha,
    const float* a, const float* b, const float* beta, float* c,
    tblis::stride_type rs, tblis::stride_type cs)
{
    ++tiles;
    __real_bli_sgemm_asm_6x16(k,alpha,a,b,beta,c,rs,cs);
}
#endif
