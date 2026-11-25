#define _GNU_SOURCE
#include <asm/hwcap.h>
#include <blis.h>
#include <inttypes.h>
#include <linux/prctl.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/auxv.h>
#include <sys/prctl.h>

#if defined(BLIS_ENABLE_PBA_POOLS) || defined(BLIS_ENABLE_SBA_POOLS) || \
    !defined(BLIS_DISABLE_PBA_POOLS) || !defined(BLIS_DISABLE_SBA_POOLS)
#error This comparator requires per-call allocation instead of BLIS packing/small-block pools.
#endif

#define MICROKERNEL bli_sgemm_armsve_asm_2vx10_unindexed
#define UKR_PARAMS dim_t m, dim_t n, dim_t k, const void *alpha, const void *a, const void *b, \
                   const void *beta, void *c, inc_t rs, inc_t cs, const auxinfo_t *aux, const cntx_t *cntx
extern void MICROKERNEL(UKR_PARAMS);

#ifdef RANK_BLIS_TRACE
static _Thread_local uint64_t calls, volume;
static _Thread_local int sve_bits;
void __real_bli_sgemm_armsve_asm_2vx10_unindexed(UKR_PARAMS);
void __wrap_bli_sgemm_armsve_asm_2vx10_unindexed(UKR_PARAMS)
{
    ++calls;
    volume += (uint64_t)m * (uint64_t)n * (uint64_t)k;
    __real_bli_sgemm_armsve_asm_2vx10_unindexed(m, n, k, alpha, a, b, beta, c, rs, cs, aux, cntx);
}

void rank_blis_snapshot(uint64_t out[2])
{
    out[0] = calls;
    out[1] = volume;
}
#endif

int rank_vendor_init(void)
{
    if (!(getauxval(AT_HWCAP) & HWCAP_SVE)) {
        fputs("BLIS armsve unavailable: CPU/OS lacks SVE\n", stderr);
        return 1;
    }
    int vl = prctl(PR_SVE_GET_VL);
    if (vl < 0) return 1;
#ifdef RANK_BLIS_TRACE
    sve_bits = (vl & PR_SVE_VL_LEN_MASK) * 8;
#endif
    bli_init();
    bli_thread_set_num_threads(1);
    arch_t arch = bli_arch_query_id();
    if (arch != BLIS_ARCH_ARMSVE || bli_info_get_enable_threading() ||
        bli_info_get_enable_pba_pools() || bli_info_get_enable_sba_pools() ||
        bli_thread_get_num_threads() != 1) return 1;
    const cntx_t *context = bli_gks_query_cntx();
    if (bli_cntx_get_ukr_dt(BLIS_FLOAT, BLIS_GEMM_UKR, context) != (void *)MICROKERNEL) return 1;
    fprintf(stderr, "BLIS tag=2.1 revision=caf0db6be1202d9c83c79b51e75eceb96aa8b556 "
            "config=%s sve_vl_bytes=%d threaded=0 pba_pools=0 sba_pools=0 entry=bli_sgemm "
            "kernel=bli_sgemm_armsve_asm_2vx10_unindexed\n", bli_arch_string(arch), vl & PR_SVE_VL_LEN_MASK);
    return 0;
}

void rank_vendor_sgemm(int m, int n, int k, const float *a, int lda, const float *b, int ldb, float *c, int ldc)
{
    const float alpha = 1.0f, beta = 0.0f;
#ifdef RANK_BLIS_TRACE
    calls = volume = 0;
#endif
    bli_sgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, m, n, k,
              &alpha, a, lda, 1, b, ldb, 1, &beta, c, ldc, 1);
#ifdef RANK_BLIS_TRACE
    fprintf(stderr, "ARM_PEER_TRACE method=blis m=%d n=%d k=%d lda=%d ldb=%d ldc=%d "
            "kernel=bli_sgemm_armsve_asm_2vx10_unindexed calls=%" PRIu64 " sve_bits=%d "
            "kernel_mnk=%" PRIu64 "\n", m, n, k, lda, ldb, ldc, calls, sve_bits, volume);
#endif
}
