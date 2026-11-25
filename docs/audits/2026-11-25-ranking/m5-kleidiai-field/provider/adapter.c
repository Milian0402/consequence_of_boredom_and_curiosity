/* Benchmark ABI for ordinary FP32 overwrite GEMM. Every call packs both inputs. */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "kai/kai_common.h"
#if defined(KLEIDI_ELASTIC)
#include "kai/ukernels/matmul/kai_matmul.h"
#include "kai/ukernels/matmul/kai_matmul_pack_lhs.h"
#include "kai/ukernels/matmul/kai_matmul_pack_rhs.h"
#else
#include "kai/ukernels/matmul/pack/kai_lhs_pack_f32p2vlx1_f32_sme.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa.h"
#endif

struct mtmm_config { int num_threads; };

/* Both pack formats have per-column FP32 bias. Reserve zeroes inside this invocation. */
static void *workspace(size_t lhs, size_t rhs, size_t n, void **packed_b, float **bias) {
    if (lhs > SIZE_MAX - 63 || rhs > SIZE_MAX - 63) abort();
    lhs = (lhs + 63) & ~(size_t)63;
    rhs = (rhs + 63) & ~(size_t)63;
    if (lhs > SIZE_MAX - rhs || n > (SIZE_MAX - lhs - rhs) / sizeof(float)) abort();
    void *p = NULL;
    if (posix_memalign(&p, 64, lhs + rhs + n*sizeof(float))) abort();
    *packed_b = (char *)p + lhs;
    *bias = (float *)((char *)*packed_b + rhs);
    memset(*bias, 0, n * sizeof(float));
    return p;
}

/* The upstream header exposes this common assembler symbol only for the SVE2 compiler target. */
extern uint64_t kai_get_sme_vector_length_u8(void);
size_t kleidi_streaming_vector_bytes(void) { return kai_get_sme_vector_length_u8(); }

void mtmm_sgemm_rowmajor(int m, int n, int k, const float *a, int lda, const float *b, int ldb,
                         float *c, int ldc, const struct mtmm_config *config) {
    if (!config || config->num_threads != 1 || m < 0 || n < 0 || k < 0) abort();
    if (m == 0 || n == 0) return;
    if (!c || ldc < n) abort();
    if (k == 0) {
        for (int i = 0; i < m; ++i) memset(c + (size_t)i * ldc, 0, (size_t)n * sizeof(float));
        return;
    }
    if (!a || !b || lda < k || ldb < n) abort();
    void *packed_b;
    float *bias;
#if defined(KLEIDI_ELASTIC)
    const struct kai_matmul_pack_lhs_uker_api lhs = kai_matmul_pack_lhs_mxk_x32p4vsx1_x32_sme();
    const struct kai_matmul_pack_rhs_uker_api rhs = kai_matmul_pack_rhs_kxn_x32p4vsx1bx32_x32_x32_sme();
    const struct kai_matmul_uker_api gemm = kai_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_8vsx8vs_sme2_mopa();
    const struct kai_matmul_pack_lhs_uker_lhs_packed_dim_args dims_a = {.m=m, .k=k};
    const struct kai_matmul_pack_rhs_uker_rhs_packed_dim_args dims_b = {.n=n, .k=k};
    const struct kai_matmul_pack_lhs_uker_lhs_packed_stride_args stride_a = lhs.get_lhs_packed_stride(NULL, &dims_a);
    const struct kai_matmul_pack_rhs_uker_rhs_packed_stride_args stride_b = rhs.get_rhs_packed_stride(NULL, &dims_b);
    void *packed_a = workspace(lhs.get_lhs_packed_size(NULL, &dims_a, &stride_a),
                               rhs.get_rhs_packed_size(NULL, &dims_b, &stride_b), n, &packed_b, &bias);
    struct kai_matmul_pack_lhs_uker_args pack_a = {.shape={.m=m, .k=k},
        .operand={.lhs={.ptr=a, .stride={.m=(size_t)lda*sizeof(float)}}, .lhs_packed={.ptr=packed_a, .stride=stride_a}}};
    struct kai_matmul_pack_rhs_uker_args pack_b = {.shape={.n=n, .k=k},
        .operand={.rhs={.ptr=b, .stride={.k=(size_t)ldb*sizeof(float), .n=sizeof(float)}},
                  .rhs_packed={.ptr=packed_b, .stride=stride_b}, .bias_n={.ptr=bias}}};
    struct kai_matmul_uker_args product = {.shape={.m=m, .n=n, .k=k},
        .operand={.lhs={.ptr=packed_a, .stride={.m=stride_a.m}}, .rhs={.ptr=packed_b, .stride={.n=stride_b.n}},
                  .dst={.ptr=c, .stride={.m=(size_t)ldc*sizeof(float)}}}};
    lhs.run(NULL, &pack_a);
    rhs.run(NULL, &pack_b);
    gemm.run(NULL, &product);
#else
    size_t mr = kai_get_mr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
    size_t nr = kai_get_nr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
    void *packed_a = workspace(kai_get_lhs_packed_size_lhs_pack_f32p2vlx1_f32_sme(m, k, mr, 1, 1),
        kai_get_rhs_packed_size_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme(n, k), n, &packed_b, &bias);
    kai_run_lhs_pack_f32p2vlx1_f32_sme(m, k, mr, 1, 1, 0, a, (size_t)lda*sizeof(float), packed_a);
    kai_run_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme(1, n, k, nr, 1, 1, (size_t)ldb*sizeof(float), b, bias,
                                                       NULL, packed_b, 0, NULL);
    kai_run_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa(m, n, k, packed_a, packed_b, c,
                                                                 (size_t)ldc*sizeof(float), sizeof(float), -INFINITY, INFINITY);
#endif
    free(packed_a);
}
