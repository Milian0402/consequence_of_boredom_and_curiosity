#ifndef COB_GEMM_H
#define COB_GEMM_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    COB_SGEMM_MR = 8,
    COB_SGEMM_NR = 8
};

typedef struct cob_packed_b_f32 {
    int k;
    int n;
    int nr;
    size_t bytes;
    float* data;
} cob_packed_b_f32;

void cob_sgemm_ref_rowmajor(
    int m,
    int n,
    int k,
    const float* a,
    int lda,
    const float* b,
    int ldb,
    float* c,
    int ldc);

int cob_sgemm_pack_b(
    cob_packed_b_f32* packed,
    int k,
    int n,
    const float* b,
    int ldb);

void cob_sgemm_free_packed_b(cob_packed_b_f32* packed);

void cob_sgemm_rowmajor_packed_b(
    int m,
    int n,
    int k,
    const float* a,
    int lda,
    const cob_packed_b_f32* packed_b,
    float* c,
    int ldc);

void cob_sgemm_rowmajor(
    int m,
    int n,
    int k,
    const float* a,
    int lda,
    const float* b,
    int ldb,
    float* c,
    int ldc);

#ifdef __cplusplus
}
#endif

#endif

