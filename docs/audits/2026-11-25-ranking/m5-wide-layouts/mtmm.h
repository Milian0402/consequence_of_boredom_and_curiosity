#ifndef MTMM_H
#define MTMM_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mtmm_config {
    int num_threads;
} mtmm_config;

typedef struct mtmm_packed_b_f32 {
    int k;
    int n;
    size_t bytes;
    void* internal;
} mtmm_packed_b_f32;

typedef struct mtmm_packed_a_f32 {
    int m;
    int k;
    size_t bytes;
    void* internal;
} mtmm_packed_a_f32;

typedef enum mtmm_sgemm_strategy {
    MTMM_SGEMM_STRATEGY_INVALID = 0,
    MTMM_SGEMM_STRATEGY_SINGLE_THREAD = 1,
    MTMM_SGEMM_STRATEGY_PACKED_B = 2,
    MTMM_SGEMM_STRATEGY_DIRECT_B = 3
} mtmm_sgemm_strategy;

int mtmm_default_thread_count(void);

mtmm_sgemm_strategy mtmm_sgemm_one_shot_strategy(
    int m,
    int n,
    int k,
    const mtmm_config* config);

const char* mtmm_sgemm_strategy_name(mtmm_sgemm_strategy strategy);

int mtmm_sgemm_effective_threads(
    int m,
    int n,
    int k,
    const mtmm_config* config);

int mtmm_sgemm_direct_b_effective_threads(
    int m,
    int n,
    int k,
    const mtmm_config* config);

int mtmm_sgemm_packed_b_effective_threads(
    int m,
    int n,
    int k,
    const mtmm_config* config);

int mtmm_sgemm_packed_a_effective_threads(
    int m,
    int n,
    int k,
    const mtmm_config* config);

int mtmm_sgemm_packed_ab_effective_threads(
    int m,
    int n,
    int k,
    const mtmm_config* config);

int mtmm_sgemm_split_k_effective_threads(
    int m,
    int n,
    int k,
    const mtmm_config* config);

void mtmm_sgemm_ref_rowmajor(
    int m,
    int n,
    int k,
    const float* a,
    int lda,
    const float* b,
    int ldb,
    float* c,
    int ldc);

int mtmm_sgemm_pack_b(
    mtmm_packed_b_f32* packed,
    int k,
    int n,
    const float* b,
    int ldb);

void mtmm_sgemm_free_packed_b(mtmm_packed_b_f32* packed);

int mtmm_sgemm_pack_a(
    mtmm_packed_a_f32* packed,
    int m,
    int k,
    const float* a,
    int lda);

void mtmm_sgemm_free_packed_a(mtmm_packed_a_f32* packed);

void mtmm_sgemm_rowmajor_packed_b(
    int m,
    int n,
    int k,
    const float* a,
    int lda,
    const mtmm_packed_b_f32* packed_b,
    float* c,
    int ldc,
    const mtmm_config* config);

void mtmm_sgemm_rowmajor_packed_ab(
    int m,
    int n,
    int k,
    const mtmm_packed_a_f32* packed_a,
    const mtmm_packed_b_f32* packed_b,
    float* c,
    int ldc,
    const mtmm_config* config);

void mtmm_sgemm_rowmajor_packed_a(
    int m,
    int n,
    int k,
    const mtmm_packed_a_f32* packed_a,
    const float* b,
    int ldb,
    float* c,
    int ldc,
    const mtmm_config* config);

void mtmm_sgemm_rowmajor_direct_b(
    int m,
    int n,
    int k,
    const float* a,
    int lda,
    const float* b,
    int ldb,
    float* c,
    int ldc,
    const mtmm_config* config);

void mtmm_sgemm_rowmajor_split_k(
    int m,
    int n,
    int k,
    const float* a,
    int lda,
    const float* b,
    int ldb,
    float* c,
    int ldc,
    const mtmm_config* config);

void mtmm_sgemm_rowmajor(
    int m,
    int n,
    int k,
    const float* a,
    int lda,
    const float* b,
    int ldb,
    float* c,
    int ldc,
    const mtmm_config* config);

#ifdef __cplusplus
}
#endif

#endif
