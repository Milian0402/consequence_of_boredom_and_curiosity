#ifndef PRECISION_ONEDNN_BF16_BACKEND_H
#define PRECISION_ONEDNN_BF16_BACKEND_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

void *bf16_plan_create(int m, int n, int k);
void *bf16_plan_create_strided(int m, int n, int k, int lda, int ldb, int ldc);
int bf16_plan_exec(void *plan, const uint16_t *a, const uint16_t *b, float *c);
void bf16_plan_destroy(void *plan);
int bf16_plan_native(const void *plan);
const char *bf16_plan_implementation(const void *plan);
const char *bf16_plan_route(const void *plan);
uint64_t bf16_plan_executions(const void *plan);
int bf16_plan_status(void);
const char *bf16_plan_error(void);
void bf16_plan_counts(uint64_t *created, uint64_t *executed, uint64_t *destroyed);
void bf16_plan_reset_counts(void);
#ifdef __cplusplus
}
#endif
#endif
