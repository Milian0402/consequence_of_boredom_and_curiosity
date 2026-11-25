#ifndef COB_PRECISION_EXPERIMENT_H
#define COB_PRECISION_EXPERIMENT_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
int precision_sgemm(int mode,int m,int n,int k,const float *a,int lda,const float *b,int ldb,float *c,int ldc);
int precision_last_route(void);
int precision_last_products(void);
const char *precision_last_reason(void);
int precision_last_plan_count(void);
/* Per call, one entry per distinct initialized primitive, even if executed repeatedly. */
const char *precision_last_plan_impl(int index);
int precision_last_plan_native(int index);
int precision_last_plan_k(int index);
uint16_t precision_bf16_rne(float x);
void precision_range_rescue(int m,int n,int k,const float *a,int lda,const float *b,int ldb,float *c,int ldc);
void precision_fp32_control(int m,int n,int k,const float *a,int lda,const float *b,int ldb,float *c,int ldc);
#ifdef __cplusplus
}
#endif
#endif
