#ifndef COB_EXACT_CHECKED_INT8_H
#define COB_EXACT_CHECKED_INT8_H
#ifdef __cplusplus
extern "C" {
#endif
/* Valid row-major, nonoverlapping output. Fallback must implement ordinary FP32 overwrite SGEMM. */
typedef void (*exact_int8_fallback)(int,int,int,const float*,int,const float*,int,float*,int);
/* 1 = exact integer native, 2 = input/depth bypass, 3 = unavailable/error, -1 = missing fallback. */
int exact_int8_checked_sgemm(int m,int n,int k,const float *a,int lda,const float *b,int ldb,
                             float *c,int ldc,exact_int8_fallback fallback);
int exact_int8_checked_last_route(void);
const char *exact_int8_checked_last_reason(void);
unsigned long long exact_int8_checked_last_inspected(void);
int exact_int8_checked_last_amax(void);
int exact_int8_checked_last_bmax(void);
int exact_int8_checked_last_native(void);
int exact_int8_checked_last_plans(void);
int exact_int8_checked_last_products(void);
int exact_int8_checked_last_fallback_calls(void);
int exact_int8_checked_last_status(void);
const char *exact_int8_checked_last_impl(void);
#ifdef __cplusplus
}
#endif
#endif
