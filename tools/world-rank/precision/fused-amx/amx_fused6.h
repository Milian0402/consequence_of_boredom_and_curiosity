#ifndef COB_PRECISION_AMX_FUSED6_H
#define COB_PRECISION_AMX_FUSED6_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
int amx_fused6_init(void);
int amx_fused6_sgemm(int m, int n, int k, const float *a, int lda,
                     const float *b, int ldb, float *c, int ldc);
int amx_fused6_last_route(void);
int amx_fused6_last_products(void);
const char *amx_fused6_last_reason(void);
size_t amx_fused6_pack_words(int right, int rows, int cols);
int amx_fused6_pack(int right, int rows, int cols, const float *input,
                    int ld, uint16_t *packed);
#ifdef __cplusplus
}
#endif
#endif
