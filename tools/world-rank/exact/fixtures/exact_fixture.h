#ifndef EXACT_FIXTURE_H
#define EXACT_FIXTURE_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Logical origins; all strides are in elements. Only the logical cells are written. */
int exact_fixture_fill(const char *spec,const char *corpus_dir,uint64_t seed,
                       int m,int n,int k,float *a,int lda,float *b,int ldb);
/* Opens the registered single-thread OpenBLAS plugin. Outputs are contiguous M x N.
   Validated numerical domain: finite half-integer operands with magnitude <= 512. */
int exact_fixture_golden(const char *dgemm_plugin,int m,int n,int k,
                         const float *a,int lda,const float *b,int ldb,
                         double *reference,double *weights);
int exact_fixture_spots(void);
const char *exact_fixture_error(void);
#ifdef __cplusplus
}
#endif
#endif
