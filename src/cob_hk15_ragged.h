#ifndef COB_HK15_RAGGED_H
#define COB_HK15_RAGGED_H

typedef void (*cob_hk15_base_fn)(int, int, int, const float*, int,
    const float*, int, float*, int);

void cob_hk15_ragged_finish(int m, int n, int k, const float* a, int lda,
    const float* b, int ldb, float* c, int ldc, float* workspace, cob_hk15_base_fn base);

#endif
