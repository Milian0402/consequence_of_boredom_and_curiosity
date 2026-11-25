#ifndef RANK_MKL_HANDLE_H
#define RANK_MKL_HANDLE_H
#include <stdint.h>
struct geometry { int m,n,k,lda,ldb,ldc; };

struct observation {
    int creates, gets, calls, destroys, create_status, execution_status, destroy_status;
    int layout, transa, transb, m, n, k, lda, ldb, ldc;
    float alpha, beta;
    uintptr_t a,b,c,handle,kernel;
};
#endif
