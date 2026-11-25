#include <stdint.h>
#include <stddef.h>

#if COB_ENABLE_OPENBLAS_LEAF
#include <cblas.h>
typedef enum CBLAS_ORDER order_t;
typedef enum CBLAS_TRANSPOSE transpose_t;
typedef blasint blas_int;
#elif COB_ENABLE_MKL_SEQUENTIAL_LEAF
#include <mkl.h>
typedef CBLAS_LAYOUT order_t;
typedef CBLAS_TRANSPOSE transpose_t;
typedef MKL_INT blas_int;
#else
#include <armpl.h>
typedef CBLAS_LAYOUT order_t;
typedef enum CBLAS_TRANSPOSE transpose_t;
typedef armpl_int_t blas_int;
#endif

enum { slots = 64, fields = 11 };
static _Thread_local uint64_t record[slots][fields];
static _Thread_local unsigned counts[3];

unsigned rank_backend_leaf_count(int tag)
{
    return tag >= 0 && tag < 3 ? counts[tag] : 0;
}

uint64_t rank_backend_leaf_field(unsigned sequence, unsigned field)
{
    if (sequence >= counts[2] || counts[2]-sequence > slots || field >= fields) return UINT64_MAX;
    return record[sequence % slots][field];
}

extern __typeof__(cblas_sgemm) __real_cblas_sgemm;
void __wrap_cblas_sgemm(order_t order,transpose_t ta,transpose_t tb,blas_int m,blas_int n,blas_int k,
                        float alpha,const float *a,blas_int lda,const float *b,blas_int ldb,float beta,float *c,blas_int ldc)
{
    const unsigned seq = counts[2]++;
    const unsigned tag = (beta == 0.0f ? 0 : (beta == 1.0f ? 1 : 2));
    if (tag < 2) ++counts[tag];
    uint64_t* r = record[seq % slots];
    r[0] = m; r[1] = n; r[2] = k; r[3] = lda; r[4] = ldb; r[5] = ldc;
    r[6] = (uintptr_t)a; r[7] = (uintptr_t)b; r[8] = (uintptr_t)c; r[9] = tag;
    r[10] = (order==CblasRowMajor && ta==CblasNoTrans && tb==CblasNoTrans && alpha==1.0f);
    __real_cblas_sgemm(order,ta,tb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc);
}
