#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "iface/3m/mult.h"
#include "stra_iface/3m/stra_mult.h"
#include "stra_iface/3m/stra_mult_2level.h"

#if TBLIS_AB_LEVEL != 1 && TBLIS_AB_LEVEL != 2
#error Choose one or two AB Strassen levels
#endif

extern const tblis_config* rank_tblis_config(void);

static tblis_matrix matrix(float scale, const float *data, int rows, int cols, int stride)
{
    tblis_matrix view = {
        .type = TYPE_FLOAT, .data = (void *)data, .m = rows, .n = cols, .rs = stride, .cs = 1
    };
    memcpy(view.scalar, &scale, sizeof(scale));
    return view;
}

static void ordinary(int m, int n, int k, const float *a, int lda,
                     const float *b, int ldb, float *c, int ldc, float beta)
{
    tblis_matrix left = matrix(1, a, m, k, lda);
    tblis_matrix right = matrix(1, b, k, n, ldb);
    tblis_matrix out = matrix(beta, c, m, n, ldc);
    tblis_matrix_mult(tblis_single, rank_tblis_config(), &left, &right, &out);
}

int rank_vendor_init(void)
{
    tblis_set_num_threads(1);
    fprintf(stderr, "TBLIS-Strassen FP32 AB level %d, single communicator\n", TBLIS_AB_LEVEL);
    extern int rank_tblis_haswell(void);
    return tblis_get_num_threads() != 1 || tblis_single == NULL || !rank_tblis_haswell();
}

void rank_vendor_sgemm(int m, int n, int k, const float *a, int lda,
                       const float *b, int ldb, float *c, int ldc)
{
    const int divisor = 1 << TBLIS_AB_LEVEL;
    const int rows = m - m % divisor;
    const int cols = n - n % divisor;
    const int depth = k - k % divisor;
    if (m == 0 || n == 0) return;
    if (k == 0) {
        for (int i = 0; i < m; ++i) memset(c + (size_t)i * ldc, 0, n * sizeof(*c));
        return;
    }
    if (rows == 0 || cols == 0 || depth == 0) {
        ordinary(m, n, k, a, lda, b, ldb, c, ldc, 0);
        return;
    }

    /* Upstream AB accumulates each product into C. Zeroing belongs inside the call. */
    for (int i = 0; i < rows; ++i) memset(c + (size_t)i * ldc, 0, cols * sizeof(*c));
    tblis_matrix left = matrix(1, a, rows, depth, lda);
    tblis_matrix right = matrix(1, b, depth, cols, ldb);
    tblis_matrix out = matrix(1, c, rows, cols, ldc);
#if TBLIS_AB_LEVEL == 1
    stra_tblis_matrix_mult_ab(tblis_single, rank_tblis_config(), &left, &right, &out);
#else
    stra_tblis_matrix_mult_2level_ab(tblis_single, rank_tblis_config(), &left, &right, &out);
#endif
    if (depth != k)
        ordinary(rows, cols, k - depth, a + depth, lda, b + (size_t)depth * ldb,
                 ldb, c, ldc, 1);
    if (cols != n)
        ordinary(rows, n - cols, k, a, lda, b + cols, ldb, c + cols, ldc, 0);
    if (rows != m)
        ordinary(m - rows, n, k, a + (size_t)rows * lda, lda, b, ldb,
                 c + (size_t)rows * ldc, ldc, 0);
}
