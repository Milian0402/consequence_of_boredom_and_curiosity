#include <blis.h>
#include <stdio.h>
#include <string.h>

int rank_vendor_init(void)
{
    bli_init();
    bli_thread_set_num_threads(1);
    const char *arch = bli_arch_string(bli_arch_query_id());
    const char *model = bli_model_string(bli_model_query_id());
    const int threads = (int)bli_thread_get_num_threads();
    const int threaded = (int)bli_info_get_enable_threading();
    const int bits = (int)bli_info_get_blas_int_type_size();
    const int cblas = (int)bli_info_get_enable_cblas();
    fprintf(stderr, "AOCL-BLAS version=%s arch=%s model=%s threading_compiled=%d threads=%d blas_integer_bits=%d cblas=%d\n",
            bli_info_get_version_str(), arch, model, threaded, threads, bits, cblas);
    return !arch || strcmp(arch, "generic") == 0 || threaded || threads != 1 || bits != 32 || !cblas;
}

void rank_vendor_sgemm(int m, int n, int k,
                       const float *a, int lda, const float *b, int ldb,
                       float *c, int ldc)
{
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k,
                1.0f, a, lda, b, ldb, 0.0f, c, ldc);
}
