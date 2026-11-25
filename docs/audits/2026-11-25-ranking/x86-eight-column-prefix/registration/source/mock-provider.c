/* Local diagnostic provider. Hosted measurements use the pinned sequential oneMKL adapter. */
#include "prefix.h"
#include <string.h>
int rank_vendor_init(void) { return 0; }
void rank_vendor_sgemm(GEMM_ARGS)
{
    for(int i=0;i<m;++i) {
        float *c_row=c+(size_t)i*ldc;
        memset(c_row,0,(size_t)n*sizeof(*c));
        for(int p=0;p<k;++p) {
            float value=a[(size_t)i*lda+p];
            for(int j=0;j<n;++j) c_row[j]+=value*b[(size_t)p*ldb+j];
        }
    }
}
