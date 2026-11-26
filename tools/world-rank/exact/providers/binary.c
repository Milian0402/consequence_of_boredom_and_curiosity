#include <stdint.h>
#include <stdlib.h>

static _Thread_local int last_words;
int rank_vendor_init(void) { return 0; }
int exact_binary_last_words(void) { return last_words; }

void rank_vendor_sgemm(int m,int n,int k,const float *a,int lda,const float *b,int ldb,float *c,int ldc)
{
    if (m<1 || n<1 || k<1 || lda<k || ldb<n || ldc<n || k>4096) abort();
    const int words=(k+63)/64;
    uint64_t *left=calloc((size_t)m*words,sizeof *left);
    uint64_t *right=calloc((size_t)n*words,sizeof *right);
    if (!left || !right) abort();
    for (int i=0;i<m;++i) {
        for (int p=0;p<k;++p) {
            float x=a[(size_t)i*lda+p];
            if (x!=0.0f && x!=1.0f) abort();
            left[(size_t)i*words+p/64]|=(uint64_t)x<<(p%64);
        }
    }
    for (int p=0;p<k;++p) {
        for (int j=0;j<n;++j) {
            float x=b[(size_t)p*ldb+j];
            if (x!=0.0f && x!=1.0f) abort();
            right[(size_t)j*words+p/64]|=(uint64_t)x<<(p%64);
        }
    }
    for(int i=0;i<m;++i) for(int j=0;j<n;++j) {
        unsigned count=0;
        const uint64_t *x=left+(size_t)i*words,*y=right+(size_t)j*words;
        for(int w=0;w<words;++w) count+=(unsigned)__builtin_popcountll(x[w]&y[w]);
        c[(size_t)i*ldc+j]=(float)count;
    }
    last_words=words;
    free(right);free(left);
}
