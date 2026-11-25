/* Check the admitted path on the four timing layouts without performing SGEMM. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "gemm.c"

static float fixture_draw(uint32_t *seed) {
    uint32_t value = *seed;
    value ^= value << 13;
    value ^= value >> 17;
    value ^= value << 5;
    *seed = value;
    return (float)((int)(value >> 9) - 4194304) * (1.f / 8388608.f);
}

int main(void) {
    const int shapes[][3] = {{2024,3081,3005},{2153,3007,3159},
                             {1949,3227,3023},{1920,2880,2880}};
    puts("m,n,k,seed,full_certificate,ordinary_decision");
    for (int trial=0;trial<2;++trial) {
        for (int shape=0;shape<4;++shape) {
            const int m=shapes[shape][0],n=shapes[shape][1],k=shapes[shape][2];
            const int lda=k+5,ldb=n+9;
            const uint32_t seed=(trial ? 362004289u : 37774301u)+shape;
            uint32_t state=seed;
            float *ap=calloc((size_t)m*lda+4,sizeof(float));
            float *bp=calloc((size_t)k*ldb+9,sizeof(float));
            if (!ap || !bp) return EXIT_FAILURE;
            float *a=ap+3, *b=bp+7;
            for (int i=0;i<m;++i)
                for (int p=0;p<k;++p) a[(size_t)i*lda+p]=fixture_draw(&state);
            for (int p=0;p<k;++p)
                for (int j=0;j<n;++j) b[(size_t)p*ldb+j]=fixture_draw(&state);
            uint32_t counts[3*1280+2];
            int cert=cob_hk15_regular_certificate(m,n,k,a,lda,b,ldb,counts);
            int policy=cob_hk15_policy_for(m,n,k,a,lda,b,ldb);
            printf("%d,%d,%d,%u,%d,%d\n",m,n,k,seed,cert,policy);
            free(ap); free(bp);
            if (cert!=1 || policy!=COB_HK15_REGULAR) return EXIT_FAILURE;
        }
    }
    return 0;
}
