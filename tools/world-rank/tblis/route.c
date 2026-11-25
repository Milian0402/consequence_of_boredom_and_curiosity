#include <stdio.h>
#include <stdlib.h>

int rank_vendor_init(void);
void rank_vendor_sgemm(int,int,int,const float*,int,const float*,int,float*,int);
unsigned long long rank_tblis_tiles(void);

int main(void)
{
    enum { m=24,n=64,k=32 };
    float a[m*k], b[k*n], c[m*n];
    for (int i=0;i<m*k;++i) a[i]=1;
    for (int i=0;i<k*n;++i) b[i]=2;
    if (rank_vendor_init()) return 1;
    rank_vendor_sgemm(m,n,k,a,k,b,n,c,n);
    for (int i=0;i<m*n;++i) if (c[i]!=2*k) return 2;
    const unsigned long long calls=rank_tblis_tiles();
    const unsigned long long expected=TBLIS_AB_LEVEL==1 ? 7*2*2 : 49;
    printf("level=%d m=%d n=%d k=%d haswell_16x6_tiles=%llu expected=%llu\n",
        TBLIS_AB_LEVEL,m,n,k,calls,expected);
    return calls!=expected;
}
