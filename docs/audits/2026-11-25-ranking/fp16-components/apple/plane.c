#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
int cob_half_svl(void);
void apple_half_tiles(int,int,int,int,const _Float16*,const _Float16*,const _Float16*,const _Float16*,float*,int);

/* Adapter for a single binary16 plane from the independent common splitter. */
int apple_half_plane(int m,int n,int k,const _Float16 *a,int lda,const _Float16 *b,int ldb,float *c,int ldc)
{
    if (m<=0 || n<=0 || k<=0 || !a || !b || !c || lda<k || ldb<n || ldc<n || !cob_half_svl()) return 0;
    const size_t kp=(size_t)k+(k&1), mp=((size_t)m+31)&~(size_t)31, np=((size_t)n+31)&~(size_t)31;
    if (mp+np > SIZE_MAX/sizeof(_Float16)/kp) return 0;
    const size_t count=(mp+np)*kp;
    _Float16 *buf=calloc(count,sizeof(*buf));
    if (!buf) return 0;
    _Float16 *bp=buf+mp*kp;
    for (int i=0;i<m;++i)
        for(int p=0;p<k;++p)
            buf[(size_t)(i/32)*32*kp+(size_t)(p/2)*64+(i%32)*2+(p&1)]=a[(size_t)i*lda+p];
    for (int p=0;p<k;++p)
        for(int j=0;j<n;++j)
            bp[(size_t)(j/32)*32*kp+(size_t)(p/2)*64+(j%32)*2+(p&1)]=b[(size_t)p*ldb+j];
    apple_half_tiles(0,m,n,k,buf,NULL,bp,NULL,c,ldc);
    free(buf);
    return 1;
}
