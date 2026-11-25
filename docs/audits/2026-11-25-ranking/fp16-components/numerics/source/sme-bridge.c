#include <stdlib.h>

int apple_half_plane(int m,int n,int k,const _Float16 *a,int lda,const _Float16 *b,int ldb,float *dst,int ldc);
void fp16_block(const _Float16 *a,const _Float16 *b,float *dst,int m,int n,int k,int stride)
{
    if(apple_half_plane(m,n,k,a,k,b,stride,dst,stride)!=1)abort();
}
