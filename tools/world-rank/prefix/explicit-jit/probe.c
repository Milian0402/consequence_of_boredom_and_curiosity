/* Untimed eligibility and full-output test of the explicit oneMKL JIT API. */
#define _GNU_SOURCE
#include <mkl_cblas.h>
#include <mkl_service.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

struct matrix {
    float *mapping, *data;
    size_t bytes, length, origin;
    int rows, cols, ld;
};

static void stop(const char *why)
{
    fprintf(stderr, "%s\n", why);
    exit(1);
}

static struct matrix matrix(int rows, int cols, int ld, int row, int col)
{
    if (rows < 1 || cols < 1 || ld < col+cols || row < 0 || col < 0)
        stop("invalid matrix extent");
    size_t page=(size_t)sysconf(_SC_PAGESIZE), origin=(size_t)row*ld+col;
    size_t bytes=(((size_t)(row+rows+2)*ld*sizeof(float)+page-1)/page)*page;
    float *map=mmap(NULL,bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    if (map==MAP_FAILED) stop("cannot allocate matrix");
    return (struct matrix){map,map+origin,bytes,bytes/sizeof(float),origin,rows,cols,ld};
}

int main(int argc, char **argv)
{
    if (argc != 13 || mkl_get_max_threads()!=1) stop("requires twelve dimensions and one thread");
    int v[12];
    for (int i=0;i<12;++i) {
        char *end;
        long x=strtol(argv[i+1],&end,10);
        if (!argv[i+1][0] || *end || x<0 || x>32768) stop("dimension out of range");
        v[i]=(int)x;
    }
    const int m=v[0],n=v[1],k=v[2];
    struct matrix a=matrix(m,k,v[3],v[6],v[7]), b=matrix(k,n,v[4],v[8],v[9]),
                  c=matrix(m,n,v[5],v[10],v[11]);
    for (size_t t=0;t<a.length;++t) a.mapping[t]=NAN;
    for (size_t t=0;t<b.length;++t) b.mapping[t]=NAN;
    for (size_t t=0;t<c.length;++t) c.mapping[t]=-512.0f;
    for (int i=0;i<m;++i) {
        for (int p=0;p<k;++p) a.data[(size_t)i*a.ld+p]=(float)((i+2*p)%7-3);
        for (int j=0;j<n;++j) c.data[(size_t)i*c.ld+j]=NAN;
    }
    for (int p=0;p<k;++p)
        for (int j=0;j<n;++j)
            b.data[(size_t)p*b.ld+j]=(p==j%k) ? (j%2 ? -1.0f : 1.0f) : 0.0f;
    if (mprotect(a.mapping,a.bytes,PROT_READ) || mprotect(b.mapping,b.bytes,PROT_READ))
        stop("input protection failed");
    void *handle=NULL;
    mkl_jit_status_t status=mkl_jit_create_sgemm(&handle,MKL_ROW_MAJOR,MKL_NOTRANS,MKL_NOTRANS,
        m,n,k,1.0f,a.ld,b.ld,0.0f,c.ld);
    const char *name=status==MKL_JIT_SUCCESS ? "MKL_JIT_SUCCESS" :
                     status==MKL_NO_JIT ? "MKL_NO_JIT" : "MKL_JIT_ERROR";
    if (status!=MKL_JIT_SUCCESS && status!=MKL_NO_JIT) {
        printf("%s,%d,0,0,0\n",name,(int)status);
        return 0;
    }
    if (!handle) stop("missing explicit JIT handle");
    sgemm_jit_kernel_t kernel=mkl_jit_get_sgemm_ptr(handle);
    if (!kernel) stop("missing explicit callable");
    kernel(handle,a.data,b.data,c.data);
    size_t checked=0, guards=0;
    for (size_t t=0;t<c.length;++t) {
        if (t>=c.origin) {
            size_t u=t-c.origin,i=u/c.ld,j=u%c.ld;
            if (i<(size_t)m && j<(size_t)n) {
                int value=(int)((i+2*(j%(size_t)k))%7)-3;
                float expected=(float)(j%2 ? -value : value);
                if (c.mapping[t]!=expected) stop("wrong SGEMM output");
                ++checked;
                continue;
            }
        }
        if (c.mapping[t]!=-512.0f) stop("C guard changed");
        ++guards;
    }
    if (mkl_jit_destroy(handle)!=MKL_JIT_SUCCESS) stop("JIT handle destroy failed");
    printf("%s,%d,1,%zu,%zu\n",name,(int)status,checked,guards);
    return 0;
}
