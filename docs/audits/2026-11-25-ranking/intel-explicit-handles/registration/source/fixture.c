#define _GNU_SOURCE
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
struct mat {float *base,*v; size_t bytes;int rows,cols,ld,r,c;};
struct fixture {struct mat a,b,c;double *ref;size_t checked,guards;double error;};
static struct mat make(int rows,int cols,int ld,int r,int c)
{
    if(rows<1 || cols<1 || r<0 || c<0 || ld<cols+c) abort();
    size_t page=(size_t)sysconf(_SC_PAGESIZE), bytes=(((size_t)(r+rows+2)*ld*4+page-1)/page)*page;
    char *map=mmap(NULL,bytes+2*page,PROT_NONE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    if (map==MAP_FAILED || mprotect(map+page,bytes,PROT_READ|PROT_WRITE)) abort();
    float *base=(float *)(map+page);
    for(size_t q=0;q<bytes/4;++q) base[q]=-987.25f;
    return (struct mat){base,base+(size_t)r*ld+c,bytes,rows,cols,ld,r,c};
}
static void unmap(struct mat m)
{
    size_t page=(size_t)sysconf(_SC_PAGESIZE);
    if(munmap((char *)m.base-page,m.bytes+2*page)) abort();
}
struct fixture *fixture_new(int m,int n,int k,int lda,int ldb,int ldc,int ar,int ac,int br,int bc,int cr,int cc,int seed)
{
    struct fixture *f=calloc(1,sizeof *f);if(!f) abort();
    f->a=make(m,k,lda,ar,ac); f->b=make(k,n,ldb,br,bc); f->c=make(m,n,ldc,cr,cc);
    f->ref=calloc((size_t)m*n,sizeof(double));if(!f->ref)abort();
    for(int i=0;i<m;++i) for(int p=0;p<k;++p) f->a.v[(size_t)i*lda+p]=(float)((i*29+p*73+seed)%97-48)/83.0f;
    for(int p=0;p<k;++p) for(int j=0;j<n;++j) f->b.v[(size_t)p*ldb+j]=(float)((p*41+j*31+3*seed)%101-50)/87.0f;
    for(int i=0;i<m;++i) {
        for(int j=0;j<n;++j) f->c.v[(size_t)i*ldc+j]=NAN;
        for(int p=0;p<k;++p) {
            double av=f->a.v[(size_t)i*lda+p];
            for(int j=0;j<n;++j) f->ref[(size_t)i*n+j]+=av*f->b.v[(size_t)p*ldb+j];
        }
    }
    if(mprotect(f->a.base,f->a.bytes,PROT_READ)||mprotect(f->b.base,f->b.bytes,PROT_READ))abort();
    return f;
}
void fixture_addresses(struct fixture *f,uintptr_t *out) {
    out[0]=(uintptr_t)f->a.v;out[1]=(uintptr_t)f->b.v;out[2]=(uintptr_t)f->c.v;
}
int fixture_check(struct fixture *f,size_t *checked,size_t *guards,double *error)
{
    *checked=*guards=0; *error=0; struct mat c=f->c;
    for(size_t t=0;t<c.bytes/4;++t) {
        int i=(int)(t/c.ld)-c.r,j=(int)(t%c.ld)-c.c;
        if(i>=0 && j>=0 && i<c.rows && j<c.cols) {
            double e=fabs(c.base[t]-f->ref[(size_t)i*c.cols+j]);
            if(!isfinite(e) || e>0.002) return 0;
            if(e>*error)*error=e;
            (*checked)++;
        } else {if(c.base[t]!=-987.25f) return 0;(*guards)++;}
    }
    return 1;
}
void fixture_free(struct fixture *f) {unmap(f->a);unmap(f->b);unmap(f->c);free(f->ref);free(f);}
void fixture_reset(struct fixture *f) {
    for(int i=0;i<f->c.rows;++i) for(int j=0;j<f->c.cols;++j) f->c.v[(size_t)i*f->c.ld+j]=NAN;
}
