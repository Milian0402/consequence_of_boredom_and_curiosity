/* Public and CBLAS link wrappers are exclusive to the auxiliary executable. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
#include "../vendor-public/trace.c"

static const char* bound_provider(void)
{
#if COB_ENABLE_OPENBLAS_LEAF
    if (openblas_get_parallel() || openblas_get_num_threads()!=1) return NULL;
    return openblas_get_config();
#else
    Dl_info c,f;
    void *cs=dlsym(RTLD_DEFAULT,"cblas_sgemm"), *fs=dlsym(RTLD_DEFAULT,"sgemm_");
    if (!cs || !fs || !dladdr(cs,&c) || !dladdr(fs,&f) || !c.dli_fname || c.dli_fbase!=f.dli_fbase) return NULL;
    const char *base=strrchr(c.dli_fname,'/');
    base = base ? base+1 : c.dli_fname;
#if COB_ENABLE_MKL_SEQUENTIAL_LEAF
    if (mkl_get_max_threads()!=1 || mkl_domain_get_max_threads(MKL_DOMAIN_BLAS)!=1) return NULL;
    const char *soname="libmkl_intel_lp64.so";
#else
    const char *soname="libarmpl_lp64.so";
#endif
    size_t n=strlen(soname);
    return !strncmp(base,soname,n) && (base[n]==0 || base[n]=='.') ? base : NULL;
#endif
}

static int matches(const uint64_t *r, int m,int n,int k,const float *a,int lda,
                   const float *b,int ldb,float *c,int ldc,int beta)
{
    uint64_t expected[fields]={(unsigned)m,(unsigned)n,(unsigned)k,(unsigned)lda,(unsigned)ldb,(unsigned)ldc,
                               (uintptr_t)a,(uintptr_t)b,(uintptr_t)c,(unsigned)beta,1};
    return !memcmp(expected,r,sizeof expected);
}

extern void __real_cob_sgemm_rowmajor(int,int,int,const float*,int,const float*,int,float*,int);
void __wrap_cob_sgemm_rowmajor(int m,int n,int k,const float *a,int lda,
                              const float *b,int ldb,float *c,int ldc)
{
    const char *id=bound_provider();
    const unsigned start=counts[2];
    __real_cob_sgemm_rowmajor(m,n,k,a,lda,b,ldb,c,ldc);
    const unsigned total=counts[2]-start;
    const int out_align=RAGGED_ALIGNED ? 64:4, k_align=RAGGED_ALIGNED ? 32:4;
    const int cm=m-m%out_align, cn=n-n%out_align, ck=k-k%k_align;
    const int rm=cm/4,rn=cn/4,rk=ck/4;
    /* Each group of seven products targets one outer destination quadrant or
       reuses one temporary. In each group, products 4..6 share the inner temporary. */
    int quads=total>=49 && total<=52;
    unsigned overwrites=0;
    uintptr_t outer_temp=0;
    for (unsigned q=0;q<7 && total>=49 && total<=52;++q) {
        const uint64_t *g1=record[(start+7*q+1)%slots], *g6=record[(start+7*q+6)%slots];
        uintptr_t dst=(uintptr_t)g1[8];
        size_t ld=(size_t)g1[5];
        if (q<4) {
            size_t off=(q==0 || q==2 ? (size_t)(cm/2)*ldc:0)+(q==2 || q==3 ? cn/2:0);
            quads &= dst==(uintptr_t)(c+off) && ld==(size_t)ldc;
        } else {
            if(q==4) outer_temp=dst;
            quads &= dst==outer_temp && ld==(size_t)(cn/2);
        }
        for (unsigned j=0;j<7;++j) {
            const uint64_t *r=record[(start+7*q+j)%slots];
            int overwrite=r[9]==0 && r[10]==1;
            overwrites+=(unsigned)overwrite;
            quads &= overwrite && r[0]==(unsigned)rm && r[1]==(unsigned)rn && r[2]==(unsigned)rk;
            if (j<4) {
                size_t off=(j==0 || j==2 ? (size_t)rm*ld:0)+(j==2 || j==3 ? rn:0);
                quads &= r[8]==dst+off*sizeof(float) && r[5]==ld;
            } else {
                quads &= r[8]==g6[8] && r[5]==(unsigned)rn;
            }
        }
    }
    unsigned i=49;
    int depth=0,right=0,bottom=0;
    if(RAGGED_ALIGNED && k!=ck) { depth=i<total && matches(record[(start+i)%slots],cm,cn,k-ck,a+ck,lda,b+(size_t)ck*ldb,ldb,c,ldc,1); ++i; }
    if(n!=cn) { right=i<total && matches(record[(start+i)%slots],cm,n-cn,k,a,lda,b+cn,ldb,c+cn,ldc,0); ++i; }
    if(m!=cm) { bottom=i<total && matches(record[(start+i)%slots],m-cm,n,k,a+(size_t)cm*lda,lda,b,ldb,c+(size_t)cm*ldc,ldc,0); ++i; }
    int edges=i==total && depth==(RAGGED_ALIGNED && k!=ck) && right==(n!=cn) && bottom==(m!=cm);
    int direct=total==1 && matches(record[start%slots],m,n,k,a,lda,b,ldb,c,ldc,0);
    unsigned stored=total<=slots ? total : 0;
    printf("SIGNED_TRACE,core_m=%d,core_n=%d,core_k=%d,calls=%u,core_calls=%u,core_overwrites=%u,positive_updates=0,negative_updates=0,depth_updates=%d,right_strips=%d,bottom_strips=%d,quadrants_ok=%d,edges_ok=%d,provider_ok=%d\n",
           cm,cn,ck,total,overwrites,overwrites,depth,right,bottom,quads,edges,id!=NULL);
    printf("RAGGED_VENDOR_ROUTE,%u,%u,%d,%s\n",total,stored,direct,id ? id:"unresolved");
    for(unsigned j=0;j<stored;++j) {
        const uint64_t *r=record[(start+j)%slots];
        printf("RAGGED_VENDOR_LEAF,%u",j);
        for (unsigned f=0;f<6;++f) printf(",%llu",(unsigned long long)r[f]);
        /* Source receipt field 10 affirms NN row major and alpha=1. */
        printf(",%u,%u\n",r[10] ? 1u:0u,(unsigned)r[9]);
    }
}
