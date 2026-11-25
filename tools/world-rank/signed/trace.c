/* Separately linked fixture witness; no wrap or counter enters the timed executable. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#if COB_ENABLE_OPENBLAS_LEAF
#include <cblas.h>
typedef enum CBLAS_ORDER order_t;
typedef enum CBLAS_TRANSPOSE trans_t;
typedef blasint index_t;
#elif COB_ENABLE_MKL_SEQUENTIAL_LEAF
#include <mkl.h>
typedef CBLAS_LAYOUT order_t;
typedef CBLAS_TRANSPOSE trans_t;
typedef MKL_INT index_t;
#else
#include <armpl.h>
typedef CBLAS_LAYOUT order_t;
typedef enum CBLAS_TRANSPOSE trans_t;
typedef armpl_int_t index_t;
#endif
typedef struct { const float *a,*b; float *c; int m,n,k,lda,ldb,ldc; float alpha,beta; } call;
static call calls[52];
static unsigned total;
static int abi_ok;
extern __typeof__(cblas_sgemm) __real_cblas_sgemm;
void __wrap_cblas_sgemm(order_t order, trans_t ta, trans_t tb,index_t m,index_t n,index_t k,float alpha,
    const float* a,index_t lda,const float* b,index_t ldb,float beta,float* c,index_t ldc)
{
    if(total < sizeof(calls)/sizeof(calls[0]))
        calls[total]=(call){a,b,c,m,n,k,lda,ldb,ldc,alpha,beta};
    ++total;
    abi_ok &= order==CblasRowMajor && ta==CblasNoTrans && tb==CblasNoTrans;
    __real_cblas_sgemm(order,ta,tb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc);
}

static int provider_ok(void)
{
#if COB_ENABLE_OPENBLAS_LEAF
    return openblas_get_parallel()==0 && openblas_get_num_threads()==1;
#else
    Dl_info ci,fi;
    void* cs=dlsym(RTLD_DEFAULT,"cblas_sgemm");
    void* fs=dlsym(RTLD_DEFAULT,"sgemm_");
    if(!cs || !fs || !dladdr(cs,&ci) || !dladdr(fs,&fi) || !ci.dli_fname || ci.dli_fbase!=fi.dli_fbase) return 0;
    const char* name=strrchr(ci.dli_fname,'/'); name=name?name+1:ci.dli_fname;
#if COB_ENABLE_MKL_SEQUENTIAL_LEAF
    const char* prefix="libmkl_intel_lp64.so";
    if(mkl_get_max_threads()!=1 || mkl_domain_get_max_threads(MKL_DOMAIN_BLAS)!=1) return 0;
#else
    const char* prefix="libarmpl_lp64.so";
#endif
    size_t len=strlen(prefix);
    return !strncmp(name,prefix,len) && (!name[len] || name[len]=='.');
#endif
}

static int edge(const call* x,const float* a,const float* b,float* c,
    int m,int n,int k,int lda,int ldb,int ldc,int beta)
{
    return x->a==a && x->b==b && x->c==c && x->m==m && x->n==n && x->k==k &&
        x->lda==lda && x->ldb==ldb && x->ldc==ldc && x->alpha==1 && x->beta==beta;
}

extern void __real_cob_sgemm_rowmajor(int,int,int,const float*,int,const float*,int,float*,int);
void __wrap_cob_sgemm_rowmajor(int m,int n,int k,const float* a,int lda,const float* b,int ldb,float* c,int ldc)
{
    total=0; abi_ok=1;
    const int identity=provider_ok();
    __real_cob_sgemm_rowmajor(m,n,k,a,lda,b,ldb,c,ldc);
    const int cm=m-m%64,cn=n-n%64,ck=k-k%32,rm=cm/4,rn=cn/4,rk=ck/4;
    int quad=abi_ok && identity && total>=49 && total<=52;
    int overwrites=0,pos=0,neg=0;
    float* parent_temp=NULL;
    for(int q=0;q<7 && total>=49 && total<=52;++q){
        const call* g=calls+7*q;
        float* start=g[1].c;
        int stride=g[1].ldc;
        if(q<4){
            const size_t offset=(q==0 || q==2 ? (size_t)(cm/2)*ldc:0)+(q==2 || q==3 ? cn/2:0);
            quad &= start==c+offset && stride==ldc;
        }else{
            if(q==4) parent_temp=start;
            quad &= start==parent_temp && stride==cn/2;
        }
        for(int j=0;j<7;++j){
            const call* x=g+j;
            quad &= x->m==rm && x->n==rn && x->k==rk;
            if(x->beta==0 && x->alpha==1) ++overwrites;
            if(x->beta==1 && x->alpha==1) ++pos;
            if(x->beta==1 && x->alpha==-1) ++neg;
            int signed_here=SIGNED_MODE && (j==4 || j==5);
            quad &= x->beta==signed_here && x->alpha==(signed_here && j==5 ? -1:1);
            if(j<4 || signed_here){
                int quadrant=j<4 ? j : (j==4 ? 3:0);
                size_t delta=(quadrant==0 || quadrant==2 ? (size_t)rm*stride:0)+(quadrant==2 || quadrant==3 ? rn:0);
                quad &= x->c==start+delta && x->ldc==stride;
            }else{
                quad &= x->c==g[6].c && x->ldc==rn;
            }
        }
    }
    unsigned i=49;
    int es=abi_ok && total>=49 && total<=52;
    int depth=0,right=0,bottom=0;
    if(k!=ck){depth=i<total && i<52 && edge(calls+i,a+ck,b+(size_t)ck*ldb,c,cm,cn,k-ck,lda,ldb,ldc,1); es&=depth; ++i;}
    if(n!=cn){right=i<total && i<52 && edge(calls+i,a,b+cn,c+cn,cm,n-cn,k,lda,ldb,ldc,0); es&=right; ++i;}
    if(m!=cm){bottom=i<total && i<52 && edge(calls+i,a+(size_t)cm*lda,b,c+(size_t)cm*ldc,m-cm,n,k,lda,ldb,ldc,0); es&=bottom; ++i;}
    es &= i==total;
    printf("SIGNED_TRACE,core_m=%d,core_n=%d,core_k=%d,calls=%u,core_calls=%d,core_overwrites=%d,positive_updates=%d,negative_updates=%d,depth_updates=%d,right_strips=%d,bottom_strips=%d,quadrants_ok=%d,edges_ok=%d,provider_ok=%d\n",
           cm,cn,ck,total,overwrites+pos+neg,overwrites,pos,neg,depth,right,bottom,quad,es,identity);
}
