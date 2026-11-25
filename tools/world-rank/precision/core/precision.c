/* Single-call component-product experiments, C11. All inputs and outputs are row-major. */
#include "precision.h"
#include "../backend/backend.h"
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>


static _Thread_local struct {
    int route, executions, plans;
    const char *reason;
    char implementation[3][192];
    int k[3], native[3];
} last;

int precision_last_route(void) { return last.route; }
int precision_last_products(void) { return last.executions; }
const char *precision_last_reason(void) { return last.reason ? last.reason : "no-call"; }
int precision_last_plan_count(void) { return last.plans; }
int precision_last_plan_native(int i) { return (i>=0 && i<last.plans) ? last.native[i] : 0; }
int precision_last_plan_k(int i) { return (i>=0 && i<last.plans) ? last.k[i] : 0; }
const char *precision_last_plan_impl(int i) { return (i>=0 && i<last.plans) ? last.implementation[i] : ""; }
static uint32_t bits(float x) { uint32_t v; memcpy(&v,&x,4); return v; }
static float expand(uint16_t h) { uint32_t v=(uint32_t)h<<16; float x; memcpy(&x,&v,4); return x; }
uint16_t precision_bf16_rne(float x)
{
    uint32_t v=bits(x);
    if ((v&UINT32_C(0x7fffffff))>UINT32_C(0x7f800000)) return (uint16_t)((v>>16)|0x40u);
    return (uint16_t)((v+0x7fffu+((v>>16)&1u))>>16);
}

/* Normal finite research domain bounds all nonzero residual products away from FTZ.
   Test unsigned absolute bits to detect infinities/NaNs even on fast SIMD targets. */
static int split(const float *source,int rows,int cols,int ld,uint16_t **planes,int count)
{
    unsigned int unsafe=0;
    for(int r=0;r<rows;++r){
        size_t base=(size_t)r*cols;
        for(int j=0;j<cols;++j){
            const float x=source[(size_t)r*ld+j];
            size_t at=base+j;
            const uint16_t hi=precision_bf16_rne(x);
            planes[0][at]=hi;
            if(count==1) continue;
            const uint32_t mag=bits(x)&UINT32_C(0x7fffffff);
            if(mag!=0 && (mag<UINT32_C(0x30800000) || mag>UINT32_C(0x4e800000))){unsafe=1;continue;}
            const float r1=x-expand(hi);
            const uint16_t mid=precision_bf16_rne(r1);
            planes[1][at]=mid;
            if(count==3){
                const float r2=r1-expand(mid);
                const uint16_t lo=precision_bf16_rne(r2);
                planes[2][at]=lo;
                unsafe |= r2!=expand(lo);
            }
        }
    }
    return !unsafe;
}

static void *allocate(size_t n,size_t width)
{ return width && n<=SIZE_MAX/width ? malloc(n*width) : NULL; }

static int remember(void *plan,int k)
{
    if(!plan){last.reason="backend-create";return 0;}
    int i=last.plans++;
    last.k[i]=k;
    last.native[i]=bf16_plan_native(plan);
    const char *name=bf16_plan_implementation(plan);
    if(name){strncpy(last.implementation[i],name,sizeof(last.implementation[i])-1);}
#ifdef PRECISION_NATIVE_ONLY
    if(!last.native[i]){last.reason="unadmitted-backend-route";return 0;}
#endif
    return 1;
}

static int product(void *plan,const uint16_t *a,const uint16_t *b,float *result)
{
    if(bf16_plan_exec(plan,a,b,result)){last.reason="backend-execution";return 0;}
    ++last.executions;
    return 1;
}

static void merge(const float *part,float *sum,float *c,int m,int n,int ldc,int first,int final)
{
    for(int r=0;r<m;++r){
        size_t at=(size_t)r*n;
        float *dest=final ? c+(size_t)r*ldc : sum+at;
        for(int j=0;j<n;++j) dest[j]=first ? part[at+j] : sum[at+j]+part[at+j];
    }
}

static const unsigned char pairs[][2]={{2,2},{1,2},{2,1},{0,2},{1,1},{2,0},{0,1},{1,0},{0,0}};

int precision_sgemm(int mode,int m,int n,int k,const float *a,int lda,const float *b,int ldb,float *c,int ldc)
{
    memset(&last,0,sizeof(last));last.reason="invalid-argument";
    int count=mode==1?1:mode==3?2:3;
    if((mode!=1 && mode!=3 && mode!=6 && mode!=9 && mode!=16) || m<0 || n<0 || k<0) return 1;
    if(!m || !n){last.reason="empty-output";return 0;}
    if(!c || ldc<n) return 1;
    if(!k){for(int r=0;r<m;++r) memset(c+(size_t)r*ldc,0,(size_t)n*sizeof(float));last.reason="zero-k";return 0;}
    if(!a || !b || lda<k || ldb<n) return 1;
    const size_t ac=(size_t)m*k,bc=(size_t)k*n,cc=(size_t)m*n;
    uint16_t *ap[3]={0},*bp[3]={0},*aa=NULL,*bb=NULL,*wide=NULL;
    float *sum=NULL,*partial=NULL;
    void *plan=NULL;
    int code=2;
    last.reason="workspace";
    if(mode==16 && k>INT_MAX/3) goto cleanup;
    aa=allocate(ac,count*sizeof(uint16_t)); bb=allocate(bc,count*sizeof(uint16_t));
    partial=allocate(cc,sizeof(float));
    if(mode!=1)sum=allocate(cc,sizeof(float));
    if(mode==16)wide=allocate(ac,3*sizeof(uint16_t));
    if(!aa || !bb || !partial || (mode!=1 && !sum) || (mode==16 && !wide))goto cleanup;
    for(int i=0;i<count;++i){ap[i]=aa+(size_t)i*ac;bp[i]=bb+(size_t)(count-1-i)*bc;}
    if(!split(a,m,k,lda,ap,count) || !split(b,k,n,ldb,bp,count)){
        last.reason="fp64-range-rescue";last.route=2;code=0;
        goto cleanup;
    }
    if(mode==16){
        for(int degree=2;degree>=0;--degree){
            const int terms=degree+1;
            const uint16_t *left=ap[0];
            if(degree){
                for(int r=0;r<m;++r)
                    for(int i=0;i<terms;++i) memcpy(wide+((size_t)r*terms+i)*k,ap[i]+(size_t)r*k,(size_t)k*sizeof(uint16_t));
                left=wide;
            }
            plan=bf16_plan_create(m,n,terms*k);
            if(!remember(plan,terms*k) || !product(plan,left,bp[degree],partial)) goto cleanup;
            merge(partial,sum,c,m,n,ldc,degree==2,degree==0);
            bf16_plan_destroy(plan);plan=NULL;
        }
    } else {
        plan=bf16_plan_create(m,n,k);
        if(!remember(plan,k))goto cleanup;
        const int start=9-mode;
        for(int t=start;t<9;++t){
            if(!product(plan,ap[pairs[t][0]],bp[pairs[t][1]],partial)) goto cleanup;
            merge(partial,sum,c,m,n,ldc,t==start,t==8);
        }
    }
    last.route=1;last.reason="component-product";code=0;
cleanup:
    if(plan)bf16_plan_destroy(plan);
    free(wide);free(sum);free(partial);free(bb);free(aa);
    if(last.route==2)precision_range_rescue(m,n,k,a,lda,b,ldb,c,ldc);
    return code;
}
