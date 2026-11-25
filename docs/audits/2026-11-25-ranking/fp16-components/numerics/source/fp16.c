#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#ifndef DEPTH_BALANCE
#define DEPTH_BALANCE 0
#endif
#ifndef ENGINE
#define ENGINE "scalar-fmaf"
#endif
void fp16_block(const _Float16 *a,const _Float16 *b,float *dst,int m,int n,int k,int stride);
void precision_range_rescue(int,int,int,const float*,int,const float*,int,float*,int);

static _Thread_local struct {int route,terms,k; const char *reason;} trace;
int precision_last_route(void){return trace.route;}
int precision_last_products(void){return trace.terms;}
const char *precision_last_reason(void){return trace.reason;}
int precision_last_plan_count(void){return trace.terms?1:0;}
int precision_last_plan_k(int i){return i==0?trace.k:0;}
int precision_last_plan_native(int i){return i==0 && trace.terms;}
const char *precision_last_plan_impl(int i){return i==0?ENGINE:"";}

static int exponent(float x)
{
    uint32_t u;
    memcpy(&u,&x,4);
    u&=0x7fffffff;
    if (!u) return INT_MIN;
    int e=(int)(u>>23);
    if (e==255) return INT_MAX;
    return e?e-127:31-__builtin_clz(u)-149;
}
static void split(float x,_Float16 *v,size_t plane,int count)
{
    for(int q=0;q<count;q++){
        _Float16 h=(_Float16)x;
        v[q*plane]=h;
        if(q+1<count) x=(x-(float)h)*2048.0f;
    }
}
void fp16_check_split(float v,uint16_t *words)
{
    _Float16 h[3];
    split(v,h,1,3);
    memcpy(words,h,sizeof h);
}
uint64_t fp16_control_register(void)
{
#if defined(__aarch64__)
    uint64_t value; __asm__("mrs %0, fpcr":"=r"(value));return value;
#else
    return 0;
#endif
}
static const unsigned char terms[][2]={{2,2},{1,2},{2,1},{0,2},{1,1},{2,0},{0,1},{1,0},{0,0}};
static const unsigned char four[][2]={{1,1},{0,1},{1,0},{0,0}};

int precision_sgemm(int mode,int m,int n,int k,const float *a,int lda,const float *b,int ldb,float *c,int ldc)
{
    trace.route=trace.terms=0;trace.k=k;trace.reason="invalid";
    if(m<0||n<0||k<0||(mode!=1&&mode!=3&&mode!=4&&mode!=6&&mode!=9))return 1;
    if(!m||!n){trace.reason="empty";return 0;}
    if(!c||ldc<n)return 1;
    if(!k){
        for(int i=0;i<m;i++) memset(c+(size_t)i*ldc,0,(size_t)n*sizeof *c);
        trace.reason="zero-k";return 0;
    }
    if(!a||!b||lda<k||ldb<n || n>INT_MAX-3) return 1;
    int bstride=(n+3)&~3, count=mode==1?1:mode<=4?2:3;
    size_t asize=(size_t)m*k,bsize=(size_t)k*bstride,csize=(size_t)m*bstride;
    if(asize>SIZE_MAX/(sizeof(_Float16)*count)||bsize>SIZE_MAX/(sizeof(_Float16)*count)||csize>SIZE_MAX/sizeof(float))return 1;
    int *depth=malloc((size_t)k*sizeof *depth),*rmax=malloc((size_t)m*sizeof *rmax),*bmax=malloc((size_t)n*sizeof *bmax);
    _Float16 *ap=calloc(asize*count,sizeof *ap),*bp=calloc(bsize*count,sizeof *bp);
    float *part=malloc(csize*sizeof *part),*acc=calloc(csize,sizeof *acc);
    const char *rescue=NULL;
    int rc=0;
    if(!depth||!rmax||!bmax||!ap||!bp||!part||!acc){trace.reason="allocation";rc=2;goto done;}
    for(int i=0;i<m;i++)rmax[i]=INT_MIN;
    for(int j=0;j<n;j++)bmax[j]=INT_MIN;
    for(int p=0;p<k;p++){
        int am=INT_MIN,bm=INT_MIN;
        for(int i=0;i<m;i++){int e=exponent(a[(size_t)i*lda+p]);if(e>am)am=e;}
        for(int j=0;j<n;j++){int e=exponent(b[(size_t)p*ldb+j]);if(e>bm)bm=e;}
        if(am==INT_MAX||bm==INT_MAX){rescue="binary64-nonfinite-input";goto done;}
        if(am==INT_MIN||bm==INT_MIN){depth[p]=INT_MIN;continue;}
        int d=bm-am;
        depth[p]=DEPTH_BALANCE?(d>=0?d/2:-((-d+1)/2)):0;
        for(int i=0;i<m;i++){
            int e=exponent(a[(size_t)i*lda+p]);
            if(e!=INT_MIN && e+depth[p]>rmax[i])rmax[i]=e+depth[p];
        }
        for(int j=0;j<n;j++){
            int e=exponent(b[(size_t)p*ldb+j]);
            if(e!=INT_MIN && e-depth[p]>bmax[j])bmax[j]=e-depth[p];
        }
    }
    for(int i=0;i<m;i++)rmax[i]=rmax[i]==INT_MIN?0:12-rmax[i];
    for(int j=0;j<n;j++)bmax[j]=bmax[j]==INT_MIN?0:12-bmax[j];
    for(int p=0;p<k;p++){
        if(depth[p]==INT_MIN)continue;
        for(int i=0;i<m;i++){
            float x=a[(size_t)i*lda+p];int e=exponent(x),s=rmax[i]+depth[p];
            if(e!=INT_MIN&&e+s < -14){rescue="binary64-axis-spread";goto done;}
            split(scalbnf(x,s),ap+(size_t)i*k+p,asize,count);
        }
        for(int j=0;j<n;j++){
            float x=b[(size_t)p*ldb+j];int e=exponent(x),s=bmax[j]-depth[p];
            if(e!=INT_MIN&&e+s < -14){rescue="binary64-axis-spread";goto done;}
            split(scalbnf(x,s),bp+(size_t)p*bstride+j,bsize,count);
        }
    }
    const unsigned char (*order)[2]=mode==4?four:terms+(9-mode);
    for(int t=0;t<mode;t++){
        int l=order[t][0],r=order[t][1];
        fp16_block(ap+(size_t)l*asize,bp+(size_t)r*bsize,part,m,n,k,bstride);
        trace.terms++;
        for(int i=0;i<m;i++)for(int j=0;j<n;j++){
            size_t idx=(size_t)i*bstride+j;
            float weighted=scalbnf(part[idx],-11*(l+r));
            acc[idx]=t?acc[idx]+weighted:weighted;
        }
    }
    for(int i=0;i<m;i++)for(int j=0;j<n;j++){
        float final=scalbnf(acc[(size_t)i*bstride+j],-rmax[i]-bmax[j]);
        if(!isfinite(final)){rescue="binary64-nonfinite-output";goto done;}
        c[(size_t)i*ldc+j]=final;
    }
    trace.route=1;trace.reason=DEPTH_BALANCE?"depth-row-column":"row-column";
done:
    free(depth);free(rmax);free(bmax);free(ap);free(bp);free(part);free(acc);
    if(rescue){
        precision_range_rescue(m,n,k,a,lda,b,ldb,c,ldc);
        trace.route=2;trace.reason=rescue;
    }
    return rc;
}
