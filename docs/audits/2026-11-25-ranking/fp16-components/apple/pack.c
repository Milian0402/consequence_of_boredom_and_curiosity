#define _POSIX_C_SOURCE 200112L
#include <fenv.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

extern int cob_half_svl(void);
extern void apple_half_tiles(int,int,int,int,const _Float16 *,const _Float16 *,
                             const _Float16 *,const _Float16 *,float *,int);

enum apple_half_status { HALF_DONE=0, HALF_EMPTY=1, HALF_INVALID=2, HALF_UNSUPPORTED=3,
    HALF_SIZE=4, HALF_NO_MEMORY=5, HALF_NONFINITE=6, HALF_RANGE=7, HALF_OUTPUT=8 };
static _Thread_local int last_route;
static _Thread_local int last_status;
int apple_fp16_last_route(void) { return last_route; }
int apple_fp16_last_status(void) { return last_status; }

static int decline(int status) { last_status=status; return 0; }

static int matrix_end(const float *p,int rows,int cols,int ld,uintptr_t *end)
{
    if (!p || ld<cols) return 0;
    const uint64_t length=(uint64_t)(rows-1)*(unsigned)ld+(unsigned)cols;
    if (length>SIZE_MAX/sizeof(float)) return 0;
    const size_t bytes=(size_t)length*sizeof(float);
    if (bytes-1>UINTPTR_MAX-(uintptr_t)p) return 0;
    *end=(uintptr_t)p+bytes-1;
    return 1;
}

static int reserve(size_t *used,size_t count,size_t width,size_t *offset)
{
    if (*used>SIZE_MAX-63) return 0;
    const size_t start=(*used+63)&~(size_t)63;
    if (count>(SIZE_MAX-start)/width) return 0;
    *offset=start; *used=start+count*width;
    return 1;
}

static uint32_t magnitude(float x)
{
    uint32_t u;
    memcpy(&u,&x,sizeof u);
    return u&UINT32_C(0x7fffffff);
}

static int exponent(uint32_t u)
{
    const unsigned exponent_field=u>>23;
    return exponent_field ? (int)exponent_field-127 : 31-__builtin_clz(u)-149;
}

struct scale { float high,low; int shift; };
static int axis_scale(uint32_t smallest,uint32_t largest,struct scale *s)
{
    if (largest>=UINT32_C(0x7f800000)) return HALF_NONFINITE;
    if (!largest) { *s=(struct scale){1,1,0}; return HALF_DONE; }
    int max_e=exponent(largest);
    if (max_e-exponent(smallest)>26) return HALF_RANGE;
    int e=12-max_e;
    int main=e>126 ? 126 : e;
    *s=(struct scale){scalbnf(1.0f,main),scalbnf(1.0f,e-main),e};
    return HALF_DONE;
}

static int scales(int m,int n,int k,const float *a,int lda,const float *b,int ldb,
                  struct scale *as,struct scale *bs,uint32_t *lo,uint32_t *hi)
{
    for (int i=0;i<m;i++) {
        uint32_t low=UINT32_MAX,high=0;
        const float *r=a+(size_t)i*lda;
        for (int p=0;p<k;p++) {
            uint32_t u=magnitude(r[p]);
            if (u && u<low) low=u;
            if (u>high) high=u;
        }
        int st=axis_scale(low,high,as+i);
        if (st) return st;
    }
    for (int j=0;j<n;j++) {lo[j]=UINT32_MAX;hi[j]=0;}
    for (int p=0;p<k;p++) {
        const float *r=b+(size_t)p*ldb;
        for (int j=0;j<n;j++) {
            uint32_t u=magnitude(r[j]);
            if (u && u<lo[j]) lo[j]=u;
            if (u>hi[j]) hi[j]=u;
        }
    }
    for (int j=0;j<n;j++) {
        int st=axis_scale(lo[j],hi[j],bs+j);
        if (st) return st;
    }
    return HALF_DONE;
}

static inline void parts(float v,_Float16 *h,_Float16 *l)
{
    const _Float16 top=(_Float16)v;
    *h=top;
    if (l) *l=(_Float16)((v-(float)top)*2048.0f);
}

static void pack_planes(int m,int n,int k,int lda,int ldb,const float *a,const float *b,
                        const struct scale *as,const struct scale *bs,size_t tile,size_t pk,
                        _Float16 *ah,_Float16 *al,_Float16 *bh,_Float16 *bl)
{
    /* Each two-depth pair has two adjacent FP16 values per SME input lane. */
    for (size_t first=0;first<(size_t)m;first+=tile) {
        size_t length=(size_t)m-first < tile ? (size_t)m-first : tile;
        for (size_t lane=0;lane<length;lane++) {
            size_t row=first+lane;
            const float *r=a+row*(size_t)lda;
            const float high=as[row].high,low=as[row].low;
            size_t off=first*pk+2*lane;
            for (size_t p=0;p<(size_t)k;p+=2,off+=2*tile) {
                parts((r[p]*high)*low,ah+off,al?al+off:NULL);
                if (p+1<(size_t)k) parts((r[p+1]*high)*low,ah+off+1,al?al+off+1:NULL);
            }
        }
    }
    for (size_t p=0;p<(size_t)k;p+=2) {
        const float *r=b+p*(size_t)ldb;
        for (size_t first=0;first<(size_t)n;first+=tile) {
            size_t length=(size_t)n-first < tile ? (size_t)n-first : tile;
            size_t off=first*pk+p*tile;
            for (size_t lane=0;lane<length;lane++,off+=2) {
                const size_t j=first+lane;
                const float high=bs[j].high,low=bs[j].low;
                parts((r[j]*high)*low,bh+off,bl?bl+off:NULL);
                if (p+1<(size_t)k) parts((r[(size_t)ldb+j]*high)*low,bh+off+1,bl?bl+off+1:NULL);
            }
        }
    }
}

int cob_half_sgemm(int mode,int m,int n,int k,const float *a,int lda,const float *b,int ldb,
                   float *c,int ldc)
{
    last_route=0;last_status=HALF_DONE;
    if (mode<0 || mode>1 || m<0 || n<0 || k<0) return decline(HALF_INVALID);
    if (!m || !n) {last_status=HALF_EMPTY;return 1;}
    uintptr_t ce,ae,be;
    if (!matrix_end(c,m,n,ldc,&ce)) return decline(HALF_INVALID);
    if (!k) {
        for (int i=0;i<m;i++) memset(c+(size_t)i*ldc,0,(size_t)n*sizeof(float));
        last_status=HALF_EMPTY;return 1;
    }
    if (!matrix_end(a,m,k,lda,&ae) || !matrix_end(b,k,n,ldb,&be) ||
        ((uintptr_t)c<=ae && (uintptr_t)a<=ce) || ((uintptr_t)c<=be && (uintptr_t)b<=ce)) return decline(HALF_INVALID);
    if (fegetround()!=FE_TONEAREST) return decline(HALF_UNSUPPORTED);
    int svl=cob_half_svl();
    if (svl<1 || svl>64) return decline(HALF_UNSUPPORTED);
    size_t tile=2*(size_t)svl,pk=(size_t)k+((unsigned)k&1);
    size_t pm=((size_t)m+tile-1)/tile*tile,pn=((size_t)n+tile-1)/tile*tile;
    if (pm>SIZE_MAX/pk || pn>SIZE_MAX/pk || (size_t)m>SIZE_MAX/(size_t)n) return decline(HALF_SIZE);
    size_t used=0,ai,bi,ali=0,bli=0,ci,asi,bsi,loi,hii;
    if (!reserve(&used,pm*pk,sizeof(_Float16),&ai) || !reserve(&used,pn*pk,sizeof(_Float16),&bi) ||
        (mode && (!reserve(&used,pm*pk,sizeof(_Float16),&ali) || !reserve(&used,pn*pk,sizeof(_Float16),&bli))) ||
        !reserve(&used,(size_t)m*n,sizeof(float),&ci) ||
        !reserve(&used,m,sizeof(struct scale),&asi) || !reserve(&used,n,sizeof(struct scale),&bsi) ||
        !reserve(&used,n,sizeof(uint32_t),&loi) || !reserve(&used,n,sizeof(uint32_t),&hii)) return decline(HALF_SIZE);
    void *allocation=NULL;
    if (posix_memalign(&allocation,64,used)) return decline(HALF_NO_MEMORY);
    unsigned char *mem=allocation;
    struct scale *as=(void *)(mem+asi),*bs=(void *)(mem+bsi);
    int status=scales(m,n,k,a,lda,b,ldb,as,bs,(void *)(mem+loi),(void *)(mem+hii));
    if (status) {free(mem);return decline(status);}
    _Float16 *ah=(void *)(mem+ai),*bh=(void *)(mem+bi);
    _Float16 *al=mode?(_Float16 *)(mem+ali):NULL,*bl=mode?(_Float16 *)(mem+bli):NULL;
    /* Zero every padded lane, including the second half of an odd final depth. */
    memset(mem,0,ci);
    pack_planes(m,n,k,lda,ldb,a,b,as,bs,tile,pk,ah,al,bh,bl);
    float *output=(void *)(mem+ci);
    apple_half_tiles(mode,m,n,k,ah,al,bh,bl,output,n);
    for (int i=0;i<m;i++) {
        float *row=output+(size_t)i*n;
        for (int j=0;j<n;j++) {
            int restore=-as[i].shift-bs[j].shift;
            float v;
            if (restore>=-126 && restore<=127) {
                uint32_t word=(uint32_t)(restore+127)<<23;
                float factor;memcpy(&factor,&word,sizeof factor);
                v=row[j]*factor;
            } else v=scalbnf(row[j],restore);
            if (!isfinite(v)) {free(mem);return decline(HALF_OUTPUT);}
            row[j]=v;
        }
    }
    for (int i=0;i<m;i++) memcpy(c+(size_t)i*ldc,output+(size_t)i*n,(size_t)n*sizeof(float));
    free(mem);last_route=mode?3:1;return 1;
}

int apple_fp16_raw(int m,int n,int k,const float *a,int lda,const float *b,int ldb,float *c,int ldc)
{
    return cob_half_sgemm(0,m,n,k,a,lda,b,ldb,c,ldc);
}
int apple_fp16_three(int m,int n,int k,const float *a,int lda,const float *b,int ldb,float *c,int ldc)
{
    return cob_half_sgemm(1,m,n,k,a,lda,b,ldb,c,ldc);
}
