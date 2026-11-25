#define _GNU_SOURCE
#include "../backend/backend.h"
#include <arm_neon.h>
#include <asm/hwcap.h>
#include <dlfcn.h>
#include <limits.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/auxv.h>
#include <sys/stat.h>

#define ACL_SYMBOL "_ZN8arm_gemm34a64_interleaved_bf16fp32_mmla_8x12EPKN11arm_compute8bfloat16ES3_Pfiii"
#define ACL_KERNEL "a64_interleaved_bf16fp32_mmla_8x12"
#define KERNEL_ARGS const uint16_t *a, const uint16_t *b, float *c, int ab, int bb, int kp
extern void acl_kernel(KERNEL_ARGS) __asm__(ACL_SYMBOL);
enum { MR=8, NR=12, KR=4, PANEL=8 };
struct plan {
    int m,n,k,lda,ldb,ldc,ab,bb,kp;
    size_t abytes,bbytes,cbytes;
    uint16_t *ap,*bp;
    float *cp;
    uint64_t execs;
};
static _Thread_local struct {
    int code;
    const char *error;
    uint64_t created,executed,destroyed;
} history;

static int status(int code,const char *message)
{
    history.code=code; history.error=message; return code;
}
#ifdef ACL_BF16_TRACE
static _Thread_local struct plan *active;
static _Thread_local uint64_t hits;
void acl_real(KERNEL_ARGS) __asm__("__real_" ACL_SYMBOL);
void acl_wrap(KERNEL_ARGS) __asm__("__wrap_" ACL_SYMBOL);
void acl_wrap(KERNEL_ARGS)
{
    struct plan *p=active;
    if (!p || kp!=p->kp || ab<1 || ab>PANEL || bb<1 || bb>PANEL || c!=p->cp) abort();
    ptrdiff_t ao=a-p->ap, bo=b-p->bp;
    if(ao<0 || bo<0 || (size_t)ao%((size_t)MR*kp) || (size_t)bo%((size_t)NR*kp) ||
       ((size_t)ao+(size_t)ab*MR*kp)*2>p->abytes || ((size_t)bo+(size_t)bb*NR*kp)*2>p->bbytes) abort();
    ++hits;
    acl_real(a,b,c,ab,bb,kp);
}
#endif

static int native_available(void)
{
    static _Thread_local int available;
    if (available) return 1;
    if (!(getauxval(AT_HWCAP)&HWCAP_ASIMD) || !(getauxval(AT_HWCAP2)&HWCAP2_BF16)) return 0;
    Dl_info owner;
    struct stat expected, actual;
#ifdef ACL_BF16_TRACE
    const void *target=(const void *)acl_real;
#else
    const void *target=(const void *)acl_kernel;
#endif
    if(!dladdr(target,&owner) || !owner.dli_fname || stat(owner.dli_fname,&actual) ||
       stat(ACL_PINNED_PATH,&expected) || actual.st_dev!=expected.st_dev || actual.st_ino!=expected.st_ino) return 0;
    available=1;
    return 1;
}

void *bf16_plan_create_strided(int m,int n,int k,int lda,int ldb,int ldc)
{
    if(m<1 || n<1 || k<1 || k>INT_MAX-3 || lda<k || ldb<n || ldc<n) {
        status(2,"invalid dimensions or strides");return NULL;
    }
    if (!native_available()) { status(3,"requires AArch64 ASIMD BF16 and the pinned ACL DSO");return NULL; }
    size_t ab=((size_t)m+MR-1)/MR, bb=((size_t)n+NR-1)/NR, kp=((size_t)k+KR-1)/KR*KR;
    size_t ma=ab<PANEL?ab:PANEL, nb=bb<PANEL?bb:PANEL;
    if(ab>SIZE_MAX/(2*MR)/kp || bb>SIZE_MAX/(2*NR)/kp) { status(1,"packed sizes overflow");return NULL; }
    size_t abytes=ab*MR*kp*2, bbytes=bb*NR*kp*2, cbytes=ma*nb*MR*NR*4;
    if(bbytes>PTRDIFF_MAX-cbytes || abytes>(size_t)PTRDIFF_MAX-cbytes-bbytes) {
        status(1,"workspace exceeds addressable storage");return NULL;
    }
    struct plan *p=calloc(1,sizeof *p);
    void *workspace=NULL;
    if(!p || posix_memalign(&workspace,64,abytes+bbytes+cbytes)) {
        free(p);status(1,"workspace allocation failed");return NULL;
    }
    *p=(struct plan){.m=m,.n=n,.k=k,.lda=lda,.ldb=ldb,.ldc=ldc,.ab=(int)ab,.bb=(int)bb,.kp=(int)kp,
                     .abytes=abytes,.bbytes=bbytes,.cbytes=cbytes,.ap=workspace,
                     .bp=(uint16_t *)((char *)workspace+abytes),.cp=(float *)((char *)workspace+abytes+bbytes)};
    ++history.created; status(0,"");return p;
}
void *bf16_plan_create(int m,int n,int k) { return bf16_plan_create_strided(m,n,k,k,n,n); }

static void pack_a(struct plan *p,const uint16_t *a)
{
    for(int tile=0;tile<p->ab;++tile) {
        const int first=tile*MR, nr=p->m-first<MR?p->m-first:MR;
        uint16_t *dest=p->ap+(size_t)first*p->kp;
        for(int k=0;k<p->kp;k+=KR) {
            int nk=p->k-k<KR?p->k-k:KR;
            for(int row=0;row<nr;++row) {
                const uint16_t *src=a+(size_t)(first+row)*p->lda+k;
                if(nk==KR) memcpy(dest+row*KR,src,KR*sizeof *src);
                else { memset(dest+row*KR,0,KR*sizeof *src);memcpy(dest+row*KR,src,(size_t)nk*sizeof *src); }
            }
            if(nr<MR) memset(dest+nr*KR,0,(size_t)(MR-nr)*KR*sizeof *dest);
            dest+=MR*KR;
        }
    }
}
static void pack_b(struct plan *p,const uint16_t *b)
{
    for(int tile=0;tile<p->bb;++tile) {
        const int col=tile*NR, nc=p->n-col<NR?p->n-col:NR;
        uint16_t *dest=p->bp+(size_t)col*p->kp;
        for(int k=0;k<p->kp;k+=KR) {
            int nk=p->k-k<KR?p->k-k:KR;
            const uint16_t *s=b+(size_t)k*p->ldb+col;
            if(nk==KR && nc==NR) {
                const uint16_t *s1=s+p->ldb,*s2=s1+p->ldb,*s3=s2+p->ldb;
                uint16x8x4_t lo={{vld1q_u16(s),vld1q_u16(s1),vld1q_u16(s2),vld1q_u16(s3)}};
                uint16x4x4_t hi={{vld1_u16(s+8),vld1_u16(s1+8),vld1_u16(s2+8),vld1_u16(s3+8)}};
                vst4q_u16(dest,lo); vst4_u16(dest+8*KR,hi);
            } else {
                memset(dest,0,NR*KR*sizeof *dest);
                for(int lane=0;lane<nk;++lane)
                    for(int j=0;j<nc;++j) dest[j*KR+lane]=s[(size_t)lane*p->ldb+j];
            }
            dest+=NR*KR;
        }
    }
}
int bf16_plan_exec(void *opaque,const uint16_t *a,const uint16_t *b,float *c)
{
    struct plan *p=opaque;
    if(!p || !a || !b || !c) return status(2,"null plan or operand");
    pack_a(p,a);pack_b(p,b);
#ifdef ACL_BF16_TRACE
    active=p;hits=0;
#endif
    for(int at=0;at<p->ab;at+=PANEL) {
        int ar=p->ab-at<PANEL?p->ab-at:PANEL;
        for(int bt=0;bt<p->bb;bt+=PANEL) {
            int bc=p->bb-bt<PANEL?p->bb-bt:PANEL;
            acl_kernel(p->ap+(size_t)at*MR*p->kp,p->bp+(size_t)bt*NR*p->kp,p->cp,ar,bc,p->kp);
            for(int i=0;i<ar;++i) for(int j=0;j<bc;++j) {
                int row=(at+i)*MR,col=(bt+j)*NR;
                int nr=p->m-row<MR?p->m-row:MR, nc=p->n-col<NR?p->n-col:NR;
                const float *block=p->cp+((size_t)i*bc+j)*MR*NR;
                for(int y=0;y<nr;++y)
                    memcpy(c+(size_t)(row+y)*p->ldc+col,block+y*NR,(size_t)nc*sizeof *c);
            }
        }
    }
#ifdef ACL_BF16_TRACE
    active=NULL;
    const uint64_t expected=(uint64_t)((p->ab+PANEL-1)/PANEL)*((p->bb+PANEL-1)/PANEL);
    if(hits!=expected) abort();
    fprintf(stderr,"ACL_BF16_KERNEL,name=" ACL_KERNEL ",calls=%llu,abytes=%zu,bbytes=%zu,cbytes=%zu,allocations=2,m=%d,n=%d,k=%d,ablocks=%d,bblocks=%d,kp=%d,planbytes=%zu\n",
            (unsigned long long)hits,p->abytes,p->bbytes,p->cbytes,p->m,p->n,p->k,p->ab,p->bb,p->kp,sizeof *p);
#endif
    ++p->execs;++history.executed;return status(0,"");
}
void bf16_plan_destroy(void *opaque)
{
    struct plan *p=opaque;
    if(p){free(p->ap);free(p);++history.destroyed;}
}
int bf16_plan_native(const void *p) { return p!=NULL; }
const char *bf16_plan_implementation(const void *p) { return p ? ACL_KERNEL : ""; }
const char *bf16_plan_route(const void *p) { return p ? "arm-acl-neon-bfmmla" : "no-descriptor"; }
uint64_t bf16_plan_executions(const void *p) { return p?((const struct plan *)p)->execs:0; }
int bf16_plan_status(void) { return history.code; }
const char *bf16_plan_error(void) { return history.error?history.error:""; }
void bf16_plan_counts(uint64_t *a,uint64_t *b,uint64_t *c)
{
    if(a)*a=history.created;
    if(b)*b=history.executed;
    if(c)*c=history.destroyed;
}
void bf16_plan_reset_counts(void) { history.created=history.executed=history.destroyed=0; }
