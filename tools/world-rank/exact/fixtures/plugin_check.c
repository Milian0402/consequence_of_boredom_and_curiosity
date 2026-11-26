/* Untimed integration check for actual dispatch, matched fallback and OpenBLAS DGEMM DSOs. */
#define _GNU_SOURCE
#include "exact_fixture.h"
#include <dlfcn.h>
#include <math.h>
#include <openssl/evp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

struct view { float *parent,*x; int m,n,ld,off; size_t size; };
static void die(const char *s){fprintf(stderr,"FAIL %s\n",s);exit(1);}
static struct view view(int m,int n,int pad,int off)
{
    int ld=n+pad;size_t size=(off+(size_t)m*ld+9)*sizeof(float);
    float *p=mmap(NULL,size,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    if(p==MAP_FAILED)die("mmap");
    for(size_t q=0;q<size/sizeof(float);q++)p[q]=-54321.25f;
    struct view v={p,p+off,m,n,ld,off,size};return v;
}
static int logical(struct view v,size_t q){
    if(q<(size_t)v.off)return 0;
    size_t a=q-v.off;return a/v.ld<(size_t)v.m && a%v.ld<(size_t)v.n;
}
static void guards(struct view v){
    for(size_t q=0;q<v.size/sizeof(float);q++)
        if(!logical(v,q) && v.parent[q]!=-54321.25f)die("changed parent guard");
}
typedef void (*sgemm)(int,int,int,const float*,int,const float*,int,float*,int);
static void *sym(void *h,const char *n){void *x=dlsym(h,n);if(!x)die(n);return x;}

static void corpus_hash(struct view c,const double *ref,const char *want)
{
    unsigned char *row=malloc((size_t)c.n*4);
    if(!row)die("hash row");
    for(int source=0;source<2;source++){
        EVP_MD_CTX *ctx=EVP_MD_CTX_new();
        if(!ctx || !EVP_DigestInit_ex(ctx,EVP_sha256(),NULL))die("sha init");
        for(int i=0;i<c.m;i++){
            for(int j=0;j<c.n;j++){
                float v=source?(float)ref[(size_t)i*c.n+j]:c.x[(size_t)i*c.ld+j];uint32_t u;
                memcpy(&u,&v,4);
                for(int l=0;l<4;l++)row[4*j+l]=(unsigned char)(u>>(8*l));
            }
            if(!EVP_DigestUpdate(ctx,row,4*(size_t)c.n))die("sha update");
        }
        unsigned char digest[EVP_MAX_MD_SIZE];unsigned int count;
        if(!EVP_DigestFinal_ex(ctx,digest,&count) || count!=32)die("sha final");
        char text[65];for(unsigned int q=0;q<count;q++)sprintf(text+2*q,"%02x",digest[q]);
        if(strcmp(text,want))die("corpus FP32 known-answer SHA mismatch");
        EVP_MD_CTX_free(ctx);
    }
    free(row);
}

int main(int argc,char **argv)
{
    if(argc!=5)die("usage: plugin_check DISPATCH OPENBLAS DATA small|full");
    const char *fbpath=getenv("EXACT_FALLBACK_PLUGIN");
    const char *enable=getenv("EXACT_NATIVE_ENABLED");
    if(!fbpath || !enable)die("missing provider environment");
    int native=!strcmp(enable,"1");
    void *dispatch=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL),*fp=dlopen(fbpath,RTLD_NOW|RTLD_LOCAL);
    if(!dispatch||!fp)die(dlerror());
    if(((int(*)(void))sym(dispatch,"rank_vendor_init"))()!=0)die("dispatch init");
    sgemm mul=(sgemm)sym(dispatch,"rank_vendor_sgemm"),fallback=(sgemm)sym(fp,"rank_vendor_sgemm");
    int (*route)(void)=sym(dispatch,"exact_last_route"), (*products)(void)=sym(dispatch,"exact_last_products");
    int (*get_amax)(void)=sym(dispatch,"exact_last_amax"), (*get_bmax)(void)=sym(dispatch,"exact_last_bmax");
    unsigned long long (*scanned)(void)=sym(dispatch,"exact_last_inspected");
    const char *(*why)(void)=sym(dispatch,"exact_last_reason"),*(*impl)(void)=sym(dispatch,"exact_last_impl");
    struct row {const char *name,*fixture;int m,n,k,eligible;};
    struct row rows[]={
        {"signed","exact",7,13,31,1},{"early-A","early",7,13,31,0},{"late-B","late",7,13,31,0},
        {"257-last","exact",3,5,8,0},{"equal-bound","exact",3,5,256,1},{"over-bound","exact",3,5,257,0},
        {"positive-limit","exact",1,1,4096,1},{"depth-over","exact",2,3,4097,0},
        {"infinity-last","exact",3,5,7,0},{"nan-first","exact",3,5,7,0},{"signed-zero","exact",3,5,7,1},
        {"cube","exact",4096,4096,4096,1},{"heldout","exact",1079,1553,1201,1},
        {"heldout-early","early",1079,1553,1201,0},{"heldout-late","late",1079,1553,1201,0},
        {"digits","optdigits",1797,3823,64,1},{"images","optdigits-images",1797,1934,1024,1}
    };
    size_t end=!strcmp(argv[4],"full")?sizeof rows/sizeof *rows:11;
    puts("case,m,n,k,route,reason,inspected,amax,bmax,products,integer_spots,max_abs_error,implementation");
    for(size_t t=0;t<end;t++){
        int m=rows[t].m,n=rows[t].n,k=rows[t].k;
        struct view a=view(m,k,5,3),b=view(k,n,7,5),c=view(m,n,11,7),d=view(m,n,13,9);
        if(exact_fixture_fill(rows[t].fixture,argv[3],UINT64_C(0x84fd2117),m,n,k,a.x,a.ld,b.x,b.ld))die(exact_fixture_error());
        if(t==3)b.x[(size_t)(k-1)*b.ld+n-1]=257;
        if(t==4||t==5||t==6){
            for(int i=0;i<m;i++)for(int j=0;j<k;j++)a.x[(size_t)i*a.ld+j]=t==6?64.f:256.f;
            for(int i=0;i<k;i++)for(int j=0;j<n;j++)b.x[(size_t)i*b.ld+j]=t==6?64.f:256.f;
        }
        if(t==8)b.x[(size_t)(k-1)*b.ld+n-1]=INFINITY;
        if(t==9)a.x[0]=NAN;
        if(t==10)a.x[0]=-0.0f;
        guards(a);guards(b);
        if(mprotect(a.parent,a.size,PROT_READ)||mprotect(b.parent,b.size,PROT_READ))die("protect inputs");
        size_t cells=(size_t)m*n;
        int finite=t!=8 && t!=9;
        double *ref=NULL,*wt=NULL;
        if(finite){
            ref=malloc(cells*sizeof(double));wt=malloc(cells*sizeof(double));if(!ref||!wt)die("allocate references");
            if(exact_fixture_golden(argv[2],m,n,k,a.x,a.ld,b.x,b.ld,ref,wt))die(exact_fixture_error());
        }
        for(int i=0;i<m;i++)for(int j=0;j<n;j++){c.x[(size_t)i*c.ld+j]=NAN;d.x[(size_t)i*d.ld+j]=NAN;}
        mul(m,n,k,a.x,a.ld,b.x,b.ld,c.x,c.ld);
        int routed=route(),wanted=native?(rows[t].eligible?1:2):3,prod=products();
        unsigned long long count=scanned(), expected=native && k<=4096?(unsigned long long)m*k+(unsigned long long)k*n:0;
        int early_a=t==1 || t==9 || t==13;
        int bad_b=t==2 || t==3 || t==8 || t==14;
        int magnitude=t==4 || t==5?256:t==6?64:t==15?16:t==16?1:63;
        int amax=get_amax(), bmax=get_bmax();
        int expect_a=expected && !early_a?magnitude:-1;
        int expect_b=expect_a>=0 && !bad_b?magnitude:-1;
        if(expected && early_a)expected=k<64?k:64;
        if(routed!=wanted || prod!=(wanted==1) || count!=expected || amax!=expect_a || bmax!=expect_b){
            fprintf(stderr,"%s route=%d reason=%s products=%d scanned=%llu amax=%d bmax=%d\n",
                    rows[t].name,routed,why(),prod,count,amax,bmax);
            die("unexpected dispatch");
        }
        if(wanted!=1)fallback(m,n,k,a.x,a.ld,b.x,b.ld,d.x,d.ld);
        double maxerr=0;
        for(int i=0;i<m;i++)for(int j=0;j<n;j++){
            size_t ix=(size_t)i*n+j;float z=c.x[(size_t)i*c.ld+j];
            if(wanted!=1 && memcmp(&z,&d.x[(size_t)i*d.ld+j],sizeof z))die("FP32 fallback bits differ");
            if(finite){
                double err=fabs((double)z-ref[ix]);if(!isfinite(z)||err>.002 || err*(16777216.-k)>k*wt[ix])die("FP64 tolerance");
                if(wanted==1 && (double)z!=ref[ix])die("integer result differs");
                if(err>maxerr)maxerr=err;
            }
        }
        if(t==15)corpus_hash(c,ref,"ec47b13672085cc2096a990dd166c76e2e29843c7f79ea88d1d483db94b05db0");
        if(t==16)corpus_hash(c,ref,"eec353b47234bfe0fa3544331dc16aceee6a9b1ee3b7165b51e30fe565a9e825");
        guards(c);guards(d);
        printf("%s,%d,%d,%d,%d,%s,%llu,%d,%d,%d,%d,",rows[t].name,m,n,k,routed,why(),count,amax,bmax,prod,finite?exact_fixture_spots():0);
        if(finite)printf("%.17g",maxerr);
        printf(",%s\n",impl());
        fflush(stdout);free(ref);free(wt);munmap(a.parent,a.size);munmap(b.parent,b.size);munmap(c.parent,c.size);munmap(d.parent,d.size);
    }
}
