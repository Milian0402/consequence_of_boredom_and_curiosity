#include "exact_fixture.h"

#include <dlfcn.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static _Thread_local const char *message;
static _Thread_local int last_spots;

const char *exact_fixture_error(void) { return message ? message : ""; }
int exact_fixture_spots(void) { return last_spots; }
static int fail(const char *reason) { message=reason; return 1; }

static uint64_t next_word(uint64_t *state)
{
    uint64_t x=(*state+=UINT64_C(0x9e3779b97f4a7c15));
    x=(x^(x>>30))*UINT64_C(0xbf58476d1ce4e5b9);
    x=(x^(x>>27))*UINT64_C(0x94d049bb133111eb);
    return x^(x>>31);
}

/* Positive dimensions only. Make pointer index and byte-count arithmetic representable. */
static int valid_extent(int rows,int cols,int ld,size_t size)
{
    if(rows<1 || cols<1 || ld<cols)return 0;
    return (size_t)rows<=PTRDIFF_MAX/size/(size_t)ld;
}

static void synthetic(float *v,int rows,int cols,int ld,uint64_t state,int first)
{
    for(int i=0;i<rows;i++)
        for(int j=0;j<cols;j++)
            v[(size_t)i*ld+j]=(float)((int)(next_word(&state)%127)-63);
    v[0]=(float)first;
    if(rows>1 || cols>1) v[(size_t)(rows-1)*ld+cols-1]=(float)-first;
}

static int load_bytes(const char *dir,const char *name,int rows,int cols,int largest,float *v,int ld)
{
    if(!dir)return fail("missing corpus directory");
    size_t len=strlen(dir)+strlen(name)+2;
    char *path=malloc(len);
    unsigned char *row=malloc((size_t)cols);
    if(!path || !row){free(path);free(row);return fail("corpus allocation");}
    snprintf(path,len,"%s/%s",dir,name);
    FILE *f=fopen(path,"rb");
    free(path);
    if(!f){free(row);return fail("could not open corpus file");}
    int rc=0;
    for(int i=0;i<rows && !rc;i++){
        if(fread(row,1,(size_t)cols,f)!=(size_t)cols){rc=fail("short corpus file");break;}
        for(int j=0;j<cols;j++){
            if(row[j]>largest){rc=fail("corpus value exceeds allowed maximum");break;}
            v[(size_t)i*ld+j]=(float)row[j];
        }
    }
    if(!rc && (fgetc(f)!=EOF || ferror(f)))rc=fail("corpus read or length mismatch");
    fclose(f);free(row);return rc;
}

int exact_fixture_fill(const char *spec,const char *corpus_dir,uint64_t seed,
                       int m,int n,int k,float *a,int lda,float *b,int ldb)
{
    message=NULL;
    if(!spec || !a || !b || !valid_extent(m,k,lda,sizeof *a) ||
       !valid_extent(k,n,ldb,sizeof *b)) return fail("invalid fixture layout");
    if(strcmp(spec,"optdigits")==0){
        if(m!=1797 || n!=3823 || k!=64)return fail("optdigits requires 1797x3823x64");
        if(load_bytes(corpus_dir,"digits_A.u8",m,k,16,a,lda))return 1;
        return load_bytes(corpus_dir,"digits_B.u8",k,n,16,b,ldb);
    }
    if(strcmp(spec,"optdigits-images")==0){
        if(m!=1797 || n!=1934 || k!=1024)return fail("optdigits-images requires 1797x1934x1024");
        if(load_bytes(corpus_dir,"image_A.u8",m,k,1,a,lda))return 1;
        return load_bytes(corpus_dir,"image_B.u8",k,n,1,b,ldb);
    }
    int early=!strcmp(spec,"early"), late=!strcmp(spec,"late");
    if(!early && !late && strcmp(spec,"exact"))return fail("unknown fixture");
    synthetic(a,m,k,lda,seed^UINT64_C(0x243f6a8885a308d3),-63);
    synthetic(b,k,n,ldb,seed^UINT64_C(0x13198a2e03707344),63);
    if(early)a[0]+=0.5f;
    if(late)b[(size_t)(k-1)*ldb+n-1]+=0.5f;
    return 0;
}

static int copy_dyadics(const float *src,int rows,int cols,int ld,double *dest)
{
    for(int i=0;i<rows;i++)for(int j=0;j<cols;j++){
        const double x=src[(size_t)i*ld+j];
        if(!isfinite(x) || fabs(x)>512. || 2*x!=(double)(int)(2*x))
            return fail("oracle only supports finite half-integers of magnitude at most 512");
        dest[(size_t)i*cols+j]=x;
    }
    return 0;
}

static int check_spot(int i,int j,int n,int k,const float *a,int lda,const float *b,int ldb,
                      const double *ref,const double *weight)
{
    int64_t numerator=0,magnitude=0;
    for(int p=0;p<k;p++){
        int64_t x=(int64_t)(2.*a[(size_t)i*lda+p]);
        int64_t y=(int64_t)(2.*b[(size_t)p*ldb+j]);
        int64_t xy=x*y;
        numerator+=xy;
        magnitude+=xy<0 ? -xy:xy;
    }
    size_t q=(size_t)i*n+j;
    return ref[q]==(double)numerator*.25 && weight[q]==(double)magnitude*.25;
}

int exact_fixture_golden(const char *plugin,int m,int n,int k,const float *a,int lda,
                         const float *b,int ldb,double *ref,double *weight)
{
    message=NULL;last_spots=0;
    if(!plugin || !a || !b || !ref || !weight || ref==weight ||
       !valid_extent(m,k,lda,sizeof *a) || !valid_extent(k,n,ldb,sizeof *b) ||
       !valid_extent(m,k,k,sizeof(double)) || !valid_extent(k,n,n,sizeof(double)) ||
       !valid_extent(m,n,n,sizeof(double))) return fail("invalid oracle layout");
    void *lib=dlopen(plugin,RTLD_NOW|RTLD_LOCAL);
    if(!lib)return fail("could not load DGEMM plugin");
    int (*init)(void)=NULL;
    void (*gemm)(int,int,int,const double*,int,const double*,int,double*,int)=NULL;
    void *fn=dlsym(lib,"rank_vendor_init");memcpy(&init,&fn,sizeof init);
    fn=dlsym(lib,"rank_vendor_dgemm");memcpy(&gemm,&fn,sizeof gemm);
    if(!init || !gemm || init()!=0){dlclose(lib);return fail("OpenBLAS DGEMM adapter initialization");}
    size_t na=(size_t)m*k,nb=(size_t)k*n;
    double *ad=malloc(na*sizeof *ad),*bd=malloc(nb*sizeof *bd);
    int rc=0;
    if(!ad || !bd){rc=fail("DGEMM oracle allocation");goto done;}
    if(copy_dyadics(a,m,k,lda,ad) || copy_dyadics(b,k,n,ldb,bd)){rc=1;goto done;}
    gemm(m,n,k,ad,k,bd,n,ref,n);
    for(size_t q=0;q<na;q++)ad[q]=fabs(ad[q]);
    for(size_t q=0;q<nb;q++)bd[q]=fabs(bd[q]);
    gemm(m,n,k,ad,k,bd,n,weight,n);
    /* Each input times 2 is an integer <= 1024. INT_MAX*1024^2 < 2^51.
       Every product, partial signed sum, and magnitude sum is exact in both FP64 and int64. */
    size_t seen[32],cells=(size_t)m*n;
    int target=cells<32?(int)cells:32;
    uint64_t state=UINT64_C(0xf4ac92e7b01dc365);
    for(size_t t=0;last_spots<target;t++){
        size_t q;
        if(cells<=32)q=t;
        else if(t<4){
            size_t corner[4]={0,(size_t)n-1,(size_t)(m-1)*n,cells-1};q=corner[t];
        }else q=next_word(&state)%cells;
        int duplicate=0;
        for(int x=0;x<last_spots;x++)if(seen[x]==q){duplicate=1;break;}
        if(duplicate)continue;
        if(!check_spot((int)(q/n),(int)(q%n),n,k,a,lda,b,ldb,ref,weight)){
            rc=fail("FP64 DGEMM disagrees with independent integer numerator");goto done;
        }
        seen[last_spots++]=q;
    }
done:
    free(ad);free(bd);dlclose(lib);return rc;
}
