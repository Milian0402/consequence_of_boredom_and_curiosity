#define _GNU_SOURCE
#include <dirent.h>
#include <dlfcn.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef void (*sgemm_fn)(int,int,int,const float *,int,const float *,int,float *,int);
static int threads(void)
{
    DIR *d=opendir("/proc/self/task"); if (!d) return -1;
    struct dirent *e; int n=0;
    while ((e=readdir(d))) if (e->d_name[0]!='.') ++n;
    closedir(d); return n;
}
static void *sym(void *handle,const char *name)
{
    void *p=dlsym(handle,name); if (!p) {fprintf(stderr,"missing %s\n",name);exit(2);}return p;
}
static int bv(int p,int j){return (p*3+j*5)%17-8;}
static int av(int i,int term){return (i*5+term*2)%13-6;}
static void fill(float *ptr,size_t count,float v){for(size_t i=0;i<count;++i)ptr[i]=v;}
int main(int argc,char **argv)
{
    if (argc!=6) return 2;
    const int enable=atoi(argv[3]),trace=atoi(argv[4]);
    void *adapter=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);
    if (!adapter || ((int (*)(void))sym(adapter,"rank_vendor_init"))())return 2;
    sgemm_fn run=(sgemm_fn)sym(adapter,"rank_vendor_sgemm");
    void (*read_trace)(int *,const char **)=NULL; void (*observed)(int *)=NULL;
    if(trace){
        read_trace=(void (*)(int *,const char **))sym(adapter,"rank_host_trace");
        char filename[4096];snprintf(filename,sizeof filename,"%s/%s.so",getenv("RANK_HOST_BLAS_DIR"),argv[5]);
        void *p=dlopen(filename,RTLD_NOW|RTLD_NOLOAD|RTLD_LOCAL); if(!p)return 2;
        observed=(void (*)(int *))sym(p,"rank_observed_call");
    }
    FILE *file=fopen(argv[2],"r");char line[256];
    if(!file || !fgets(line,sizeof line,file))return 2;
    int count=0;
    while(fgets(line,sizeof line,file)){
        char label[80];int m,n,k,pad;
        if(sscanf(line,"%79[^,],%d,%d,%d,%d",label,&m,&n,&k,&pad)!=5)return 2;
        int lda=k+pad,ldb=n+pad,ldc=n+pad;
        size_t na=(size_t)m*lda+32, nb=(size_t)k*ldb+32,nc=(size_t)m*ldc+32;
        float *ab=malloc(na*4),*bb=malloc(nb*4),*cb=malloc(nc*4),*a0=malloc(na*4),*b0=malloc(nb*4);
        if(!ab||!bb||!cb||!a0||!b0)return 2;
        fill(ab,na,129);fill(bb,nb,129);fill(cb,nc,129);
        float *a=ab+16,*b=bb+16,*c=cb+16;
        const int terms[]={0,k/2,k-1};
        for(int i=0;i<m;++i){memset(a+(size_t)i*lda,0,k*4);for(int t=0;t<3;++t)a[(size_t)i*lda+terms[t]]=av(i,t);}
        for(int p=0;p<k;++p)for(int j=0;j<n;++j)b[(size_t)p*ldb+j]=bv(p,j);
        memcpy(a0,ab,na*4);memcpy(b0,bb,nb*4);
        int cpu=sched_getcpu();if(threads()!=1)return 3;
        run(m,n,k,a,lda,b,ldb,c,ldc);
        if(threads()!=1||cpu!=sched_getcpu()||memcmp(ab,a0,na*4)||memcmp(bb,b0,nb*4))return 3;
        for(int j=0;j<16;++j)if(cb[j]!=129||cb[nc-16+j]!=129)return 4;
        for(int i=0;i<m;++i)for(int j=0;j<ldc;++j){
            int v=129;if(j<n){v=0;for(int t=0;t<3;++t)v+=av(i,t)*bv(terms[t],j);}
            if(c[(size_t)i*ldc+j]!=v)return 4;
        }
        int newm=(m+7)&~7;
        int active=enable&&m>=24&&m<=96&&n>=1024&&n<=4096&&k>=96&&k<=384&&n>=8*m&&newm>m&&8*(newm-m)<=m;
        int expected[]={active,active?newm:m,n,k,active?(k+15)&~15:lda,ldb,active?(n+15)&~15:ldc,1};
        ++count;
        if(trace){
            int got[8],call[7];const char *name;
            read_trace(got,&name);observed(call);
            if(strcmp(name,argv[5])||memcmp(got,expected,sizeof got)||call[0]!=count||memcmp(call+1,expected+1,6*sizeof(int)))return 5;
        }
        printf("%s,%s,%d,%d,%d,%d,%s,%d,%d,%d,%d,%d,%d,%d,%d,1\n",enable?"host-pad8":"host-direct",label,m,n,k,pad,argv[5],trace,active,expected[1],expected[4],ldb,expected[6],m*n,cpu);
        free(ab);free(bb);free(cb);free(a0);free(b0);
    }
    fclose(file);return count>0 ? 0:2;
}
