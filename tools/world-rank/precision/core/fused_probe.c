#define _GNU_SOURCE
#include <dlfcn.h>
#include <math.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <dirent.h>
static int tasks(void){DIR *d=opendir("/proc/self/task"); if(!d)return -1; int n=0;for(struct dirent *e;(e=readdir(d));)n+=e->d_name[0]!='.';closedir(d);return n;}
static float *parent(int rows,int ld,int ro,int co,float **allocation,size_t *cells){
    *cells=(size_t)(rows+ro+2)*ld+32;*allocation=malloc(*cells*sizeof(float));if(!*allocation)exit(4);
    for(size_t i=0;i<*cells;++i)(*allocation)[i]=-1999.0f;
    return *allocation+(size_t)ro*ld+co;
}
int main(int argc,char **argv){
    if(argc!=14)return 2;
    void *s=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);if(!s){fprintf(stderr,"%s\n",dlerror());return 3;}
    int (*init)(void)=dlsym(s,"amx_fused6_init");
    int (*run)(int,int,int,const float*,int,const float*,int,float*,int)=dlsym(s,"amx_fused6_sgemm");
    uint64_t (*count)(int)=dlsym(s,"cob_amx_six_trace");
    if(!init||!run||!count)return 3;
    int status=init();if(status){printf("{\"eligible\":false,\"reason\":\"AMX CPU or OS unavailable\",\"init_status\":%d}\n",status);return 0;}
    int m=atoi(argv[2]),n=atoi(argv[3]),k=atoi(argv[4]),la=atoi(argv[5]),lb=atoi(argv[6]),lc=atoi(argv[7]);
    int ar=atoi(argv[8]),ac=atoi(argv[9]),br=atoi(argv[10]),bc=atoi(argv[11]),cr=atoi(argv[12]),cc=atoi(argv[13]);
    float *aa,*bb,*oo;size_t na,nb,nc;float *a=parent(m,la,ar,ac,&aa,&na),*b=parent(k,lb,br,bc,&bb,&nb),*c=parent(m,lc,cr,cc,&oo,&nc);
    for(int i=0;i<m;++i)for(int p=0;p<k;++p)a[(size_t)i*la+p]=(float)((i%5)-2)*(p%2?-1:1);
    for(int p=0;p<k;++p)for(int j=0;j<n;++j)b[(size_t)p*lb+j]=(float)((j%7)-3)*(p%2?-1:1);
    uint64_t pre[3];for(int t=0;t<3;++t)pre[t]=count(t);
    int rc=run(m,n,k,a,la,b,lb,c,lc);for(int t=0;t<3;++t)pre[t]=count(t)-pre[t];
    size_t wrong=0;for(size_t q=0;q<nc;++q){size_t off=(size_t)cr*lc+cc;float expected=-1999.0f;
        if(q>=off && (q-off)/lc<(size_t)m && (q-off)%lc<(size_t)n){size_t i=(q-off)/lc,j=(q-off)%lc;expected=(float)(((int)(i%5)-2)*((int)(j%7)-3)*k);}
        wrong+=oo[q]!=expected;
    }
    size_t damaged=0;
    for(size_t q=0;q<na;++q){size_t off=(size_t)ar*la+ac;float want=-1999.f;if(q>=off && (q-off)/la<(size_t)m && (q-off)%la<(size_t)k)want=(float)((int)(((q-off)/la)%5)-2)*(((q-off)%la)%2?-1:1);damaged+=aa[q]!=want;}
    for(size_t q=0;q<nb;++q){size_t off=(size_t)br*lb+bc;float want=-1999.f;if(q>=off && (q-off)/lb<(size_t)k && (q-off)%lb<(size_t)n)want=(float)((int)(((q-off)%lb)%7)-3)*(((q-off)/lb)%2?-1:1);damaged+=bb[q]!=want;}
    uint64_t tiles=(uint64_t)((m+15)/16)*((n+15)/16);uint64_t blocks=tiles*((k+31)/32);
    int nt=tasks();int ok=rc==0 && !wrong && !damaged && nt==1 && pre[0]==tiles && pre[1]==blocks && pre[2]==6*blocks;
    printf("{\"eligible\":%s,\"reason\":\"%s\",\"route\":\"amx-tdpbf16ps-fused6\",\"tiles\":%llu,\"blocks\":%llu,\"dot_instructions\":%llu,\"output_cells\":%zu,\"outside_C_cells\":%zu,\"mismatches\":%zu,\"input_or_padding_changes\":%zu,\"cpu\":%d,\"threads\":%d}\n",ok?"true":"false",ok?"confirmed actual AMX dot instructions":"route or output mismatch",(unsigned long long)pre[0],(unsigned long long)pre[1],(unsigned long long)pre[2],(size_t)m*n,nc-(size_t)m*n,wrong,damaged,sched_getcpu(),nt);
    free(aa);free(bb);free(oo);dlclose(s);return ok?0:1;
}
