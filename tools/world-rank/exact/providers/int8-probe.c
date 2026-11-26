#define _GNU_SOURCE
#include <dlfcn.h>
#include <dirent.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sched.h>
static int tasks(void){DIR*d=opendir("/proc/self/task");if(!d)return -1;int c=0;struct dirent*e;while((e=readdir(d)))if(e->d_name[0]!='.')++c;closedir(d);return c;}
int main(int argc,char**argv){
    if(argc!=5)return 2;int m=atoi(argv[2]),n=atoi(argv[3]),k=atoi(argv[4]);
    if(m<1 || n<1 || k<1 || k>4096)return 2;
    void*h=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);if(!h){fprintf(stderr,"%s\n",dlerror());return 2;}
    int(*init)(void)=dlsym(h,"rank_vendor_init");
    void(*product)(int,int,int,const float*,int,const float*,int,float*,int)=dlsym(h,"rank_vendor_sgemm");
    int(*terms)(void)=dlsym(h,"exact_int8_last_terms");const char*(*name)(void)=dlsym(h,"exact_int8_implementation");
    if(!init||!product||!terms||!name||init()||tasks()!=1)return 3;
    size_t ac=(size_t)m*k,bc=(size_t)k*n,cc=(size_t)m*n;
    float*a=malloc(ac*sizeof*a),*b=malloc(bc*sizeof*b),*c=malloc(cc*sizeof*c);if(!a||!b||!c)return 4;
    for(int i=0;i<m;++i)for(int p=0;p<k;++p)a[(size_t)i*k+p]=(float)((i%5-2)*(p%2?2:-2));
    for(int p=0;p<k;++p)for(int j=0;j<n;++j)b[(size_t)p*n+j]=(float)((j%7-3)*(p%2?1:-1));
    memset(c,0xff,cc*sizeof*c);product(m,n,k,a,k,b,n,c,n);
    for(int i=0;i<m;++i)for(int j=0;j<n;++j)if(c[(size_t)i*n+j]!=(float)(2*(i%5-2)*(j%7-3)*k))return 5;
    int t=tasks();printf("INT8_PROOF,%s,%d,%zu,%d,%d\n",name(),terms(),cc,t,sched_getcpu());
    free(a);free(b);free(c);return terms()!=1||t!=1;
}
