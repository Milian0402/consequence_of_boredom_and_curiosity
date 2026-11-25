#define _GNU_SOURCE
#include <dlfcn.h>
#include <limits.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/mman.h>

typedef void (*sgemm_fn)(int,int,int,const float*,int,const float*,int,float*,int);
static sgemm_fn sgemm;
static size_t actual_stack;

static void *small_product(void *unused)
{
    (void)unused;
#if defined(__APPLE__)
    actual_stack = pthread_get_stacksize_np(pthread_self());
#elif defined(__linux__)
    pthread_attr_t attr;
    if (pthread_getattr_np(pthread_self(), &attr)) return (void*)1;
    if (pthread_attr_getstacksize(&attr, &actual_stack)) return (void*)1;
    pthread_attr_destroy(&attr);
#endif
    float a[4]={1,2,3,4}, b[4]={5,6,7,8}, c[6]={0,0,93,0,0,94};
    for (int t=0;t<32;++t) {
        sgemm(2,2,2,a,2,b,2,c,3);
        if(c[0]!=19 || c[1]!=22 || c[2]!=93 || c[3]!=43 || c[4]!=50 || c[5]!=94)
            return (void*)1;
    }
    return NULL;
}

int main(int argc, char **argv)
{
    if (argc!=2) return 2;
    void *lib=dlopen(argv[1],RTLD_NOW | RTLD_LOCAL);
    if (!lib || !(sgemm=(sgemm_fn)dlsym(lib,"cob_sgemm_rowmajor"))) return 3;
    /* Resolve lazy external symbols before constraining the calling thread. */
    float a=1,b=2,c=0;
    sgemm(1,1,1,&a,1,&b,1,&c,1);
    if (c!=2) return 4;
    const size_t page=(size_t)sysconf(_SC_PAGESIZE);
    const size_t stack=((size_t)PTHREAD_STACK_MIN + page-1)/page*page;
    unsigned char* region=mmap(NULL, stack+2*page, PROT_NONE,
                               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (region==MAP_FAILED || mprotect(region+page,stack,PROT_READ | PROT_WRITE)) return 5;
    pthread_attr_t attr;
    pthread_t thread;
    if (pthread_attr_init(&attr) || pthread_attr_setstack(&attr,region+page,stack) ||
        pthread_create(&thread,&attr,small_product,NULL)) return 5;
    pthread_attr_destroy(&attr);
    void *result;
    if (pthread_join(thread,&result) || result || actual_stack!=stack) {
        fprintf(stderr, "worker=%p requested=%zu observed=%zu\n",result,stack,actual_stack);
        return 6;
    }
    printf("requested_stack_bytes,observed_stack_bytes,public_calls,outputs_checked,padding_checked\n");
    printf("%zu,%zu,32,128,64\n",stack,actual_stack);
    munmap(region,stack+2*page);
    dlclose(lib);
    return 0;
}
