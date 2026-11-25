#include <math.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/sysctl.h>

int ring32(unsigned chips, float *out);
int ring16(unsigned chips, float *out);
enum { max_lanes = 64, fmopa_per_chip = 8192 };
struct result { double seconds; int lanes; };

static void unsupported(int sig)
{
    (void)sig;
    static const char text[]="Unusable streaming F16/F32 FMOPA (SIGILL).\n";
    (void)write(STDERR_FILENO,text,sizeof(text)-1);
    _exit(77);
}

static struct result run(int h,unsigned chips)
{
    float out[4*max_lanes*max_lanes+13];
    for (size_t i=0;i<sizeof(out)/sizeof(out[0]);++i) out[i]=-1337.f;
    struct timespec a,b;
    clock_gettime(CLOCK_MONOTONIC_RAW,&a);
    int lanes=h?ring16(chips,out):ring32(chips,out);
    clock_gettime(CLOCK_MONOTONIC_RAW,&b);
    if (lanes<1 || lanes>max_lanes) exit(3);
    float fac=h?2.f:1.f;
    const float expected[4]={2048.f*fac,4096.f*fac,1024.f*fac,-2048.f*fac};
    for (int r=0;r<lanes;++r) for (int t=0;t<4;++t) for (int c=0;c<lanes;++c) {
        size_t index=((size_t)r*4+t)*lanes+c;
        if (out[index]!=expected[t]) {
            fprintf(stderr,"tile %d cell %d,%d got %.9g expected %.9g\n",t,r,c,out[index],expected[t]);
            exit(4);
        }
    }
    for (size_t i=4u*lanes*lanes;i<sizeof(out)/sizeof(out[0]);++i) if (out[i]!=-1337.f)exit(5);
    return (struct result){(double)(b.tv_sec-a.tv_sec)+(double)(b.tv_nsec-a.tv_nsec)*1e-9,lanes};
}

int main(int argc,char **argv)
{
    if (argc!=2 || (strcmp(argv[1],"check") && strcmp(argv[1],"measure")))return 2;
    struct sigaction sa={0};
    sa.sa_handler=unsupported;
    sigaction(SIGILL,&sa,NULL);
    int qos=pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED,0);
    char cpu[128]={0};size_t n=sizeof(cpu);
    if (sysctlbyname("machdep.cpu.brand_string",cpu,&n,NULL,0)!=0)strcpy(cpu,"unavailable");
    struct result a=run(0,1), b=run(1,1);
    if(a.lanes!=b.lanes)return 3;
    fprintf(stderr,"cpu=%s lanes=%d svl_bytes=%d cells_checked_per_call=%d qos_result=%d cpu_affinity=unavailable\n",
            cpu,a.lanes,4*a.lanes,4*a.lanes*a.lanes,qos);
    if(!strcmp(argv[1],"check"))return 0;
    /* A common chip count keeps the number of retired FMOPA instructions equal. */
    unsigned chips=64;
    do {
        a=run(0,chips); b=run(1,chips);
        if (fmax(a.seconds,b.seconds)>=.003 || chips>=65536)break;
        chips*=2;
    } while(1);
    unsigned estimate=(unsigned)fmax(1.,fmin(1000000.,round(chips*.09/fmax(a.seconds,b.seconds))));
    printf("round,position,mode,chips,lanes,fmopa,seconds,gflops\n");
    for(int i=0;i<6;i++)for(int j=0;j<2;j++) {
        int half=(i+j)&1;
        struct result s=run(half,estimate);
        uint64_t calls=(uint64_t)estimate*fmopa_per_chip;
        double work=(double)calls*s.lanes*s.lanes*(half?4.:2.);
        printf("%d,%d,%s,%u,%d,%llu,%.9f,%.5f\n",i,j,half?"fp16":"fp32",estimate,s.lanes,
               (unsigned long long)calls,s.seconds,work/(1e9*s.seconds));
    }
}
