#include <sys/types.h>
#include <sys/sysctl.h>
#include <sys/utsname.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
int main(void) {
    struct utsname u;uname(&u);printf("os=%s %s %s\n",u.sysname,u.release,u.machine);
    const char *keys[]={"machdep.cpu.brand_string","hw.optional.arm.FEAT_SME","hw.optional.arm.FEAT_SME2","hw.optional.arm.FEAT_SME_F16F32","hw.optional.arm.SME_F16F32","hw.optional.arm.FEAT_SME2p1","hw.optional.arm.FEAT_FP16"};
    for(int i=0;i<7;i++){char buf[128]={0};size_t n=sizeof(buf);int rc=sysctlbyname(keys[i],buf,&n,0,0);if(rc==0){int v=0;memcpy(&v,buf,sizeof(v));if(i==0)printf("%s=%s\n",keys[i],buf);else printf("%s=%d\n",keys[i],v);}else printf("%s=%s\n",keys[i],strerror(errno));}
}
