#include <libproc.h>
#include <sys/proc_info.h>
#include <unistd.h>
int apple_threads(int pid) {
    struct proc_taskinfo t;
    int n=proc_pidinfo(pid,PROC_PIDTASKINFO,0,&t,sizeof t);
    return n==sizeof t?t.pti_threadnum:-1;
}
