/* Run only after numerical verification, with a separately observed plugin. */
#include "../handles/source/handle.h"
static void report_audit(const char *plugin_path,int warm_calls,struct view a,struct view b,struct view c)
{
    if (!strcmp(plugin_path,"public")) return;
    void *library=dlopen(plugin_path,RTLD_NOW|RTLD_NOLOAD);
    if (!library) fatal("could not examine already loaded plugin");
    void (*read_observation)(struct observation *)=NULL;
    void *entry=dlsym(library,"rank_handle_trace_read");
    memcpy(&read_observation,&entry,sizeof read_observation);
    if (read_observation) {
        struct observation o={0};
        read_observation(&o);
        const int retained=dlsym(library,"rank_handle_prepare")!=NULL;
        const int calls=1+warm_calls;
        const int agrees=o.calls==calls && o.creates==(retained?1:calls) &&
            o.gets==o.creates && o.destroys==(retained?0:calls) &&
            o.create_status>=0 && o.create_status<=1 && o.execution_status==o.create_status &&
            (retained||o.destroy_status==0) && o.handle!=0 && o.kernel!=0 &&
            o.layout==101 && o.transa==111 && o.transb==111 && o.alpha==1 && o.beta==0 &&
            o.m==a.rows && o.n==b.cols && o.k==a.cols &&
            o.lda==a.ld && o.ldb==b.ld && o.ldc==c.ld &&
            o.a==(uintptr_t)a.data && o.b==(uintptr_t)b.data && o.c==(uintptr_t)c.data;
        if (!agrees) fatal("explicit oneMKL handle lifecycle or caller mismatch");
        printf("HANDLE_PROOF,%d,%d,%d,%d,%d,%d\n",o.creates,o.gets,o.calls,o.destroys,o.execution_status,retained);
    }
    dlclose(library);
}
