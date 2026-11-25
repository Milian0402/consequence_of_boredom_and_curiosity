/* Linux-only process and kernel identity checks for the Arm SGEMM panel.
   No code from this header runs between either pair of timer readings. */
#include <linux/filter.h>
#include <linux/seccomp.h>
#include <stddef.h>
#include <sys/prctl.h>
#include <sys/syscall.h>

/* Vendors are initialized after the filter is active. A helper thread or
   subprocess would terminate the worker even if it exited before a task count. */
__attribute__((constructor)) static void enforce_one_task(void)
{
    struct sock_filter rules[] = {
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, nr)),
#define BLOCK_NR(n) BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, n, 0, 1), BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS)
        BLOCK_NR(__NR_clone),
#ifdef __NR_clone3
        BLOCK_NR(__NR_clone3),
#endif
#ifdef __NR_fork
        BLOCK_NR(__NR_fork),
#endif
#ifdef __NR_vfork
        BLOCK_NR(__NR_vfork),
#endif
#undef BLOCK_NR
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
    };
    struct sock_fprog program = {sizeof(rules) / sizeof(rules[0]), rules};
    if (prctl(PR_SET_NO_NEW_PRIVS, 1L, 0L, 0L, 0L) ||
        prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &program))
        fatal("cannot enforce one task");
    if (getenv("ARM_ASSERT_NO_CLONE")) {
        /* The native smoke expects SIGSYS here; exit 2 in both processes if
           fork ever returns, so a broken filter cannot pass this check. */
        (void)fork();
        _exit(2);
    }
}

static void report_audit(const char *name, int calls, struct view a, struct view b,
                         struct view c)
{
    uint64_t hash = UINT64_C(14695981039346656037);
    const char *implementation = "direct";
    unsigned long plans = 0;
    size_t code_bytes = 0;
    for (int i = 0; i < c.rows; i++)
        for (int j = 0; j < c.cols; j++) {
            hash ^= get_bits(c.data + (size_t)i*c.ld+j);
            hash *= UINT64_C(1099511628211);
        }
    const char *dump = getenv("ARM_CONTROL_OUTPUT");
    if (dump) {
        FILE *file = fopen(dump,"wb");
        if (!file) fatal("control output open");
        for (int i=0;i<c.rows;i++)
            if (fwrite(c.data+(size_t)i*c.ld,sizeof(float),(size_t)c.cols,file)!=(size_t)c.cols)
                fatal("control output write");
        if (fclose(file)) fatal("control output close");
    }
    if (strcmp(name, "public")) {
        void *lib = dlopen(name, RTLD_NOLOAD | RTLD_NOW);
        if (!lib) fatal("active vendor unavailable");
        if (strstr(name, "libxsmm-")) {
            size_t (*inspect)(int,int,int,int,int,int);
            void *address = symbol(lib, "rank_xsmm_code_size");
            memcpy(&inspect,&address,sizeof inspect);
            code_bytes = inspect(c.rows,c.cols,a.cols,a.ld,b.ld,c.ld);
            if (!code_bytes) fatal("missing verified FP32 JIT");
            implementation = "libxsmm-fp32-jit";
        } else if (strstr(name, "onednn-f32-")) {
            const char *(*which)(void);
            unsigned long (*count)(void);
            int (*retained)(void);
            void *addr = symbol(lib, "rank_onednn_implementation");
            memcpy(&which,&addr,sizeof which);
            addr = symbol(lib, "rank_onednn_plans");
            memcpy(&count,&addr,sizeof count);
            addr = symbol(lib, "rank_onednn_cached_plan");
            memcpy(&retained,&addr,sizeof retained);
            plans = count();
            implementation = which();
            if (retained() != !!strstr(name, "cached-plan") ||
                plans != (retained() ? 1UL : (unsigned long)calls+1) ||
                !implementation || !*implementation || strchr(implementation, ','))
                fatal("oneDNN primitive identity/count");
        }
    }
    printf("AUDIT,%016" PRIx64 ",%zu,%lu,%s,1,%d,%d,%d,%d,%d,%d\n",
           hash,code_bytes,plans,implementation,c.rows,c.cols,a.cols,a.ld,b.ld,c.ld);
}
