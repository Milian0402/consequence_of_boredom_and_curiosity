/* Untimed macOS/AArch64 feature probe. The MOP4 opcode is copied from
   KleidiAI v1.33.0's 2vsx16vs FP32 leaf. Run as a standalone process. */
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/sysctl.h>
#include <unistd.h>

static void illegal(int signal_number)
{
    (void)signal_number;
    static const char msg[] = "instruction=SIGILL\n";
    (void)write(STDOUT_FILENO, msg, sizeof(msg)-1);
    _exit(77);
}

int main(int argc, char **argv)
{
    if (argc != 2) return 2;
    if (!strcmp(argv[1], "sysctl")) {
        const char *keys[] = {"hw.optional.arm.FEAT_SME", "hw.optional.arm.FEAT_SME2",
            "hw.optional.arm.FEAT_SME2p1", "hw.optional.arm.SME_F32F32",
            "hw.optional.arm.FEAT_SME_MOP4", "hw.optional.arm.FEAT_SME2p2",
            "hw.optional.arm.FEAT_SME2p3", "hw.optional.arm.FEAT_SME3", "hw.optional.arm.sme_max_svl_b"};
        for(size_t i=0; i<sizeof(keys)/sizeof(keys[0]); i++){
            int v; size_t n=sizeof(v);
            if (sysctlbyname(keys[i], &v, &n, NULL, 0)) printf("%s=unavailable (%s)\n",keys[i],strerror(errno));
            else printf("%s=%d\n",keys[i],v);
        }
        return 0;
    }
    struct sigaction handler = {0};
    handler.sa_handler = illegal;
    if (sigaction(SIGILL, &handler, NULL)) return 2;
    if (!strcmp(argv[1],"mopa")) {
        unsigned long svl;
        __asm__ volatile("smstart\n\tcntb %0\n\tzero {za}\n\tptrue p0.s\n\tfdup z4.s, #1.0\n\tfdup z16.s, #1.0\n\tfmopa za0.s, p0/m, p0/m, z4.s, z16.s\n\tsmstop":"=r"(svl)::"memory");
        printf("fp32_fmopa=executed svl_bytes=%lu\n",svl);
    } else if (!strcmp(argv[1],"mop4")) {
        __asm__ volatile("smstart\n\tzero {za}\n\tfdup z4.s, #1.0\n\tfdup z16.s, #1.0\n\tfdup z17.s, #1.0\n\t.inst 0x80100080\n\tsmstop":::"memory");
        puts("fp32_fmop4a=executed");
    } else return 2;
    return 0;
}
