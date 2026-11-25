/* GCC/Clang CPU dispatch includes the x86 OS vector-state checks. */
#include <stdio.h>
int main(void)
{
    __builtin_cpu_init();
    printf("%d %d\n", !!(__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")),
           !!__builtin_cpu_supports("avx512f"));
    return 0;
}
