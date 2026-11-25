"""Trace public FP32 kernel entries of the pinned DSO; keep every argument register intact."""
from pathlib import Path
import re
import subprocess
import sys

library, build = map(Path,sys.argv[1:])
lines=subprocess.check_output(['nm','-D','--defined-only',library],text=True).splitlines()
prefixes=('a64_sgemm_asimd','a64_hybrid_fp32_mla','a64_interleaved_fp32_mla','a64_smallK_hybrid_fp32_mla','sve_sgemm','sve_hybrid_fp32_mla','sve_interleaved_fp32_mla','sve_interleaved_fp32_mmla','sme_interleaved_nomerge_fp32_mopa','sme2_interleaved_nomerge_fp32_mopa')
matcher=re.compile(r' T (_ZN8arm_gemm\d+(?:'+'|'.join(prefixes)+r')\w+)$')
names=sorted({m[1] for line in lines if (m:=matcher.search(line))})
assert len(names)>=8
assembly=['.text','.hidden acl_called','.hidden acl_target']
for i,s in enumerate(names):
    assembly+=['.balign 4',f'.global {s}',f'.type {s}, %function',s+':','adrp x16, acl_called','add x16, x16, :lo12:acl_called',f'ldr x17, [x16, #{i*8}]','add x17, x17, #1',f'str x17, [x16, #{i*8}]','adrp x16, acl_target','add x16, x16, :lo12:acl_target',f'ldr x16, [x16, #{i*8}]','br x16',f'.size {s}, .-{s}']
(build/'wrap.S').write_text('\n'.join(assembly+['.section .note.GNU-stack,"",%progbits'])+'\n')
pretty=subprocess.check_output(['c++filt',*names],text=True).splitlines()
(build/'counts.c').write_text('''#define _GNU_SOURCE
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
__attribute__((visibility("hidden"))) uint64_t acl_called[N];
__attribute__((visibility("hidden"))) void *acl_target[N];
static const char *const mangled[]={MANGLED}, *const label[]={PRETTY};
int rank_acl_num_routes(void){return N;}
const char *rank_acl_name(int i){return label[i];}
uint64_t rank_acl_count(int i){return acl_called[i];}
int rank_acl_bind(void){
    void *lib = dlopen("libarm_compute.so",RTLD_NOW|RTLD_NOLOAD);
    if(!lib) return -1;
    for(int i=0;i<N;++i){
        Dl_info info;
        acl_target[i]=dlsym(lib,mangled[i]);
        if(!acl_target[i] || !dladdr(acl_target[i],&info) || !info.dli_fname || !strstr(info.dli_fname,"/libarm_compute.so")) return -2;
    }
    return 0;
}
'''.replace('MANGLED',','.join('"'+s+'"' for s in names)).replace('PRETTY',','.join('"'+s.split('(')[0]+'"' for s in pretty)).replace('[N]',f'[{len(names)}]').replace('return N;',f'return {len(names)};').replace('i<N;',f'i<{len(names)};'))
