"""Run each compiled ordinary DSO, preserve JIT bytes and check full numerical/guard receipts."""
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import tempfile


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def execute(work, vendor, proof, row, method):
    checked = method == 'checked'
    family = row['family']
    dims = 'm n k lda ldb ldc ar ac br bc cr cc'.split()
    args = [str(row[d]) for d in dims]
    m, n, k, lda, ldb, ldc, ar, ac, br, bc, cr, cc = map(int, args)
    cpu, = os.sched_getaffinity(0)
    module = work/'adapters'/('s8-dispatch.so' if checked else 'onednn-int8.so')
    probe = work/'checked-int8-probe'
    runtime = vendor/'onednn/prefix/lib/libdnnl.so.3'
    env = dict(os.environ, EXACT_FALLBACK_PLUGIN=str(work/'adapters/fp32-policy.so'),
               EXACT_INT8_DNNL_PATH=str(runtime), ONEDNN_VERBOSE='1', ONEDNN_JIT_DUMP='1')
    platform_arch = platform.machine()
    opcode = 'tdpbssd' if platform_arch == 'x86_64' else 'smmla'
    flags = ['-m','i386:x86-64','-M','intel'] if platform_arch == 'x86_64' else ['-m','aarch64']
    target = proof/'jit'/f'{family}-{method}'
    target.mkdir(parents=True, exist_ok=True)
    kernels = []
    with tempfile.TemporaryDirectory(prefix='checked-jit-',dir=work) as tmp:
        run = subprocess.run([probe,method,module,*args],env=env,cwd=tmp,capture_output=True,text=True,timeout=240)
        lines = (run.stdout+'\n'+run.stderr).splitlines()
        for file in sorted(Path(tmp).glob('dnnl_dump_cpu_*.bin')):
            if not re.search(r'brgemm|gemm|matmul', file.name):
                continue
            assembly = subprocess.check_output(['objdump','-D','-b','binary',*flags,file],text=True)
            matches = [s.strip() for s in assembly.splitlines() if re.search(r'\b'+opcode+r'\b',s)]
            if matches:
                (target/file.name).write_bytes(file.read_bytes())
                (target/(file.name+'.opcodes')).write_text('\n'.join(matches)+'\n')
                kernels.append(dict(path=str((target/file.name).relative_to(proof)),sha256=sha(file),
                                    operation=opcode,sites=len(matches)))
    proof_lines = [x.split('\t')[1:] for x in lines if x.startswith('PROOF\t')]
    events = [x for x in lines if ',exec,cpu,matmul,' in x and all(re.search(r'\b'+key+r'[:_]'+kind+r'\b',x)
                    for key,kind in [('src','s8'),('wei','s8'),('dst','s32')])]
    trace_name = 'S8_ROUTE' if checked else 'INT8'
    traced = [x.split('\t')[2:] for x in lines if x.startswith(trace_name+'\tPROBE\t')]
    page = os.sysconf('SC_PAGESIZE')
    guard_count = ((cr+m+2)*ldc*4+page-1)//page*(page//4)-m*n
    expected = [method,str(m*n),str(guard_count),str(cpu),'1','readonly',
                str(4*(ar*lda+ac)%64),str(4*(br*ldb+bc)%64),str(4*(cr*ldc+cc)%64)]
    full = run.returncode == 0 and proof_lines == [expected] and len(traced)==1
    ran_integer = full and ((traced[0][0]=='1' and traced[0][6]=='1') if checked else traced[0][0]=='1')
    impl = traced[0][-1] if traced else ''
    native = bool(ran_integer and kernels and any(','+impl+',' in event for event in events))
    result = dict(family=family,method=method,shape=[m,n,k],parent_args=args,architecture=platform_arch,cpu=cpu,
                affinity=[cpu],returncode=run.returncode,full_output_and_guards=full,exact_native=native,
                output_receipts=proof_lines,route_receipts=traced,matmul_events=events,
                implementation=impl,jit_kernels=kernels,objdump_args=['objdump','-D','-b','binary',*flags],
                worker_sha256=sha(probe),adapter_sha256=sha(module),onednn_sha256=sha(runtime),
                runtime=[s for s in lines if ',info,' in s and 'runtime:' in s],
                diagnostics=[s[-300:] for s in lines[-4:]] if not full else [])
    (proof/f'{family}-{method}-actual.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
