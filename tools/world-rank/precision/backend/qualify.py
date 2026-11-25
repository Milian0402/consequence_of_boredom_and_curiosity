"""Check native BF16 operation identity, generated instructions, signed exact products and complete lifecycles."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import tempfile


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as src:
        for block in iter(lambda:src.read(1<<20), b''):
            h.update(block)
    return h.hexdigest()


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def one(lines, prefix):
    values = [row.split(',')[1:] for row in lines if row.startswith(prefix+',')]
    require(len(values)==1,'missing or repeated '+prefix)
    return values[0]


def inspect_dump(folder, arch):
    objs=[]
    for p in sorted(folder.glob('dnnl_dump_cpu_*.bin')):
        if arch=='x86_64':
            flags=['-m','i386:x86-64','-M','intel']
            matcher=r'\btdpbf16ps\b'
        else:
            flags=['-m','aarch64']
            matcher=r'\b(?:bfdot|bfmmla)\s+z\d+\.'
        code=subprocess.check_output(['objdump','-D','-b','binary',*flags,p],text=True)
        hits=[s[found.start():].strip() for s in code.splitlines() if (found:=re.search(matcher,s,re.I))]
        if hits:
            objs.append({'kernel_file':p.name,'bytes':p.stat().st_size,'sha256':digest(p),
                         'instruction_sites':len(hits),'instruction_example':hits[0]})
    return objs


def run(probe, geom, diagnostic=False, emulated=False):
    arch=platform.machine()
    cpu=min(os.sched_getaffinity(0))
    records=Path('/proc/cpuinfo').read_text().splitlines()
    info=lambda label:next((s.split(':',1)[1].strip() for s in records if s.split(':')[0].strip()==label), '')
    features=set((info('flags') or info('Features')).split())
    command=['taskset','-c',str(cpu)]
    if emulated: command+=['qemu-aarch64','-cpu','neoverse-n2']
    command += [str(probe.resolve()),*map(str,geom),'diagnostic' if diagnostic else 'native-only']
    env=dict(os.environ,ONEDNN_VERBOSE='1',ONEDNN_JIT_DUMP='1',OMP_NUM_THREADS='1',OMP_THREAD_LIMIT='1')
    out={'architecture':arch,'affinity_cpu':cpu,'emulated':emulated,'shape_and_strides':geom,
         'host_relevant_features':sorted(features&{'sve','svebf16','bf16','amx_bf16','amx_tile','avx512_bf16'}),
         'host_model':info('model name') or '/'.join((info('CPU implementer'),info('CPU part'))),
         'probe_sha256':digest(probe),'bridge_sha256':digest(probe.parent/'libbf16_backend.so'),
         'build_receipt_sha256':digest(probe.parent/'bf16-build.txt'),'eligible':False}
    fingerprint=(probe.parent/'bf16-build.txt').read_text().splitlines()
    for filename, expected in ((probe.name,out['probe_sha256']),('libbf16_backend.so',out['bridge_sha256'])):
        require(expected+'  '+filename in fingerprint,'native executable disagrees with compile/link receipt: '+filename)
    mapped=subprocess.check_output(['ldd',str(probe.resolve())],text=True,env=env)
    for soname,expected in [('libbf16_backend.so',out['bridge_sha256']),('libdnnl.so.3',None)]:
        paths=re.findall(r'^\s*'+re.escape(soname)+r' => (\S+)',mapped,re.MULTILINE)
        require(len(paths)==1 and Path(paths[0]).is_file(),'missing dynamic provider '+soname)
        actual=digest(Path(paths[0]))
        require(actual==expected if expected else 'libdnnl_sha256='+actual in fingerprint, 'runtime digest mismatch for '+soname)
        if not expected: out['mapped_onednn_sha256']=actual
    with tempfile.TemporaryDirectory() as tmp:
        work=Path(tmp)
        result=subprocess.run(command,cwd=work,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        lines=result.stdout.splitlines()
        out['returncode']=result.returncode
        out['raw_worker_records']=[s for s in lines if s.startswith('BF16_')]
        out['verbose_exec_records']=[s for s in lines if 'verbose' in s and ',exec,cpu,matmul,' in s]
        out['verbose_runtime_records']=[s for s in lines if 'verbose' in s and ',info,' in s and ('runtime' in s.lower() or 'isa:' in s.lower())]
        out['jit_dump_count']=len(list(work.glob('dnnl_dump_cpu_*.bin')))
        try:
            require(result.returncode==0,'worker failed, status '+str(result.returncode))
            status,impl,route,native=one(lines,'BF16_PLAN')
            status,native=int(status),int(native)
            out.update(status=status,implementation=impl,route=route,native_descriptor=bool(native))
            life=list(map(int,one(lines,'BF16_LIFE')))
            executed=[r.split(',')[1:] for r in lines if r.startswith('BF16_EXEC,')]
            require(life==[int(status==0),len(executed),int(status==0)],'plan lifetime mismatch')
            out['lifecycle']=dict(zip(('created','executed','destroyed'),life))
            if status!=0:
                reason=one(lines,'BF16_ERROR')
                out.update(reason='descriptor-unavailable' if status==3 else 'descriptor-error',oneDNN_message=','.join(reason))
                return out
            require(impl and native in (0,1) and len(executed)==(2 if native or diagnostic else 0),'incorrect descriptor/geometry operation count')
            m,n,_,_,_,ldc=geom
            for t,row in enumerate(executed):
                order,ret,checked,guards,wrong,changed,worst,inputs,threads,actual_cpu,execs=row
                require(int(order)==t and int(ret)==0 and int(checked)==m*n and int(guards)==7+3*t+m*ldc+17-m*n and int(wrong)==int(changed)==0 and
                        math.isfinite(float(worst)) and float(worst)==0 and int(inputs)==1 and int(actual_cpu)==cpu and int(execs)==t+1,
                        'guarded BF16/FP32 exactness or counts failed')
                require(emulated or int(threads)==1,'extra native process threads')
            out['checked_outputs']=len(executed)*m*n
            if not native:
                out.update(reason='outside-registered-native-routes')
                return out
            verbose=out['verbose_exec_records']
            require(len(verbose)==2 and all(','+impl+',' in s and
                    all(re.search(r'\b'+field+r'[:_]'+dt+r'\b',s) for field,dt in [('src','bf16'),('wei','bf16'),('dst','f32')])
                    for s in verbose),'matching executed BF16 × BF16 -> F32 verbose record missing')
            require(any('runtime:sequential,nthr:1' in s for s in out['verbose_runtime_records']), 'missing single-thread runtime declaration')
            require(all(f',{m}x{geom[2]}:{geom[2]}x{n},' in s for s in verbose),'oneDNN problem dimensions disagree')
            evidence=inspect_dump(work,arch)
            out['jit_bf16_instructions']=evidence
            computational=[x for x in evidence if re.match(r'dnnl_dump_cpu_jit_(brgemm|bf16_matmul)',x['kernel_file'])]
            if not computational:
                out.update(reason='no-independent-BF16-instruction-witness')
                return out
            out.update(eligible=not emulated, reason='emulated-instruction-correctness-only' if emulated else 'verified-native-instruction-and-output')
        except (ValueError,subprocess.SubprocessError, OSError) as error:
            out.update(reason='qualification-error',error=str(error),diagnostic_tail=lines[-12:])
    return out


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('probe',type=Path);cli.add_argument('output',type=Path)
    cli.add_argument('geometry',nargs=6,type=int,metavar='dim')
    cli.add_argument('--diagnostic',action='store_true',help='Also execute non-native fallback for correctness.')
    cli.add_argument('--emulate-sve-128',action='store_true',help='Local QEMU correctness only, always ineligible for native ranking.')
    a=cli.parse_args()
    try:
        data=run(a.probe,a.geometry,a.diagnostic,a.emulate_sve_128)
    except (ValueError,OSError,subprocess.SubprocessError) as e:
        data=dict(eligible=False,reason='qualification-error',error=str(e),shape_and_strides=a.geometry)
    a.output.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    print(data['reason'],data.get('implementation',''),data.get('route',''),data.get('checked_outputs',0))
    if data['reason'] in ('qualification-error','descriptor-error'): raise SystemExit(1)
