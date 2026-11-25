"""Prove the linked Advanced SIMD BFMMLA path by ordinary and instrumented whole-call execution."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess


SYMBOL = 'a64_interleaved_bf16fp32_mmla_8x12'
ROUTE = 'arm-acl-neon-bfmmla'
SDK_SHA = '408cf4d5b3ea81031f915506f90977fba7549abc7dbff5929562b5767e215117'
MANGLED = '_ZN8arm_gemm34a64_interleaved_bf16fp32_mmla_8x12EPKN11arm_compute8bfloat16ES3_Pfiii'
DSO_SHA = '321fd6d02efe5d0cd1fa126f581d2addb5b36bdf7ab3c288c7b4d9c55bb5e8f0'


def check(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(name, probe, geometry, cpu, emulator):
    env = dict(os.environ, OMP_NUM_THREADS='1', OMP_THREAD_LIMIT='1')
    cmd = ['taskset', '-c', str(cpu), *emulator, str(probe.resolve()), *map(str, geometry), 'native-only']
    process = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    records = process.stdout.splitlines()
    def rows(kind):
        return [line.split(',')[1:] for line in records if line.startswith(kind + ',')]
    plans, life = rows('BF16_PLAN'), rows('BF16_LIFE')
    check(process.returncode == 0 and len(plans) == len(life) == 1,
          f'{name}: incomplete worker or duplicate descriptor/lifetime: {process.returncode}, {records[-6:]}')
    status, implementation, route, native = plans[0]
    common = dict(status=int(status), implementation=implementation, route=route, native_descriptor=native == '1',
                  probe_sha256=sha(probe), lifecycle=list(map(int, life[0])), raw_worker_records=records)
    numerical = rows('BF16_EXEC')
    traces = rows('ACL_BF16_KERNEL')
    if int(status) != 0:
        check(life == [['0','0','0']] and not numerical and not traces and len(rows('BF16_ERROR')) == 1,
              f'{name}: failed creation has execution evidence or lacks a description')
        return {**common, 'reason': ','.join(rows('BF16_ERROR')[0])}
    check((implementation,route,native) == (SYMBOL,ROUTE,'1') and life == [['1','2','1']] and len(numerical) == 2,
          f'{name}: true BFMMLA implementation, native flag, or plan lifecycle mismatch')
    m,n,k,_,_,ldc = geometry
    checked=guards=0
    for index,row in enumerate(numerical):
        trial, ret, count, guard, wrong, corrupt, error, unchanged, threads, actual_cpu, calls = row
        check((int(trial),int(ret),int(count),int(guard),int(wrong),int(corrupt),int(unchanged),int(actual_cpu),int(calls)) ==
              (index,0,m*n,7+3*index+m*ldc+17-m*n,0,0,1,cpu,index+1), f'{name}: guard/count/cpu failure')
        check(math.isfinite(float(error)) and float(error) == 0, f'{name}: wrong logical FP32 output')
        check(1 <= int(threads) if emulator else int(threads) == 1, f'{name}: extra native thread')
        checked += int(count)
        guards += int(guard)
    check(len(traces) == (2 if name == 'trace' else 0), f'{name}: expected kernel trace count')
    a_blocks=(m+7)//8
    b_blocks=(n+11)//12
    kp=(k+3)//4*4
    expected={'name':SYMBOL,'calls':str(((a_blocks+7)//8)*((b_blocks+7)//8)),
              'abytes':str(a_blocks*8*kp*2),'bbytes':str(b_blocks*12*kp*2),
              'cbytes':str(min(a_blocks,8)*min(b_blocks,8)*8*12*4),'allocations':'2',
              'm':str(m),'n':str(n),'k':str(k),'ablocks':str(a_blocks),'bblocks':str(b_blocks),'kp':str(kp),'planbytes':'96'}
    for trace in traces:
        pairs=[field.split('=',1) for field in trace]
        check(len(pairs)==len(expected) and dict(pairs)==expected, f'{name}: actual tile/pack/call count mismatch: {pairs}')
    return {**common, 'checked_outputs':checked, 'guard_cells':guards,'observed_trace':expected if traces else None}


def qualify(folder, geometry, emulated):
    cpu=min(os.sched_getaffinity(0))
    emulator=shlex.split(os.environ.get('ACL_PROBE_PREFIX',''))
    check(not emulator or emulated, 'emulator prefix requires --emulated')
    check(not emulated or emulator, '--emulated requires ACL_PROBE_PREFIX')
    arch=platform.machine()
    check(arch=='aarch64' or emulated, 'expected a Linux AArch64 host')
    identity_file=folder/'acl-identity.json'
    identity=json.loads(identity_file.read_text())
    check((identity['sdk_sha256'],identity['acl_dso_sha256'],identity['kernel_symbol']) ==
          (SDK_SHA,DSO_SHA,MANGLED),'pinned SDK, provider or symbol differs')
    hashes=identity['sha256']
    for item in ('acl-probe','acl-probe-trace','libacl_bf16_backend.so','libacl_bf16_trace.so','backend.o','acl-kernel.txt'):
        check(hashes[item] == sha(folder/item), f'{item}: linked or disassembled build hash mismatch')
    dso=folder/'arm_compute-v53.3.2-linux-aarch64-cpu-bin/lib/armv8a-neon/libarm_compute.so'
    check(sha(dso)==DSO_SHA,'actual kernel DSO sha256 mismatch')
    contents=(folder/'acl-kernel.txt').read_text()
    check(re.findall(r'^\s*[0-9a-f]+ <([^>]+)>:$',contents,re.MULTILINE)==[MANGLED], 'disassembly must cover exactly the target kernel symbol')
    for probe in ('acl-probe','acl-probe-trace'):
        check(MANGLED in (folder/(probe+'-kernel-import.txt')).read_text(),'actual bridge must import DSO kernel')
        deps=(folder/(probe+'-deps.txt')).read_text()
        found=re.findall(r'libarm_compute\.so => (\S+)',deps)
        check(len(found)==1 and Path(found[0]).samefile(dso),f'{probe}: dynamic loader did not resolve the pinned DSO')
    sites=[s.strip() for s in contents.splitlines() if re.search(r'\bbfmmla\s+v\d+\.4s,\s*v\d+\.8h,\s*v\d+\.8h',s,re.I)]
    check(len(sites)==identity['bfmmla_static_count'],'disassembly site count disagrees with receipt')
    check(sites,'missing binary BFMMLA V-register instruction in the official kernel')
    out=dict(architecture=arch,affinity_cpu=cpu,emulated=emulated,eligible=False,shape_and_strides=geometry,
             build_receipt_sha256=sha(identity_file),kernel_disassembly_sha256=sha(folder/'acl-kernel.txt'),
             opcode_sites=len(sites),instruction_example=sites[0])
    out['source_identity']=identity
    labels={}
    for line in Path('/proc/cpuinfo').read_text().splitlines():
        key,_,value=line.partition(':')
        labels.setdefault(key.strip(),value.strip())
    features=set(labels.get('Features','').split())
    out.update(host_model=labels.get('model name') or '/'.join((labels.get('CPU implementer',''),labels.get('CPU part',''))),
               host_relevant_features=sorted(features & {'asimd','bf16','sve','svebf16'}))
    for name,filename in [('ordinary','acl-probe'),('trace','acl-probe-trace')]:
        out[name]=execute(name,folder/filename,geometry,cpu,emulator)
    ordinary,trace=out['ordinary'],out['trace']
    check((ordinary['status'],ordinary['route'],ordinary['implementation']) ==
          (trace['status'],trace['route'],trace['implementation']),'ordinary and trace route disagree')
    if ordinary['status']!=0:
        out.update(reason='hardware-unavailable' if ordinary['status']==3 else 'descriptor-error')
    else:
        check(emulated or {'asimd','bf16'} <= features,'kernel ran without host ASIMD/BF16 CPU features')
        out.update(eligible=not emulated, reason='emulated-instruction-correctness-only' if emulated else 'verified-native-instruction-and-output',
                   implementation=SYMBOL,route=ROUTE, checked_outputs=ordinary['checked_outputs']+trace['checked_outputs'],
                   guard_cells=ordinary['guard_cells']+trace['guard_cells'])
    return out


if __name__ == '__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('folder',type=Path);cli.add_argument('output',type=Path)
    cli.add_argument('geometry',nargs=6,type=int,metavar='dim')
    cli.add_argument('--emulated',action='store_true')
    args=cli.parse_args()
    try:
        result=qualify(args.folder,args.geometry,args.emulated)
    except (ValueError,KeyError,IndexError,OSError,subprocess.SubprocessError) as e:
        result={'eligible':False,'reason':'qualification-error','error':str(e),'shape_and_strides':args.geometry}
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(result['reason'],result.get('implementation',''),result.get('checked_outputs',0),result.get('error',''))
    if result['reason'] in ('qualification-error','descriptor-error'):
        raise SystemExit(1)
