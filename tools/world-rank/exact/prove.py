"""Authenticate the actual per-geometry BF16 and int8 providers outside timed calls."""
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'precision'
def store(path,obj):
    path.write_text(json.dumps(obj,indent=2)+'\n')
def int8(work,p,proof):
    label,m,n,k=p
    with tempfile.TemporaryDirectory(prefix='int8-jit-',dir=work) as tmp:
        env=dict(os.environ,ONEDNN_VERBOSE='1',ONEDNN_JIT_DUMP='1')
        proc=subprocess.run([work/'int8-probe',work/'adapters/onednn-int8.so',m,n,k],env=env,cwd=tmp,capture_output=True,text=True,timeout=240)
        lines=(proc.stdout+'\n'+proc.stderr).splitlines()
        records=[v for v in lines if v.startswith('INT8_PROOF,')]
        events=[v for v in lines if ',exec,cpu,matmul,' in v]
        parsed=records[0].split(',') if len(records)==1 else []
        impl=parsed[1] if len(parsed)==6 else ''
        flags=['-m','i386:x86-64','-M','intel'] if platform.machine()=='x86_64' else ['-m','aarch64']
        opre=r'\b(?:tdpbssd|tdpbusd|vpdpbssd|vpdpbusd|vpmaddubsw|sdot|udot|smmla|ummla|usmmla|sudot|usdot|smull)\b'
        opcodes=[]
        for file in Path(tmp).glob('dnnl_dump_cpu_*.bin'):
            if not re.search(r'brgemm|gemm|matmul',file.name):continue
            code=subprocess.check_output(['objdump','-D','-b','binary',*flags,file],text=True)
            hits=sorted(set(re.findall(opre,code,re.I)))
            if hits:
                opcodes.append(dict(kernel=file.name,sha256=hashlib.sha256(file.read_bytes()).hexdigest(),operations=hits))
        eligible=(proc.returncode==0 and impl and 'ref' not in impl and parsed[2:5]==['1',str(int(m)*int(n)),'1'] and
                  int(parsed[5])==min(os.sched_getaffinity(0)) and
                  any(','+impl+',' in e and all(t in e for t in ('src:s8','wei:s8','dst:s32')) for e in events))
        # oneDNN verbose output used ':' or '_' delimiters across versions.
        if not eligible and proc.returncode==0 and impl and 'ref' not in impl and parsed[2:5]==['1',str(int(m)*int(n)),'1'] and int(parsed[5])==min(os.sched_getaffinity(0)):
            eligible=any(','+impl+',' in e and all(re.search(r'\b'+tag+r'[:_]'+kind+r'\b',e) for tag,kind in [('src','s8'),('wei','s8'),('dst','s32')]) for e in events)
        record=dict(eligible=bool(eligible),implementation=impl,native_integer_opcodes_observed=bool(opcodes),shape=[int(m),int(n),int(k)],events=events,probe=records,jit_instruction_evidence=opcodes,returncode=proc.returncode,
                    runtime=[s for s in lines if ',info,' in s and 'runtime:' in s],reason='executed sequential signed-integer primitive' if eligible else 'descriptor, implementation or full integer output did not qualify',diagnostic=lines[-6:] if not eligible else [])
        store(proof,record);return record
def run(work,vendor,out):
    rows=list(csv.DictReader((HERE/'panel.csv').open())); proven={};prior={};dest=out/'proof';dest.mkdir(exist_ok=True)
    intel=platform.machine()=='x86_64';byshape={}
    for row in rows:
        label=row['family'];m,n,k=(row[x] for x in 'mnk');geo=(m,n,k)
        if geo not in byshape:
            path=dest/f'{label}-bf16.json'
            source=BASE/('backend' if intel else 'acl-backend')/'qualify.py'
            exe=work/'bf16/bf16-probe' if intel else work/'bf16'
            proc=subprocess.run([sys.executable,'-B',source,exe,path,m,n,k,k,n,n],capture_output=True,text=True,timeout=600)
            if path.is_file():data=json.loads(path.read_text())
            else:data=dict(eligible=False,reason='BF16 witness returned no receipt',returncode=proc.returncode)
            if proc.returncode not in (0,1):raise RuntimeError(f'BF16 shape probe failed {label}: {proc.stderr[-400:]}')
            alt=int8(work,(label,m,n,k),dest/f'{label}-int8.json');byshape[geo]=(label,data,alt)
        source_id,raw,alt=byshape[geo]
        fallback=('release-mkl-aligned-two' if intel else 'fastmmw-1536') if label=='square4096' else ('mkl-jit' if intel else 'armpl')
        where=work/'adapters' if fallback.startswith('release-') else vendor
        plugin=where/(fallback+'.so')
        if not plugin.is_file():raise FileNotFoundError(plugin)
        seen=dict(eligible=bool(raw.get('eligible')),implementation=raw.get('implementation',''),reason=raw.get('reason',''),
                  proof='proof/'+source_id+'-bf16.json',int8_eligible=alt['eligible'],int8_implementation=alt['implementation'],
                  int8_proof='proof/'+source_id+'-int8.json',fallback_method=fallback)
        prior[label]=seen;proven[label]=dict(**seen,fallback_plugin=str(work/'adapters/fp32-policy.so'))
    store(work/'native.json',proven);store(out/'admissions.json',prior)
if __name__=='__main__':run(*(Path(x).resolve() for x in sys.argv[1:4]))
