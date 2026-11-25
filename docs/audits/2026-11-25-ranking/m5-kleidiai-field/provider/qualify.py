"""Check the two ordinary benchmark binaries, then record ABI and numerical receipts."""
import json
from pathlib import Path
import re
import subprocess
import sys
import hashlib

here = Path(__file__).resolve().parent
build = Path(sys.argv[1]).resolve()
identity = json.loads((build/'identity.json').read_text())
check = build/'check'
subprocess.run(['xcrun','clang','-std=c11','-O2','-mcpu=apple-m5',here/'check.c','-o',check],check=True)

def run(*args):
    return subprocess.check_output(args,text=True).strip()

receipts = {}
# The upstream wrappers require a target advertising SVE2 to compile. Verify that all their nonstreaming
# compiler-generated machine code uses only fixed-width registers. The .S kernel objects manage streaming.
streaming_register = re.compile(r'(?<![A-Za-z_])(?:z(?:[0-9]|[12][0-9]|3[01])|p(?:[0-9]|1[0-5]))\b|\b(?:rdvl|addvl|addpl|rdsvl|inc[bhwd]|dec[bhwd]|cnt[bhwd]|smstart|smstop)\b')
for name in ('fixed','elastic'):
    lib=build/f'kleidiai-{name}.dylib'
    if hashlib.sha256(lib.read_bytes()).hexdigest()!=identity['products'][name]['sha256']:
        raise ValueError('Unexpected binary digest')
    checked = []
    for obj in sorted((build/name).glob('*.o')):
        if obj.name.endswith('_asm.o'): continue
        asm=run('xcrun','llvm-objdump','-d','--mattr=+sme2,+sve2',obj)
        instructions=[s.split('\t',1)[1] for s in asm.splitlines() if re.match(r'^\s*[a-f0-9]+:\s+[0-9a-f]+\s*\t',s)]
        if not instructions or any(streaming_register.search(i) or '<unknown>' in i for i in instructions):
            raise ValueError('Compiler emitted unexpected nonstreaming code in '+obj.name)
        checked.append({'object':obj.name,'instructions':len(instructions)})
    if not checked: raise ValueError('Missing compiled C objects')
    syms=run('nm','-m',lib)
    for symbol in ('___arm_tpidr2_save','_kai_commit_za','_mtmm_sgemm_rowmajor'):
        if not any('(__TEXT,__text)' in line and line.endswith(' '+symbol) for line in syms.splitlines()):
            raise ValueError('Missing defined symbol '+symbol)
    metrics=dict(field.split('=') for field in run(check,lib).split(','))
    if int(metrics['fp64_cells'])!=122382 or int(metrics['c_guard_cells'])!=111090 or int(metrics['threads_before_after'])!=1 or float(metrics['max_abs_error'])>.002 or int(metrics['svcr_after'])!=0:
        raise ValueError(metrics)
    receipts[name]={'metrics':metrics,'ordinary_c_objects':checked,'library_sha256':identity['products'][name]['sha256']}
result={'compiler':identity['compiler'],'upstream_revision':identity['revision'],'upstream_tree':identity['tree'],
        'macos':run('sw_vers','-productVersion'),'build':run('sw_vers','-buildVersion'),'receipt':receipts}
(build/'qualification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:r['metrics'] for k,r in receipts.items()},indent=2))
