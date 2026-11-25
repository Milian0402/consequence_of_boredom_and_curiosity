"""Build the two upstream KleidiAI FP32 SME2 public entry points on arm64 macOS."""
import argparse
import hashlib
import json
import platform
import shlex
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REV = 'ab3f4b24a75f0667a1af9ebfe27e94efa7ce97e1'
PACK = 'kai/ukernels/matmul/pack/kai_'
GEMM = 'kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/kai_matmul_clamp_f32_'
ELASTIC = GEMM + 'f32p4vsx1_f32p4vsx1bf32_'
FIXED = GEMM + 'f32p2vlx1_f32p2vlx1biasf32_sme2_mopa'

def pair(stem):
    return [stem + '.c', stem + '_asm.S']

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def output(*cmd):
    return subprocess.check_output(cmd, text=True).strip()

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('upstream', type=Path)
parser.add_argument('output', type=Path)
args = parser.parse_args()
src, out = args.upstream.resolve(), args.output.resolve()
if (platform.system(),platform.machine()) != ('Darwin','arm64') or output('git','-C',src,'rev-parse','HEAD') != REV:
    raise SystemExit('Requires arm64 macOS and the pinned upstream source')
subprocess.run(['git','-C',src,'diff','--exit-code','HEAD','--','kai'],check=True)
out.mkdir(parents=True,exist_ok=True)
cc = output('xcrun','--find','clang')
sdk = output('xcrun','--sdk','macosx','--show-sdk-path')
inputs = {}
products = {}
commands = []
for name in ('fixed','elastic'):
    files = ['kai/kai_common_sme_asm.S']
    if name == 'fixed':
        files += pair(FIXED) + pair(PACK + 'lhs_pack_f32p2vlx1_f32_sme') + pair(PACK + 'rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme')
    else:
        files += [ELASTIC + '8vsx8vs_sme2_mopa.c']
        files += [ELASTIC + tile + '_sme2_mopa_asm.S' for tile in ('4vsx4vs','4vsx8vs','4vsx16vs','8vsx4vs','8vsx8vs','16vsx4vs')]
        files += pair(PACK + 'matmul_pack_lhs_mxk_x32p4vsx1_x32_sme') + pair(PACK + 'matmul_pack_rhs_kxn_x32p4vsx1bx32_x32_x32_sme')
    objdir = out / name
    objdir.mkdir(exist_ok=True)
    objs = []
    for rel in files + ['adapter.c']:
        p = HERE/rel if rel == 'adapter.c' else src/rel
        core = p.suffix == '.c' and rel != 'adapter.c'
        arch = '-mcpu=apple-m5' if rel == 'adapter.c' else '-march=armv9.2-a+sme2'
        obj = objdir/(p.stem+'.o')
        opts = ['-O3', arch, '-isysroot',sdk, '-fPIC', '-I',str(src)]
        if core:
            # M5 supports streaming SVE only. Enforce fixed-width or scalar code in the ordinary C wrappers.
            opts += ['-fno-vectorize','-fno-slp-vectorize']
        if rel == 'adapter.c' and name == 'elastic': opts += ['-DKLEIDI_ELASTIC=1']
        cmd = [cc,*opts,'-c',str(p),'-o',str(obj)]
        commands.append(shlex.join(cmd)); subprocess.run(cmd,check=True); objs.append(obj)
        inputs[rel] = digest(p)
    dylib = out/f'kleidiai-{name}.dylib'
    cmd = [cc,'-isysroot',sdk,'-dynamiclib','-Wl,-undefined,error','-Wl,-install_name,@rpath/'+dylib.name,*map(str,objs),'-o',str(dylib)]
    commands.append(shlex.join(cmd));subprocess.run(cmd,check=True)
    products[name] = dict(sha256=digest(dylib),dependencies=output('otool','-L',dylib).splitlines()[1:])
(out/'commands.txt').write_text('\n'.join(commands)+'\n')
(out/'identity.json').write_text(json.dumps(dict(revision=REV,tree=output('git','-C',src,'rev-parse','HEAD^{tree}'),
    compiler=output(cc,'--version').splitlines()[0],inputs=inputs,products=products),indent=2)+'\n')
