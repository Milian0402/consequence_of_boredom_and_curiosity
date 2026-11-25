"""Native replay from immutable local release, shipped patch, LIBXSMM, and the peer adapters."""
import argparse,csv,os,shutil,subprocess as sp,tarfile
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('release_repo',type=Path);p.add_argument('libxsmm_checkout',type=Path);p.add_argument('peers_csv',type=Path,help='CSV columns method,path for Accelerate/OpenBLAS/ArmPL/MaxMulSK/MpGEMM');p.add_argument('out',type=Path)
args=p.parse_args();h=Path(__file__).resolve().parent;out=args.out.resolve();out.mkdir(exist_ok=False,parents=True)
base='a30d71f0aaba4bcceb9ec5c530cffd63e6967cd6';xs=args.libxsmm_checkout.resolve()
with args.peers_csv.open() as f:peers={r['method']:str(Path(r['path']).resolve()) for r in csv.DictReader(f)}
assert set(peers)=={'Accelerate','OpenBLAS','ArmPL','MaxMulSK','MpGEMM'}
def run(*cmd,**kw):sp.run(cmd,check=True,**kw)
lib={}
for name,opt in [('release','OFF'),('candidate','ON'),('option-off','OFF')]:
 src=out/(name+'-src');b=out/(name+'-build');prefix=out/(name+'-install')
 run('git','clone','--quiet','--local','--no-checkout',args.release_repo.resolve(),src);run('git','-C',src,'checkout','--quiet','--detach',base)
 if name!='release':run('git','-C',src,'apply',h/'production.patch')
 run('cmake','-S',src,'-B',b,'-DCMAKE_BUILD_TYPE=Release','-DCOB_STRASSEN_SME=ON','-DCOB_ENABLE_APPLE_MEDIUM_WIDE='+opt)
 run('cmake','--build',b,'--target','cob_matmul','-j','2');run('cmake','--install',b,'--prefix',prefix)
 lib[name]=out/(name+'.dylib');run('cc','-O3','-dynamiclib','-I'+str(prefix/'include'),h/'adapter.c','-Wl,-force_load,'+str(prefix/'lib/libcob_matmul.a'),'-o',lib[name])
for name in ('bench','first'):run('cc','-O3','-std=c11','-I'+str(xs/'include'),h/(name+'.c'),xs/'lib/libxsmm.a','-o',out/name)
run('cc','-O3','-std=c11',h/'stress.c','-o',out/'stress')
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',OMP_THREAD_LIMIT='1',VECLIB_MAXIMUM_THREADS='1')
methods={**lib,**{m:peers[m] for m in ('Accelerate','OpenBLAS','ArmPL','MaxMulSK','MpGEMM')},'prepared-LIBXSMM':'prepared'}
shutil.copyfile(h/'panel.csv',out/'panel.csv')
with (h/'panel.csv').open() as f:shapes=list(csv.DictReader(f))
for exe,name in [('bench','production-times'),('first','production-first')]:
 with (out/(name+'.csv')).open('w') as f,(out/(name+'.log')).open('w') as log:
  csv.writer(f).writerow(next(csv.reader((h/(name+'.csv')).open())));f.flush()
  for block,seed in [(1,'318741563'),(2,'1557601921')]:
   for s in (shapes if block==1 else reversed(shapes)):
    common=[out/exe,*[s[d] for d in ('m','n','k','pad')],seed,str(block)]
    use={k:v for k,v in methods.items() if s['pad']=='0' or k not in ('MaxMulSK','MpGEMM')}
    if exe=='first':
     for k,v in (use.items() if block==1 else reversed(use.items())):run(*common,k,v,env=dict(env,FRONTIER_FIRST_ONLY='1'),stdout=f,stderr=log)
    else:run(*common,*[v for kv in use.items() for v in kv],env=env,stdout=f,stderr=log)
with (out/'stress.csv').open('w') as f:run(out/'stress',lib['candidate'],lib['option-off'],stdout=f)
