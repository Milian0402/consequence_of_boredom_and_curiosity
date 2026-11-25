"""Rebuild the frozen direct compact experiment and run the two-seed native M5 panel."""
from pathlib import Path
import argparse,csv,os,subprocess as sp,shutil
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('source_repo',type=Path,help='Local release repository containing 602a9dddce7e940a282b0716a7b2ca40bd2ad35e')
p.add_argument('libxsmm',type=Path,help='LIBXSMM headers and lib/libxsmm.a')
p.add_argument('peers_csv',type=Path,help='CSV method,path with Accelerate,OpenBLAS,ArmPL,MaxMulSK,MpGEMM adapter dylibs')
p.add_argument('new_directory',type=Path)
args=p.parse_args();kit=Path(__file__).resolve().parent;out=args.new_directory.resolve();out.mkdir(parents=True,exist_ok=False)
for name in ['common-bench.c','mtmm.h','adapter.c','first.c','bench.c','score.py','panel/panel.csv']:
 target=out/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(kit/name,target)
source=out/'source';base='602a9dddce7e940a282b0716a7b2ca40bd2ad35e'
def run(*cmd,**extra):sp.run(cmd,check=True,**extra)
run('git','clone','--quiet','--local','--no-checkout',args.source_repo.resolve(),source)
run('git','-C',source,'checkout','--detach',base)
methods={}
for name in ('incumbent','candidate'):
 if name=='candidate':run('git','-C',source,'apply',kit/'direct.patch')
 build=out/(name+'-build');dst=out/(name+'-install')
 run('cmake','-S',source,'-B',build,'-DCOB_STRASSEN_SME=ON','-DCOB_ENABLE_APPLE_MEDIUM_WIDE=ON','-DCMAKE_BUILD_TYPE=Release')
 run('cmake','--build',build,'--target','cob_matmul','-j','2');run('cmake','--install',build,'--prefix',dst)
 methods[name]=out/(name+'.dylib');run('cc','-O3','-dynamiclib','-I'+str(dst/'include'),out/'adapter.c','-Wl,-force_load,'+str(dst/'lib/libcob_matmul.a'),'-o',methods[name])
with args.peers_csv.open() as f:peers={r['method']:Path(r['path']).resolve() for r in csv.DictReader(f)}
assert set(peers)=={'Accelerate','OpenBLAS','ArmPL','MaxMulSK','MpGEMM'}
methods.update({n:peers[n] for n in ('Accelerate','OpenBLAS','ArmPL','MaxMulSK','MpGEMM')})
methods['prepared-LIBXSMM']='prepared'
xs=args.libxsmm.resolve();env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',OMP_THREAD_LIMIT='1',VECLIB_MAXIMUM_THREADS='1')
rows=list(csv.DictReader((out/'panel/panel.csv').open()))
for name,exe in [('holdout','bench'),('first','first')]:
 run('cc','-O3','-std=c11','-I'+str(xs/'include'),out/(exe+'.c'),xs/'lib/libxsmm.a','-o',out/exe)
 with (out/(name+'.csv')).open('w') as f,(out/(name+'.log')).open('w') as log:
  csv.writer(f).writerow(next(csv.reader((kit/(name+'.csv')).open())));f.flush()
  for block,seed in enumerate((427297853,1670820329),1):
   for row in (rows if block==1 else reversed(rows)):
    common=[out/exe,*[row[k] for k in ('m','n','k','pad')],str(seed),str(block)]
    use={k:v for k,v in methods.items() if row['pad']=='0' or k not in ('MaxMulSK','MpGEMM')}
    if name=='first':
     for k,v in (use.items() if block==1 else reversed(use.items())):run(*common,k,v,env=dict(env,FRONTIER_FIRST_ONLY='1'),stdout=f,stderr=log)
    else:run(*common,*[v for kv in use.items() for v in kv],env=env,stdout=f,stderr=log)
run('python3',out/'score.py',out)
