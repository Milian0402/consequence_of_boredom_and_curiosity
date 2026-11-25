"""Rebuild and time both installed public archives using the frozen Apple panel."""
import argparse,csv,os,shutil,subprocess as sp
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('release_repo',type=Path);p.add_argument('libxsmm_checkout',type=Path);p.add_argument('peer_paths_csv',type=Path);p.add_argument('new_directory',type=Path);args=p.parse_args()
h=Path(__file__).resolve().parent;out=args.new_directory.resolve();out.mkdir(parents=True,exist_ok=False);(out/'panel').mkdir()
for n in ('panel/panel.csv','score.py'):shutil.copyfile(h/n,out/n)
with args.peer_paths_csv.open() as f:peers={r['method']:Path(r['path']).resolve() for r in csv.DictReader(f)}
peer_order=('Accelerate','OpenBLAS','ArmPL','MaxMulSK','MpGEMM')
assert set(peers)==set(peer_order)
src=out/'source';base='602a9dddce7e940a282b0716a7b2ca40bd2ad35e'
def run(*cmd,**kw):sp.run(cmd,check=True,**kw)
run('git','clone','--quiet','--local','--no-checkout',args.release_repo.resolve(),src)
run('git','-C',src,'checkout','--quiet','--detach',base)
binary={}
for name in ('incumbent','candidate'):
 if name=='candidate':run('git','-C',src,'apply',h/'production.patch')
 build=out/(name+'-build');dst=out/(name+'-install')
 run('cmake','-S',src,'-B',build,'-DCMAKE_BUILD_TYPE=Release','-DCOB_STRASSEN_SME=ON','-DCOB_ENABLE_APPLE_MEDIUM_WIDE=ON')
 run('cmake','--build',build,'--target','cob_matmul','-j','2');run('cmake','--install',build,'--prefix',dst)
 binary[name]=out/(name+'.dylib')
 run('cc','-O3','-dynamiclib','-I'+str(dst/'include'),h/'adapter.c','-Wl,-force_load,'+str(dst/'lib/libcob_matmul.a'),'-o',binary[name])
methods={'A':binary['incumbent'],'B':binary['incumbent'],'D':binary['candidate'],**{k:peers[k] for k in peer_order},'prepared-LIBXSMM':'prepared'}
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',OMP_THREAD_LIMIT='1',VECLIB_MAXIMUM_THREADS='1')
xs=args.libxsmm_checkout.resolve()
for x in ('study','first'):run('cc','-O3','-std=c11','-I'+str(h),'-I'+str(xs/'include'),h/(x+'.c'),xs/'lib/libxsmm.a','-o',out/x)
panel=list(csv.DictReader((h/'panel/panel.csv').open()))
for mode in ('warm','first'):
 exe=out/('study' if mode=='warm' else 'first')
 with (out/(mode+'.csv')).open('w') as f,(out/(mode+'.log')).open('w') as log:
  csv.writer(f).writerow(next(csv.reader((h/'integrated'/(mode+'.csv')).open())));f.flush()
  for block,seed in enumerate((804162307,2065108391),1):
   for s in (panel if block==1 else reversed(panel)):
    dims=[s[k] for k in ('m','n','k','pad')];use={k:v for k,v in methods.items() if (s['pad']=='0' or k not in ('MaxMulSK','MpGEMM')) and (mode!='first' or k!='B')}
    common=[exe,*dims,str(seed),str(block)]
    if mode=='warm':run(*common,*[v for kv in use.items() for v in kv],stdout=f,stderr=log,env=env)
    else:
     for k,v in (use.items() if block==1 else reversed(use.items())):run(*common,k,v,stdout=f,stderr=log,env=dict(env,FRONTIER_FIRST_ONLY='1'))
run('python3',h/'score.py',out)
