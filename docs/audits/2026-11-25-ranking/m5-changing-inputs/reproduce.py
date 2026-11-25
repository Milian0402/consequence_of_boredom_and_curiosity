"""Build identical public COB configurations and repeat the paired M5 hot/rotating studies."""
from pathlib import Path
import argparse,csv,os,shutil,subprocess as sp,sys
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('release_repo',type=Path);p.add_argument('libxsmm_checkout',type=Path);p.add_argument('peers',type=Path,help='CSV method,path for the seven ordinary external adapters');p.add_argument('new_directory',type=Path);p.add_argument('--only',choices=['prefaulted','wired'])
args=p.parse_args();h=Path(__file__).resolve().parent;out=args.new_directory.resolve();out.mkdir(parents=True,exist_ok=False)
def run(*cmd,**opts):sp.run(cmd,check=True,**opts)
libs={};src=out/'source'
run('git','clone','--quiet','--local','--no-checkout',args.release_repo.resolve(),src);run('git','-C',src,'checkout','--quiet','--detach','602a9dddce7e940a282b0716a7b2ca40bd2ad35e')
for name in ('incumbent','shipped'):
 if name=='shipped':run('git','-C',src,'apply',h/'production.patch')
 build=out/(name+'-build');dst=out/(name+'-install')
 run('cmake','-S',src,'-B',build,'-DCMAKE_BUILD_TYPE=Release','-DCOB_STRASSEN_SME=ON','-DCOB_ENABLE_APPLE_MEDIUM_WIDE=ON')
 run('cmake','--build',build,'--target','cob_matmul','-j','2');run('cmake','--install',build,'--prefix',dst)
 libs[name]=out/(name+'.dylib');run('cc','-O3','-dynamiclib','-I'+str(dst/'include'),h/'adapter.c','-Wl,-force_load,'+str(dst/'lib/libcob_matmul.a'),'-o',libs[name])
with args.peers.open() as f:peers={r['method']:Path(r['path']).resolve() for r in csv.DictReader(f)}
methods={'A':libs['incumbent'],'B':libs['incumbent'],'D':libs['shipped']}
for n in ('Accelerate','OpenBLAS','ArmPL','MaxMulSK','MpGEMM','prepared-LIBXSMM','KleidiAI-fixed','KleidiAI-elastic'):methods[n]='prepared' if n=='prepared-LIBXSMM' else peers[n]
xs=args.libxsmm_checkout.resolve();env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',OMP_THREAD_LIMIT='1',VECLIB_MAXIMUM_THREADS='1')
cases=list(csv.DictReader((h/'panel.csv').open()))
for label,source in [('prefaulted','worker.c'),('wired','wired.c')]:
 if args.only and args.only!=label:continue
 d=out/label;d.mkdir();exe=d/'worker';run('cc','-O3','-std=c11','-I'+str(h),'-I'+str(xs/'include'),h/source,xs/'lib/libxsmm.a','-o',exe)
 for fn in ('samples','pool','accuracy','faults'):
  orig=h/label/(fn+'.csv')
  if orig.is_file():(d/(fn+'.csv')).write_text(next(orig.open()).strip()+'\n')
 shutil.copyfile(h/'panel.csv',d/'panel.csv')
 with (d/'run.log').open('w') as log:
  for block,seed in enumerate((1088945771,1870216639),1):
   for i in (range(len(cases)) if block==1 else reversed(range(len(cases)))):
    q=cases[i];run(exe,q['family'],*[q[k] for k in ('m','n','k','pad')],str(seed),str(block),str(i),d,*[o for kv in methods.items() for o in kv],env=env,stdout=log,stderr=log)
 run(sys.executable,h/'score.py',d,'--output',d/'scores')
