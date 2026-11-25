"""Replay the rejected locality pilot on a supported Apple M5 with ordinary public builds."""
from pathlib import Path
import argparse,csv,subprocess as sp,shutil,sys
q=argparse.ArgumentParser(description=__doc__);q.add_argument('source_repo',type=Path);q.add_argument('libxsmm',type=Path);q.add_argument('new_output',type=Path);a=q.parse_args()
h=Path(__file__).resolve().parent;out=a.new_output.resolve();out.mkdir(exist_ok=False,parents=True);res=out/'results';res.mkdir();shutil.copyfile(h/'panel.csv',out/'panel.csv')
def run(*cmd,**kwargs):sp.run(cmd,check=True,**kwargs)
src=out/'source';run('git','clone','--quiet','--local','--no-checkout',a.source_repo.resolve(),src);run('git','-C',src,'checkout','--quiet','--detach','69b239d6ab9b9a86425ef6874634e3a1952a7f73')
libs={}
for label in ('incumbent','candidate'):
 if label=='candidate':run('git','-C',src,'apply',h/'slab.patch')
 build=out/(label+'-build');inst=out/(label+'-install')
 run('cmake','-S',src,'-B',build,'-DCMAKE_BUILD_TYPE=Release','-DCOB_STRASSEN_SME=ON','-DCOB_ENABLE_APPLE_MEDIUM_WIDE=ON')
 run('cmake','--build',build,'--target','cob_matmul','-j','2');run('cmake','--install',build,'--prefix',inst)
 libs[label]=out/(label+'.dylib');run('cc','-O3','-dynamiclib','-I'+str(inst/'include'),h/'adapter.c','-Wl,-force_load,'+str(inst/'lib/libcob_matmul.a'),'-o',libs[label])
xs=a.libxsmm.resolve();run('cc','-O3','-std=c11','-I'+str(h),'-I'+str(xs/'include'),h/'pilot.c',xs/'lib/libxsmm.a','-o',out/'pilot')
for n in ('samples','accuracy','pool','faults'):(res/(n+'.csv')).write_text(next((h/'results'/(n+'.csv')).open()).strip()+'\n')
rows=list(csv.DictReader((h/'panel.csv').open()))
for block,seed in enumerate((619325179,1639120403),1):
 for i in (range(len(rows)) if block==1 else reversed(range(len(rows)))):
  s=rows[i];run(out/'pilot',s['family'],*[s[k] for k in ('m','n','k','pad')],str(seed),str(block),str(i),res,'A',libs['incumbent'],'B',libs['incumbent'],'D',libs['candidate'])
run(sys.executable,h/'score.py',out)
