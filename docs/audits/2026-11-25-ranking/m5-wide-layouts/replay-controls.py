"""Native-only replay of both same-function controls using matched public libraries."""
import argparse,csv,os,subprocess as sp
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
for name in ('matched_build','libxsmm','new_directory'):p.add_argument(name,type=Path)
a=p.parse_args();kit=Path(__file__).resolve().parent;out=a.new_directory.resolve();out.mkdir(parents=True,exist_ok=False)
b=a.matched_build.resolve();xs=a.libxsmm.resolve()
libs=[v for pair in [('A',b/'incumbent.dylib'),('B',b/'incumbent.dylib'),('D',b/'candidate.dylib')] for v in pair]
with (kit/'panel/panel.csv').open() as f:panel=list(csv.DictReader(f))
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',OMP_THREAD_LIMIT='1',VECLIB_MAXIMUM_THREADS='1')
for name in ('null','rapid'):
    exe=out/name
    sp.run(['cc','-std=c11','-O3','-I'+str(kit),'-I'+str(xs/'include'),kit/'matched'/(name+'.c'),xs/'lib/libxsmm.a','-o',exe],check=True)
    with (out/(name+'.csv')).open('w') as f,(out/(name+'.log')).open('w') as log,(kit/'matched'/(name+'.csv')).open() as original:
        csv.writer(f).writerow(next(csv.reader(original)));f.flush()
        for block,seed in enumerate((427297853,1670820329),1):
            for s in panel if block==1 else reversed(panel):
                sp.run([exe,*[s[k] for k in ('m','n','k','pad')],str(seed),str(block),*libs],stdout=f,stderr=log,env=env,check=True)
