"""Independently compare actual public dispatch to true ISA-specific tile counts."""
import csv
import os
from pathlib import Path
import subprocess
import sys

panel,build,out=(Path(p).resolve() for p in sys.argv[1:])
cpu=min(os.sched_getaffinity(0));os.sched_setaffinity(0,{cpu})
dims=('m','n','k','pad')
cols=['implementation','family',*dims,'isa','selected','mr','nr','packs','tiles','old_medium','blocked',
      'expected_old_medium','expected_blocked','linked_checked','trace_checked','inputs_unchanged','guards','threads','cpu','passed','error']
def read(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))
def fixture(worker,shape):
    p=subprocess.run([worker,'fixture',*map(str,shape)],check=True,capture_output=True,text=True)
    return {l.split(',')[0]:{s.split('=')[0]:int(s.split('=')[1]) for s in l.split(',')[1:]} for l in p.stdout.splitlines() if l.startswith(('FIXTURE,','STREAM,'))}
failed=0
with (out/'routes.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
    for profile in read(out/'builds.csv'):
        if profile['status']!='qualified':continue
        name,isa=profile['implementation'],profile['isa']
        for case in read(panel):
            m,n,k,pad=(int(case[d]) for d in dims);shape=m,n,k,pad
            row=dict(implementation=name,**case,isa=isa,cpu=cpu,passed=0,error='')
            try:
                ordinary=fixture(build/(name+'.worker'),shape)['FIXTURE']
                traced=fixture(build/(name+'.trace.worker'),shape)
                exact=dict(checked=m*n,output_exact=1,inputs_unchanged=1,guards=1,threads=1)
                if ordinary!=exact or traced['FIXTURE']!=exact:raise ValueError('ordinary or traced public exact/input/guard/thread fixture failed')
                eligible=24<=m<=160 and 1024<=n<=4096 and 96<=k<=384 and n>=8*m and m*k<=49152
                hit=int(profile['variant']=='candidate' and eligible)
                mr,nr=((6,64) if isa=='avx512' else (3,32)) if hit else (0,0)
                old=int(not hit and m>=64 and k>=128)
                blocked=int(not hit and not old)
                expected=dict(selected=hit,mr=mr,nr=nr,packs=(n+nr-1)//nr if hit else 0,
                              tiles=((n+nr-1)//nr)*((m+mr-1)//mr) if hit else 0,old_medium=old,blocked=blocked)
                row.update(traced['STREAM'],expected_old_medium=old,expected_blocked=blocked,linked_checked=ordinary['checked'],
                           trace_checked=traced['FIXTURE']['checked'],inputs_unchanged=1,guards=1,threads=1)
                if traced['STREAM']!=expected:raise ValueError(f"actual {traced['STREAM']} vs expected {expected}")
                row['passed']=1
            except (KeyError,OSError,ValueError,subprocess.SubprocessError) as exc:
                row['error']=str(exc);failed+=1
            w.writerow(row)
if failed:sys.exit(f'{failed} public or trace fixtures failed')
