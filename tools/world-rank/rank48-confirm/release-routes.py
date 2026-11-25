"""Check the normal linked release alongside an untimed vendor-call witness."""
import csv
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'signed'))
from routes import read_output
panel,build,out=map(Path,sys.argv[1:]);rows=[];cpu=min(os.sched_getaffinity(0))
for r in csv.DictReader(panel.open()):
    m,n,k,pad=shape=tuple(int(r[f]) for f in ('m','n','k','pad'))
    ordinary,_=read_output(build/'release.worker',shape)
    traced,leaf=read_output(build/'release.trace.worker',shape)
    actual=dict(checked=m*n,output_exact=1,inputs_unchanged=1,guards=1,threads=1)
    expected=dict(core_m=m//64*64,core_n=n//64*64,core_k=k//32*32,calls=49+(m%64!=0)+(n%64!=0)+(k%32!=0),core_calls=49,core_overwrites=49,positive_updates=0,negative_updates=0,depth_updates=int(k%32!=0),right_strips=int(n%64!=0),bottom_strips=int(m%64!=0),quadrants_ok=1,edges_ok=1,provider_ok=1)
    if ordinary!=actual or traced!=actual or leaf!=expected:sys.exit(f'failed real release route {r}: {ordinary} {traced} {leaf}')
    rows.append(dict(**r,cpu=cpu,**leaf,**ordinary,passed=1))
with out.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
