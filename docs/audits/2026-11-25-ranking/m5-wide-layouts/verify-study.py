"""Replay frozen scores on copies of the raw evidence. No native GEMM is executed."""
from pathlib import Path
import csv
import hashlib
import json
import math
import shutil
import statistics as st
import subprocess
import sys
import tempfile

root=Path(sys.argv[1] if len(sys.argv)>1 else '.').resolve()
def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def read_csv(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))
def verify_manifest():
    entries={}
    for line in (root/'SHA256SUMS').read_text().splitlines():
        expected,name=line.split('  ',1)
        assert name not in entries
        p=root/name
        assert p.is_file() and digest(p)==expected,name
        entries[name]=expected
    assert set(entries)=={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p!=root/'SHA256SUMS'}
    legacy={}
    for line in (root/'freeze.sha256').read_text().splitlines():
        expected,name=line.split('  ',1)
        if name in ('bench','direct.dylib'):
            legacy[name]=expected
        else:
            assert name in entries and entries[name]==expected,name
    assert set(legacy)=={'bench','direct.dylib'}
    receipt=json.loads((root/'recorded-binaries.json').read_text())
    assert all(receipt[name]==expected for name,expected in legacy.items())
    for label in ('candidate','incumbent'):
        assert receipt['matched-'+label]==json.loads((root/'matched/toolchain.json').read_text())[label]['dylib_sha']
    assert len(entries)>=50
    return len(entries)
def train():
    rows=read_csv(root/'train.csv')
    geom=[(24,1280,131),(39,2304,160),(55,1536,384)]
    names={'incumbent','direct','vertical','superpanel','Accelerate','prepared-LIBXSMM','MpGEMM'}
    assert len(rows)==len(geom)*2*3*len(names)
    scores={}
    for name in ('direct','vertical','superpanel'):
        medians=[]
        for dims in geom:
            section=[x for x in rows if tuple(int(x[z]) for z in ('m','n','k'))==dims]
            assert len(section)==2*3*len(names) and {x['method'] for x in section}==names
            for x in section:
                assert x['pad']=='0' and int(x['checked_cells'])==dims[0]*dims[1]
                assert x['threads_before']==x['threads_after']=='1'
                assert 0<=float(x['max_abs_error'])<=.002 and math.isfinite(float(x['warm_seconds'])) and float(x['warm_seconds'])>0
            trial=lambda method:{(int(x['seed']),int(x['round'])):float(x['warm_seconds']) for x in section if x['method']==method}
            baseline,other=trial('incumbent'),trial(name)
            assert set(baseline)==set(other)=={(s,r) for s in (675893197,1382786813) for r in (1,2,3)}
            medians.append(st.median(baseline[i]/v for i,v in other.items()))
        scores[name]={'warm':medians,'gm':st.geometric_mean(medians)}
    stored=json.loads((root/'training-scores.json').read_text())
    for n,v in scores.items():
        assert all(math.isclose(a,b,rel_tol=1e-13) for a,b in zip(v['warm'],stored[n]['warm'],strict=True))
        assert math.isclose(v['gm'],stored[n]['gm'],rel_tol=1e-13)
    assert max(scores,key=lambda n:scores[n]['gm'])==(root/'selection.txt').read_text().strip()=='direct'
def score(script,source,target,outnames):
    for name in outnames:
        (target/name).unlink(missing_ok=True)
    subprocess.run([sys.executable,'-B',script,target],check=True,stdout=subprocess.DEVNULL)
    for n in outnames:
        if n.endswith('.json'):
            assert json.loads((target/n).read_text())==json.loads((source/n).read_text()),n
        else:
            assert read_csv(target/n)==read_csv(source/n),n

n=verify_manifest()
train()
with tempfile.TemporaryDirectory() as tmp:
    d=Path(tmp)/'broad';shutil.copytree(root,d)
    for local in ('.','matched/reproduction'):
        r=root/local
        score(root/'score.py',r,d/local,['scores.csv','scores.json'])
    for name in ('null','rapid'):
        score(root/'matched'/('score-'+name+'.py'),root/'matched',d/'matched',[name+'-shapes.csv',name+'-summary.json'])
    p=d/'verify';new=p/'regenerated.json'
    subprocess.run([sys.executable,'-B',p/'summarize.py',p/'checks.csv',p/'fallbacks.csv',new],check=True,stdout=subprocess.DEVNULL)
    assert json.loads(new.read_text())==json.loads((p/'summary.json').read_text())
print(f'{n} files verified. Training, both direct fields, both A/A controls, and protected-memory results reproduced.')
