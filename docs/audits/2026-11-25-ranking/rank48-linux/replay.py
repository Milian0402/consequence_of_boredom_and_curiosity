"""Run from any directory: python3 replay.py (Python 3.9+)."""
import csv,hashlib,json
from pathlib import Path
from tensor import verify
from verify import audit,need
base=Path(__file__).resolve().parent
meta=json.loads((base/'provenance.json').read_text());need(verify(base/'schedules')=={48:4096,49:4096},'tensor certificate')
with (base/'summary.csv').open() as f:reference={(r['arch'],r['family']):r for r in csv.DictReader(f)}
for host in meta['hosts']:
 arch=host['arch'];root=base/arch;digest=host['github_zip_sha256'];receipt=json.loads((root/'github-artifact.json').read_text());need(receipt['digest']=='sha256:'+digest,'official digest')
 for name,expected in host['member_sha256'].items():
  f=root/Path(name).name
  if f.exists():need(hashlib.sha256(f.read_bytes()).hexdigest()==expected,'original member differs '+name)
 scores,_,_,result=audit(root,arch)
 for r in scores:need({k:str(v) for k,v in r.items()}==reference[(arch,r['family'])],'published summary differs')
 print(arch,result['model'],result['fp64_fixtures'],'full fixtures;',result['rank48_median_wins_vs_49'],'qualifying median wins; confirmation gate',result['gate'])
