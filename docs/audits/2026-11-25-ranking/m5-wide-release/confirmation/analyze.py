import csv,statistics as st,json
from pathlib import Path
h=Path(__file__).resolve().parent
shapes=list(csv.DictReader((h/'panel.csv').open()));rows=list(csv.DictReader((h/'times.csv').open()))
result=[];summaries={}
for method in ('candidate','predecessor'):
 fresh=[]
 for s in shapes:
  t=[x for x in rows if all(x[q]==s[q] for q in ('m','n','k','pad'))]
  by={name:{(int(x['block']),int(x['round'])):x for x in t if x['method']==name} for name in set(x['method'] for x in t)}
  assert all(len(z)==6 for z in by.values())
  def ratio(name,block=None):
   return st.median(float(x['warm_seconds'])/float(by[method][i]['warm_seconds']) for i,x in by[name].items() if block is None or i[0]==block)
  ext=set(by)-{'candidate','predecessor','release'}
  lead=min(ext,key=lambda name:st.median(float(x['warm_seconds']) for x in by[name].values()))
  d=dict(s,method=method,leading_peer=lead,vs_release=ratio('release'),vs_leading=ratio(lead),vs_predecessor=ratio('predecessor'),vs_Accelerate=ratio('Accelerate'),vs_prepared=ratio('prepared-LIBXSMM'),
     release_seed1=ratio('release',1),release_seed2=ratio('release',2))
  result.append(d)
  if s['family'].startswith('fresh-'):fresh.append(d)
 gm=lambda q:st.geometric_mean(x[q] for x in fresh)
 summary={'timings':len(rows),'max_abs':max(float(x['max_abs_error']) for x in rows),'max_threads':max(int(x['threads_after']) for x in rows),'fresh':len(fresh),'gm_release':gm('vs_release'),'gm_peer':gm('vs_leading'),'gm_predecessor':gm('vs_predecessor'),'gm_apple':gm('vs_Accelerate'),'gm_jit':gm('vs_prepared'),'wins_both_release':sum(min(d['release_seed1'],d['release_seed2'])>1 for d in fresh),'release_min':min(d['vs_release'] for d in fresh),'peer_min':min(d['vs_leading'] for d in fresh),'predecessor_min':min(d['vs_predecessor'] for d in fresh),'leading_wins':sum(d['vs_leading']>=1.01 for d in fresh)}
 summary['opt_in']=summary['gm_release']>=1.50 and summary['wins_both_release']>=7 and summary['release_min']>=.95 and summary['peer_min']>=.85 and (method=='predecessor' or (summary['gm_predecessor']>=1.02 and summary['predecessor_min']>=.90))
 summary['field']=summary['gm_peer']>=1.02 and summary['leading_wins']>=6 and summary['peer_min']>=.95
 summaries[method]=summary
with (h/'per-shape.csv').open('w') as f:
 w=csv.DictWriter(f,result[0]);w.writeheader();w.writerows(result)
(h/'results.json').write_text(json.dumps(summaries,indent=2)+'\n')
for d in result:print(d['family'],d['method'],d['leading_peer'],'release %.2f field %.3f prev %.2f'%(d['vs_release'],d['vs_leading'],d['vs_predecessor']))
print(summaries)
