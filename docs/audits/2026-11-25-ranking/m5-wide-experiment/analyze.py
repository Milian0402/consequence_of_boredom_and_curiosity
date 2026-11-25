import csv,statistics as st,json
from pathlib import Path
h=Path(__file__).resolve().parent
shapes=list(csv.DictReader((h/'panel.csv').open()));rows=list(csv.DictReader((h/'times.csv').open()))
result=[]
for s in shapes:
 t=[x for x in rows if all(x[q]==s[q] for q in ('m','n','k','pad'))]
 by={name:{(int(x['block']),int(x['round'])):x for x in t if x['method']==name} for name in set(x['method'] for x in t)}
 assert all(len(z)==6 for z in by.values())
 def ratio(name,cols='warm_seconds',block=None):
  return st.median(float(x[cols])/float(by['candidate'][i][cols]) for i,x in by[name].items() if block is None or i[0]==block)
 ext=set(by)-{'candidate','release'}
 lead=min(ext,key=lambda name:st.median(float(x['warm_seconds']) for x in by[name].values()))
 d=dict(s,leading_peer=lead,vs_release=ratio('release'),vs_leading=ratio(lead),vs_Accelerate=ratio('Accelerate'),vs_prepared=ratio('prepared-LIBXSMM') if 'prepared-LIBXSMM' in by else None,
    leading_us=1e6*st.median(float(x['warm_seconds']) for x in by[lead].values()),candidate_us=1e6*st.median(float(x['warm_seconds']) for x in by['candidate'].values()),
    release_us=1e6*st.median(float(x['warm_seconds']) for x in by['release'].values()),first_vs_release=ratio('release','first_seconds'),release_seed1=ratio('release',block=1),release_seed2=ratio('release',block=2))
 result.append(d)
with (h/'per-shape.csv').open('w') as f:
 w=csv.DictWriter(f,result[0]);w.writeheader();w.writerows(result)
fresh=[d for d in result if d['family'].startswith('fresh-')]
summary={'timings':len(rows),'max_abs':max(float(x['max_abs_error']) for x in rows),'max_threads':max(int(x['threads_after']) for x in rows),'min_threads':min(int(x['threads_before']) for x in rows),'fresh':len(fresh),'gm_release':st.geometric_mean(d['vs_release'] for d in fresh),'gm_peer':st.geometric_mean(d['vs_leading'] for d in fresh),'gm_apple':st.geometric_mean(d['vs_Accelerate'] for d in fresh),'gm_jit':st.geometric_mean(d['vs_prepared'] for d in fresh if d['vs_prepared']),'seed_wins':[sum(d['release_seed1']>1 for d in fresh),sum(d['release_seed2']>1 for d in fresh)],'release_min':min(d['vs_release'] for d in fresh),'leading_wins':sum(d['vs_leading']>=1.01 for d in fresh),'checks':sum(int(x['checked_cells']) for x in rows)+sum(int(x['checked_cells']) for x in rows if x['round']=='1')}
summary['qualifies']=summary['gm_release']>=1.10 and min(summary['seed_wins'])>=6 and summary['release_min']>=.95 and summary['leading_wins']>=4
(h/'results.json').write_text(json.dumps(summary,indent=2)+'\n')
for d in result:print(d['family'],d['leading_peer'],'release %.2f field %.2f apple %.2f'%(d['vs_release'],d['vs_leading'],d['vs_Accelerate']))
print(summary)
