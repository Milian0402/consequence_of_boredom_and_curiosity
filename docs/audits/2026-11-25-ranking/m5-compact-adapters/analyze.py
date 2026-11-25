from pathlib import Path
import csv,statistics as st,json
h=Path(__file__).resolve().parent
read=lambda n:list(csv.DictReader((h/n).open()))
warm=read('times.csv');cold=read('first.csv');panel=read('panel.csv')
assert len(warm)==180 and len(cold)==60 and len(panel)==6
checked=sum(int(r['checked_cells']) for r in warm)+sum(int(r['checked_cells']) for r in warm if r['round']=='1')+sum(int(r['checked_cells']) for r in cold)
for r in warm:
 assert int(r['threads_before'])==int(r['threads_after'])==1 and float(r['max_abs_error'])<.002
for r in cold:assert int(r['threads'])==1 and float(r['max_abs_error'])<.002
result=[];by={}
methods=['Accelerate','prepared-LIBXSMM','MaxMulSK-adapted','MpGEMM-adapted']
for s in panel:
 same=lambda x:all(x[k]==s[k] for k in ('m','n','k','pad'))
 wr={k:{(int(x['block']),int(x['round'])):float(x['warm_seconds']) for x in warm if x['method']==k and same(x)} for k in ['candidate',*methods]}
 cr={k:{int(x['seed']):float(x['first_seconds'])+float(x['jit_setup_seconds']) for x in cold if x['method']==k and same(x)} for k in wr}
 assert all(len(v)==6 for v in wr.values()) and all(len(v)==2 for v in cr.values())
 fastest=min(methods,key=lambda k:st.median(wr[k].values()));cfast=min(methods,key=lambda k:st.median(cr[k].values()))
 row=dict(s,fastest=fastest,first_fastest_name=cfast)
 for k in [*methods,'fastest']:
  row['warm_'+k]=st.median(v/wr['candidate'][i] for i,v in wr[fastest if k=='fastest' else k].items())
  row['cold_'+k]=st.median(v/cr['candidate'][i] for i,v in cr[cfast if k=='fastest' else k].items())
 result.append(row)
with (h/'scores.csv').open('w') as f:
 w=csv.DictWriter(f,result[0]);w.writeheader();w.writerows(result)
guards={}
for mode in ('candidate','MpGEMM-adapted','MaxMulSK-adapted'):
 rows=read('guards-'+mode+'.csv');assert len(rows)==24 and all(x['inputs_readonly']==x['prefix_guard']=='1' and float(x['max_abs_error'])<.002 for x in rows)
 guards[mode]=dict(rows=len(rows),cells=sum(int(x['checked']) for x in rows),largest=max(float(x['max_abs_error']) for x in rows))
summary={k:{'gm':st.geometric_mean(r[k] for r in result),'min':min(r[k] for r in result),'wins':sum(r[k]>1 for r in result)} for k in result[0] if k.startswith(('warm_','cold_'))}
d=dict(summary=summary,timings=len(warm),cold_processes=len(cold),checked=checked,guards=guards,largest=max(float(x['max_abs_error']) for x in warm+cold));(h/'scores.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d))
