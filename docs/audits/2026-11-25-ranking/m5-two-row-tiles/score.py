from pathlib import Path
import csv,itertools,json,math,statistics as st,sys
h=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
read=lambda n:list(csv.DictReader((h/n).open()))
cases=read('panel/panel.csv');data=read('warm.csv');cold=read('first.csv');out=[]
for s in cases:
 same=lambda x:all(x[k]==s[k] for k in ('m','n','k','pad'));r=[x for x in data if same(x)];c=[x for x in cold if same(x)];methods=set(x['method'] for x in r);n=len(methods);peers=methods-{'A','B','D'};assert n==(7 if s['pad']!='0' else 9)
 assert len(r)==2*6*n*n and len(c)==2*(n-1)
 for block in (1,2):
  for name in methods:
   t=[v for v in r if int(v['block'])==block and v['method']==name]
   assert sorted(int(v['position']) for v in t)==sorted(list(range(1,n+1))*6)
  orders=[tuple(x['method'] for x in sorted([a for a in r if int(a['block'])==block and int(a['round'])==j],key=lambda a:int(a['position']))) for j in range(1,6*n+1)]
  from collections import Counter
  edges=Counter(edge for order in orders for edge in zip(order,order[1:]))
  assert len(edges)==n*(n-1) and set(edges.values())=={6}
 assert all(float(x['max_abs_error'])<=.002 and x['threads_before']==x['threads_after']=='1' for x in r)
 assert all(float(x['max_abs_error'])<=.002 and int(x['threads'])==1 for x in c)
 assert all(int(x['code_bytes'])>0 for x in c if x['method']=='prepared-LIBXSMM')
 def values(col):return {k:{(int(x['block']),int(x['round'])):float(x[col]) for x in r if x['method']==k} for k in methods}
 w=values('warm_seconds');u=values('caller_seconds')
 times={k:{int(x['seed']):float(x['first_seconds'])+float(x['jit_setup_seconds']) for x in c if x['method']==k} for k in methods-{'B'}}
 fast=min(peers,key=lambda name:st.median(w[name].values()));fc=min(peers,key=lambda name:st.median(times[name].values()))
 def corr(clk,b=None):return st.median(math.sqrt(clk['A'][i]*clk['B'][i])/v for i,v in clk['D'].items() if b is None or i[0]==b)
 single=lambda name:st.median(w[name][i]/v for i,v in w['D'].items())
 col=lambda name:st.median(times[name][i]/v for i,v in times['D'].items())
 out.append(dict(s,fast=fast,cold_fast=fc,AA=st.median(w['A'][i]/v for i,v in w['B'].items()),corrected=corr(w),cpu_corrected=corr(u),seed1=corr(w,1),seed2=corr(w,2),field=single(fast),accelerate=single('Accelerate'),jit=single('prepared-LIBXSMM'),mpgemm=single('MpGEMM') if 'MpGEMM' in peers else '',maxmulsk=single('MaxMulSK') if 'MaxMulSK' in peers else '',cold_inc=col('A'),cold_field=col(fc),warm_us=1e6*st.median(w['D'].values()),external_us=1e6*st.median(w[fast].values())))
with (h/'scores.csv').open('w') as f:
 w=csv.DictWriter(f,out[0]);w.writeheader();w.writerows(out)
fresh=[x for x in out if x['family'].startswith('fresh-')];assert len(fresh)==8
gm=lambda k:st.geometric_mean(x[k] for x in fresh)
j={'fresh_gm':{k:gm(k) for k in ('corrected','cpu_corrected','field','accelerate','jit','mpgemm','maxmulsk','cold_inc','cold_field')},'seed_wins':[sum(x[k]>1 for x in fresh) for k in ('seed1','seed2')],'min_inc':min(x['corrected'] for x in fresh),'min_field':min(x['field'] for x in fresh),'field_wins_1pct':sum(x['field']>1.01 for x in fresh),'null_range':[min(x['AA'] for x in out),max(x['AA'] for x in out)],'warm_samples':len(data),'first_processes':len(cold),'max_abs':max(float(x['max_abs_error']) for x in data+cold),'fp64_cells':sum(int(x['checked_cells']) for x in data+cold)+sum(int(x['m'])*int(x['n']) for x in data if x['round']=='1')}
j['gate']=gm('corrected')>=1.04 and min(j['seed_wins'])>=6 and j['min_inc']>=.975 and gm('field')>=1.01 and j['field_wins_1pct']>=5 and j['min_field']>=.92 and max(abs(v-1) for v in j['null_range'])<=.03
(h/'scores.json').write_text(json.dumps(j,indent=2)+'\n');print(json.dumps(j))
for x in out:print(x['family'],x['fast'],round(x['corrected'],3),round(x['field'],3),round(x['AA'],3))
