from pathlib import Path
from collections import Counter
import csv,statistics as st,math,json,sys
h=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
read=lambda n:list(csv.DictReader((h/n).open()))
panel=read('panel.csv');raw=read('warm.csv');first=read('first.csv')
methods={'A','B','D','Accelerate','OpenBLAS','ArmPL','MaxMulSK','MpGEMM','prepared-LIBXSMM','KleidiAI-fixed','KleidiAI-elastic'}
assert len(panel)==8 and len(raw)==11616 and len(first)==160
res=[]
for s in panel:
 same=lambda x:all(x[k]==s[k] for k in ('m','n','k','pad'));t=[x for x in raw if same(x)];f=[x for x in first if same(x)]
 assert {x['method'] for x in t}==methods and {x['method'] for x in f}==methods-{'B'}
 assert all(float(x['max_abs_error'])<=.002 and x['threads_before']==x['threads_after']=='1' for x in t)
 assert all(float(x['max_abs_error'])<=.002 and x['threads']=='1' for x in f)
 assert all(int(x['code_bytes'])>0 for x in f if x['method']=='prepared-LIBXSMM')
 for block in (1,2):
  obs=[x for x in t if int(x['block'])==block]
  positions=Counter((x['method'],x['position']) for x in obs);assert len(positions)==121 and set(positions.values())=={6}
  edges=Counter()
  for j in range(1,67):
   seq=[x['method'] for x in sorted((x for x in obs if int(x['round'])==j),key=lambda r:int(r['position']))];assert set(seq)==methods;edges.update(zip(seq,seq[1:]))
  assert len(edges)==110 and set(edges.values())=={6}
 w={k:{(int(x['block']),int(x['round'])):float(x['warm_seconds']) for x in t if x['method']==k} for k in methods}
 c={k:{int(x['seed']):float(x['first_seconds'])+float(x['jit_setup_seconds']) for x in f if x['method']==k} for k in methods-{'B'}}
 peers=methods-{'A','B','D'};old=peers-{'KleidiAI-fixed','KleidiAI-elastic'}
 fastest=min(peers,key=lambda k:(st.median(w[k].values()),k));oldfast=min(old,key=lambda k:(st.median(w[k].values()),k));coldfast=min(peers,key=lambda k:(st.median(c[k].values()),k))
 rat=lambda source,k:st.median(source[k][i]/v for i,v in source['D'].items())
 incumbent=st.median(math.sqrt(w['A'][i]*w['B'][i])/v for i,v in w['D'].items())
 res.append(dict(s,AA=st.median(w['A'][i]/v for i,v in w['B'].items()),incumbent=incumbent,fastest=fastest,old_fastest=oldfast,cold_fastest=coldfast,field=rat(w,fastest),previous_field=rat(w,oldfast),warm_fixed=rat(w,'KleidiAI-fixed'),warm_elastic=rat(w,'KleidiAI-elastic'),cold_field=rat(c,coldfast),cold_fixed=rat(c,'KleidiAI-fixed'),cold_elastic=rat(c,'KleidiAI-elastic'),warm_cob_us=1e6*st.median(w['D'].values()),warm_fixed_us=1e6*st.median(w['KleidiAI-fixed'].values()),warm_elastic_us=1e6*st.median(w['KleidiAI-elastic'].values())))
with (h/'scores.csv').open('w') as f:
 w=csv.DictWriter(f,res[0]);w.writeheader();w.writerows(res)
v={'gm':{k:st.geometric_mean(x[k] for x in res) for k in ('incumbent','field','previous_field','warm_fixed','warm_elastic','cold_field','cold_fixed','cold_elastic')},'wins':{k:sum(x[k]>1 for x in res) for k in ('field','previous_field','warm_fixed','warm_elastic')},'wins_1pct':sum(x['field']>1.01 for x in res),'fastest_changes':sum(x['fastest']!=x['old_fastest'] for x in res),'AA_range':[min(x['AA'] for x in res),max(x['AA'] for x in res)],'warm_obs':len(raw),'first_obs':len(first),'cells':sum(int(x['checked_cells']) for x in raw+first)+sum(int(x['m'])*int(x['n']) for x in raw if x['round']=='1'),'max_abs':max(float(x['max_abs_error']) for x in raw+first),'max_by_kleidi':{k:max(float(x['max_abs_error']) for x in raw+first if x['method']==k) for k in ('KleidiAI-fixed','KleidiAI-elastic')}}
(h/'scores.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
for x in res:print(x['family'],x['fastest'],round(x['field'],3),round(x['warm_fixed'],3),round(x['warm_elastic'],3))
