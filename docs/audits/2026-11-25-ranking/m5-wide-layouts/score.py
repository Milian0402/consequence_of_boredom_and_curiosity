from pathlib import Path
import csv,json,statistics as stats,sys,hashlib
h=Path(sys.argv[1] if len(sys.argv)>1 else Path(__file__).resolve().parent)
read=lambda name:list(csv.DictReader((h/name).open()))
panel=read('panel/panel.csv');warm=read('holdout.csv');cold=read('first.csv')
assert len(warm)==880 and len(cold)==220 and len(panel)==14
emitted=[]
for s in panel:
 select=lambda x:all(s[k]==x[k] for k in ('m','n','k','pad'));wr=[x for x in warm if select(x)];cr=[x for x in cold if select(x)];names=set(x['method'] for x in wr)
 assert names==({'candidate','incumbent','Accelerate','OpenBLAS','ArmPL','prepared-LIBXSMM'} if s['pad']!='0' else {'candidate','incumbent','Accelerate','OpenBLAS','ArmPL','prepared-LIBXSMM','MpGEMM','MaxMulSK'})
 w={n:{(int(x['seed']),int(x['round'])):float(x['warm_seconds']) for x in wr if x['method']==n} for n in names}
 c={n:{int(x['seed']):float(x['first_seconds'])+float(x['jit_setup_seconds']) for x in cr if x['method']==n} for n in names}
 assert all(set(v)=={(s,b) for s in (427297853,1670820329) for b in range(1,5)} for v in w.values())
 assert all(set(v)=={427297853,1670820329} for v in c.values())
 assert all(int(x['threads_before'])==int(x['threads_after'])==1 and float(x['max_abs_error'])<=.002 and int(x['checked_cells'])==int(s['n'])*int(s['m']) for x in wr)
 assert all(int(x['threads'])==1 and float(x['max_abs_error'])<=.002 and int(x['checked_cells'])==int(s['n'])*int(s['m']) for x in cr)
 assert all(int(x['code_bytes'])>0 for x in cr+wr if x['method']=='prepared-LIBXSMM')
 peers=names-{'candidate','incumbent'}
 fastest=min(peers,key=lambda n:stats.median(w[n].values()));fc=min(peers,key=lambda n:stats.median(c[n].values()))
 def warmratio(n,seed=None):return stats.median(w[n][i]/v for i,v in w['candidate'].items() if seed is None or i[0]==seed)
 def coldratio(n):return stats.median(c[n][i]/v for i,v in c['candidate'].items())
 emitted.append(dict(s,fastest=fastest,fastest_cold=fc,warm_us=stats.median(w['candidate'].values())*1e6,incumbent_us=stats.median(w['incumbent'].values())*1e6,vs_incumbent=warmratio('incumbent'),vs_field=warmratio(fastest),vs_accelerate=warmratio('Accelerate'),vs_prepared=warmratio('prepared-LIBXSMM'),vs_mpgemm=warmratio('MpGEMM') if 'MpGEMM' in names else '',vs_maxmulsk=warmratio('MaxMulSK') if 'MaxMulSK' in names else '',seed1_incumbent=warmratio('incumbent',427297853),seed2_incumbent=warmratio('incumbent',1670820329),incumbent_paired_wins=sum(w['incumbent'][i]>t for i,t in w['candidate'].items()),field_paired_wins=sum(w[fastest][i]>t for i,t in w['candidate'].items()),cold_incumbent=coldratio('incumbent'),cold_field=coldratio(fc)))
fresh=[x for x in emitted if x['family'].startswith('fresh-')];assert len(fresh)==8
gm=lambda key:stats.geometric_mean(x[key] for x in fresh)
out={'fresh_gm':{k:gm(k) for k in ('vs_incumbent','vs_field','vs_accelerate','vs_prepared','vs_mpgemm','vs_maxmulsk','cold_incumbent','cold_field')},'seed_wins':[sum(x[k]>1 for x in fresh) for k in ('seed1_incumbent','seed2_incumbent')],'min_incumbent':min(x['vs_incumbent'] for x in fresh),'min_field':min(x['vs_field'] for x in fresh),'field_wins_onepct':sum(x['vs_field']>1.01 for x in fresh),'field_paired_wins':sum(x['field_paired_wins'] for x in fresh),'incumbent_paired_wins':sum(x['incumbent_paired_wins'] for x in fresh),'warm_obs':len(warm),'first_processes':len(cold),'cells_warm_and_first':sum(int(x['checked_cells']) for x in warm)+sum(int(x['checked_cells']) for x in warm if x['round']=='1')+sum(int(x['checked_cells']) for x in cold),'max_abs_fp64':max(float(x['max_abs_error']) for x in warm+cold)}
out['gate']=out['fresh_gm']['vs_incumbent']>=1.04 and min(out['seed_wins'])>=6 and out['min_incumbent']>=.95 and out['fresh_gm']['vs_field']>=1.01 and out['field_wins_onepct']>=5 and out['min_field']>=.92
with (h/'scores.csv').open('w') as f:
 w=csv.DictWriter(f,emitted[0]);w.writeheader();w.writerows(emitted)
(h/'scores.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
for x in emitted: print(x['family'],x['fastest'],round(x['vs_incumbent'],3),round(x['vs_field'],3))
