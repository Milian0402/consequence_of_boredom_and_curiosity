"""Verify the native records and reproduce the headline ratios without third-party packages."""
from pathlib import Path
import csv, json, math, statistics as st, sys
root=Path(sys.argv[1] if len(sys.argv)>1 else '.').resolve()
def records(p):
    with (root/p).open() as f:return list(csv.DictReader(f))
def geometry(row):return tuple(int(row[q]) for q in ('m','n','k','pad'))
panel=records('panel.csv');warm=records('production-times.csv');first=records('production-first.csv')
assert len(panel)==14 and len(warm)==624 and len(first)==208
assert {int(r['seed']) for r in warm}=={318741563,1557601921}
def summarize():
    rows=[]
    for s in panel:
        t=[r for r in warm if geometry(r)==geometry(s)]
        c=[r for r in first if geometry(r)==geometry(s)]
        methods={'release','candidate','option-off','Accelerate','ArmPL','OpenBLAS','prepared-LIBXSMM'}
        if int(s['pad'])==0: methods|={'MaxMulSK','MpGEMM'}
        by={name:{(int(x['block']),int(x['round'])):x for x in t if x['method']==name} for name in methods}
        cf={n:{int(x['seed']):x for x in c if x['method']==n} for n in methods}
        assert all(set(b)=={(s,r) for s in (1,2) for r in (1,2,3)} for b in by.values())
        assert all(set(b)=={318741563,1557601921} for b in cf.values())
        for x in t:
            assert 0<=float(x['max_abs_error'])<=.002 and int(x['threads_after'])==int(x['threads_before'])==1
            assert int(x['checked_cells'])==int(s['m'])*int(s['n']) and float(x['warm_seconds'])>0
            assert (int(x['code_bytes'])>0)==(x['method']=='prepared-LIBXSMM')
        for x in c:
            assert int(x['threads'])==1 and float(x['first_seconds'])>0 and float(x['max_abs_error'])<=.002
        ext=methods-{'release','candidate','option-off'}
        fastest=min(ext,key=lambda n:st.median(float(x['warm_seconds']) for x in by[n].values()))
        def ratio(name,block=None):return st.median(float(x['warm_seconds'])/float(by['candidate'][i]['warm_seconds']) for i,x in by[name].items() if block is None or i[0]==block)
        def cold(name):return st.median((float(x['first_seconds'])+float(x['jit_setup_seconds']))/float(cf['candidate'][i]['first_seconds']) for i,x in cf[name].items())
        cold_leader=min(ext,key=lambda n:st.median(float(x['first_seconds'])+float(x['jit_setup_seconds']) for x in cf[n].values()))
        rows.append(dict(s,fastest=fastest,release=ratio('release'),option_off=ratio('option-off'),peer=ratio(fastest),accelerate=ratio('Accelerate'),jit=ratio('prepared-LIBXSMM'),release_seed1=ratio('release',1),release_seed2=ratio('release',2),cold_peer_name=cold_leader,cold_release=cold('release'),cold_peer=cold(cold_leader),cold_on_us=1e6*st.median(float(x['first_seconds']) for x in cf['candidate'].values()),cold_release_us=1e6*st.median(float(x['first_seconds']) for x in cf['release'].values()),warm_on_us=1e6*st.median(float(x['warm_seconds']) for x in by['candidate'].values())))
    return rows
result=summarize(); fresh=[r for r in result if r['family'].startswith('fresh-')]
assert len(fresh)==9
gm=lambda field:st.geometric_mean(x[field] for x in fresh)
summary={'sample_count':len(warm),'cold_processes':len(first),'fresh_shape_count':len(fresh),'fresh_release_gm':gm('release'),'fresh_option_off_gm':gm('option_off'),'fresh_peer_gm':gm('peer'),'fresh_accelerate_gm':gm('accelerate'),'fresh_prepared_gm':gm('jit'),'fresh_min_release':min(x['release'] for x in fresh),'fresh_min_peer':min(x['peer'] for x in fresh),'fresh_both_seed_release_wins':sum(min(x['release_seed1'],x['release_seed2'])>1 for x in fresh),'fresh_peer_wins_1pct':sum(x['peer']>=1.01 for x in fresh),'cold_first_release_gm':gm('cold_release'),'cold_first_peer_gm':gm('cold_peer'),'total_warm_fp64_cells':sum(int(x['checked_cells']) for x in warm)+sum(int(x['checked_cells']) for x in warm if x['round']=='1'),'total_first_process_fp64_cells':sum(int(x['checked_cells']) for x in first),'normalized_warm_max_abs':max(float(x['max_abs_error']) for x in warm)}
summary['opt_in_qualified']=summary['fresh_release_gm']>=1.5 and summary['fresh_min_release']>=.95 and summary['fresh_both_seed_release_wins']>=7 and summary['fresh_min_peer']>=.85
summary['external_qualified']=summary['fresh_peer_gm']>=1.02 and summary['fresh_peer_wins_1pct']>=6 and summary['fresh_min_peer']>=.95
stress=records('stress.csv');assert len(stress)==108 and all(int(r['inputs_and_guards']) and int(r['unequal_outputs'])==0 for r in stress)
stress_rows=[]
for fixture in sorted({x['fixture'] for x in stress}):
    t=[x for x in stress if x['fixture']==fixture]
    assert len(t)==18
    stress_rows.append(dict(fixture=fixture,cells=sum(int(x['cells']) for x in t),on_max=max(float(x['max_on']) for x in t),off_max=max(float(x['max_off']) for x in t),nearest_max=max(float(x['max_nearest']) for x in t),on_over=sum(int(x['over_on']) for x in t),off_over=sum(int(x['over_off']) for x in t),nearest_over=sum(int(x['over_nearest']) for x in t),on_new_over=sum(int(x['newly_over_on']) for x in t)))
for mode in ('installed-on','installed-off','installed-trace','make-ON','make-OFF'):
    guards=records(mode+'-guards.csv');assert len(guards)==60
    assert all(int(x['prefix_guard'])==int(x['inputs_readonly'])==1 and 0<=float(x['max_abs_error'])<=.002 for x in guards)
    if mode=='installed-trace':assert all(x['expected_route']==x['actual_route'] for x in guards)
summary['cells_stress_per_build']=sum(x['cells'] for x in stress_rows);summary['protected_cells_per_build']=sum(int(x['checked']) for x in guards)
for file,entries in [('production-summary.csv',result),('stress-summary.csv',stress_rows)]:
    with (root/file).open('w') as f:
        w=csv.DictWriter(f,entries[0]);w.writeheader();w.writerows(entries)
(root/'verified.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary))
