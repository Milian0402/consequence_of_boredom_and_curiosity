"""Independently replay two existing local Apple timing/accuracy tables, with no kernel execution."""
import csv,json,math,re,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SHAPES=[(512,512,512),(1024,1024,1024),(2048,2048,2048),(4096,4096,4096),(1471,1973,1559),(187,2711,433)]
SEEDS=(204716321,745821359)
CONTROLS=['accelerate','cob','maxmulsk','kleidiai']
def req(v,msg):
    if not v:raise ValueError(msg)
def read(p):
    with p.open(newline='') as f:return list(csv.DictReader(f))
def index(rows,ks):
    out={tuple(r[k] for k in ks):r for r in rows}
    req(len(out)==len(rows),'duplicate observations');return out
def replay(kind,subjects):
    root=ROOT/kind;acc=read(root/'accuracy.csv');times=read(root/'time.csv')
    printed=json.loads((root/'summary.json').read_text()); ident=json.loads((root/'identity.json').read_text())
    methods=CONTROLS+subjects;med=statistics.median
    req(ident['machine']=='arm64' and ident['streaming_fp32_lanes']==16 and ident['one_thread_openblas']==1,'platform')
    req(len(acc)==144 and len(times)==432,'expected 6 layouts × 6 methods × 4 accuracy/12 warm')
    aa=index(acc,('shape','method','seed','fixture'));tt=index(times,('shape','method','seed','round'))
    result={};full={};gather={};coordinates={};ratios=[];hashes=ident['libraries']
    for h,mnk in enumerate(SHAPES):
        key=str(h);m,n,k=mnk;valid={};excluded={};observed={};ms={}
        for name in methods:
            failed=[]
            for s,fix in [(SEEDS[0],x) for x in ('centered','positive','cancellation')]+[(SEEDS[1],'centered')]:
                a=aa[key,name,str(s),fix];c=aa[key,'cob',str(s),fix]
                req(tuple(int(a[z]) for z in ('m','n','k'))==mnk and int(a['cells'])==m*n,'shape/coverage')
                if a['route']=='executed':
                    req((a['ref_max'],a['ref_rms'])==(c['ref_max'],c['ref_rms']),'golden differs')
                    detail=[ff for ff,t in [('nonfinite',int(a['nonfinite'])>0),('absolute',int(a['over_abs'])>0),
                        ('direct-max',float(a['max_abs'])>2*float(c['max_abs'])+2**-23*float(c['ref_max'])),
                        ('direct-RMS',float(a['rms'])>2*float(c['rms'])+2**-23*float(c['ref_rms']))] if t]
                else:detail=['refused/fallback']
                if detail:failed.append(f'{fix}@{s}: '+','.join(detail))
                if name=='cob':coordinates[key,str(s),fix]=tuple(c[z] for z in ('ref_max','ref_rms','max_abs','rms','over_abs'))
            valid[name]=not failed;excluded[name]=failed
            req(printed[key]['methods'][name]['accuracy_pass']==valid[name],'hosted accuracy')
            vals=[]
            for sid,s in enumerate(SEEDS):
                order=methods[::-1] if sid else methods
                for rnd in range(6):
                    sh=(rnd+h)%6;expect=order[sh:]+order[:sh];t=tt[key,name,str(s),str(rnd)]
                    wall=float(t['seconds']);caller=float(t['thread_seconds']);ratios.append(caller/wall)
                    req(tuple(int(t[z]) for z in ('m','n','k'))==mnk and int(t['position'])==expect.index(name),'schedule or geometry')
                    req(math.isfinite(wall) and math.isfinite(caller) and min(wall,caller)>0,'wall/caller clock')
                    vals.append(wall)
            observed[name]=vals;ms[name]=1000*med(vals)
            req(math.isclose(ms[name],1000*printed[key]['methods'][name]['median_seconds'],rel_tol=1e-13),'hosted median')
        strongest=min((n for n in CONTROLS if valid[n]),key=ms.get,default=None);fast=min(CONTROLS,key=ms.get)
        req((strongest,fast)==(printed[key]['best_qualified_fp32'],printed[key]['fastest_native_fp32_raw_time']),'rivals')
        comps={}
        for name in subjects:
            comp={}
            for tier,rival in [('vs_qualified',strongest),('vs_fastest_fp32_raw_time',fast)]:
                if not rival:continue
                v=[b/a for b,a in zip(observed[rival],observed[name])];mid=med(v);wins=sum(x>1 for x in v)
                orig=printed[key]['methods'][name][tier];req(orig['peer']==rival and orig['paired_wins']==wins and math.isclose(orig['paired_median_speed'],mid,rel_tol=1e-14),'hosted pair ratios')
                comp[tier]={'rival':rival,'ratio':mid,'wins':wins,'raw_pairs':v}
            comps[name]=comp
        result[key]=dict(mnk=mnk,qualified_fp32=strongest,fastest_raw=fast,warm_ms=ms,accuracy=valid,exclusions=excluded,comparisons=comps)
    for name in subjects:
        full[name]={'qualified':sum(r['accuracy'][name] for r in result.values())}
        for typ in ('vs_qualified','vs_fastest_fp32_raw_time'):
            vals=[r['comparisons'][name][typ] for r in result.values() if typ in r['comparisons'][name]]
            full[name][typ]=dict(geomean=statistics.geometric_mean(v['ratio'] for v in vals),paired_wins=sum(v['wins'] for v in vals),pairs=12*len(vals))
    return dict(cases=result,overall=full,host=ident,reference_cells=sum(int(r['cells']) for r in acc),timed=len(times),caller_wall_ratio_range=[min(ratios),max(ratios)],caller_wall_median=med(ratios)),coordinates
def run():
    a,ra=replay('results',['fp16-raw','fp16-three']);b,rb=replay('higher',['fp16-six','fp16-nine'])
    req(ra==rb and a['host']['libraries']==b['host']['libraries'],'control numerics/identities between runs')
    return dict(three_product_panel=a,six_nine_product_panel=b,
        evidence_limit='The archived input hashes are absent. The controller asserts unchanged A/B parent SHA-256 within each centered timing block; identical reference summaries and direct COB bounded errors across both local runs corroborate regenerated inputs. Each accuracy fixture was fully checked. Warm calls checked C guards and finiteness, and operator counters for FP16. OpenBLAS oracle reported one thread; the native wrappers received single-thread settings and caller CPU/wall were recorded. Per-call total process task count, CPU pinning and migration were not recorded.')
if __name__=='__main__':print(json.dumps(run(),indent=2,allow_nan=False))
