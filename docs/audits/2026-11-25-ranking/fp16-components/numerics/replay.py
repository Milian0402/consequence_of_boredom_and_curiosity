"""Check archived FP32 words, exact-dyadic witnesses and every gate without a native library."""
import argparse,csv,hashlib,json,math,struct
from collections import defaultdict
from locate import HERE,reference

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--repo')
args=ap.parse_args()
reference(args.repo)
from evaluate import score, COUNTS, MAXIMA
from fixtures import CASES,KINDS,build
from quality import FIELDS,compare,decision
from oracle import exact,ulps
from registration import SEEDS,verify

def read(name):
    with (HERE/'data'/name).open() as f:return list(csv.DictReader(f))
def equiv(x,txt):
    if x is None:return txt==''
    if isinstance(x,(int,bool,str)):return str(x)==txt
    return math.isclose(float(txt),x,rel_tol=2e-15,abs_tol=0)
def unpack(text):
    raw=bytes.fromhex(text)
    return list(struct.unpack(f'<{len(raw)//4}I',raw))
verify()
words={}
for w in read('words.csv'):
    raw=bytes.fromhex(w['little_endian_binary32_hex'])
    assert hashlib.sha256(raw).hexdigest()==w['sha256'] and w['sha256'] not in words
    words[w['sha256']]=unpack(w['little_endian_binary32_hex'])
refs={}
for c in CASES:
    for s in SEEDS:
        m,n,k,a,b=build(c,s)
        refs[c,s]=(m,n,k,exact(a,b,m,n,k))
ws={r['witness_id']:r for r in read('witnesses.csv')}
scores={}
for row in read('scores.csv'):
    c,s=row['case'],int(row['seed']);m,n,k,r=refs[c,s]
    out=words[row['output_sha256']];direct=words[row['control_output_sha256']]
    metrics,w=score(r,out,n,k);metrics.update(compare(r,out,direct))
    assert all(equiv(x,row[key]) for key,x in metrics.items()),(c,s)
    named={a:b for a,b in (x.split('=',1) for x in row['witnesses'].split('|') if x)}
    assert set(w)==set(named)
    for name, y in w.items():
        saved=ws[named[name]]
        assert saved['case']==c and int(saved['seed'])==s and all(equiv(x,saved[key]) for key,x in y.items())
    assert row['score_id'] not in scores
    scores[row['score_id']]=(c,s,metrics)
by=defaultdict(list)
for run in read('executions.csv'):
    c,s,v=scores[run['score_id']]
    assert run['case']==c and int(run['seed'])==s
    route=run['route'];route=int(route) if route.isdigit() else route
    by[run['implementation'],run['mode']].append(dict(case=c,category=KINDS[c],seed=s,route=route,returncode=int(run['returncode']),**v))
gates=read('verdicts.csv')
assert len(by)==len(gates)==48
for row in gates:
    result=decision(by[row['implementation'],row['mode']])
    for name in ('complete','ordinary_component_route','normal_envelope','hk_exact','zero_overwrite','range_classes',
                 'exceptional_classes','accuracy_gate','direct_quality','matched_direct_accuracy'):
        assert equiv(result[name],row[name])
    breaches=result['direct_quality_breaches']
    assert int(row['ordinary_companion_pass'])==18-len(breaches)
    assert row['breaches']=='|'.join(f"{b['case']}@{b['seed']}:{'M' if not b['max'] else ''}{'R' if not b['rms'] else ''}" for b in breaches)

for x in read('scalar-witnesses.csv'):
    a,b=[int(x[n],16) for n in ('A','B')]
    r=exact([a],[b],1,1,1);z=[int(x['observed'],16)];d=[int(x['direct'],16)]
    assert r[0][0]==int(x['exact_rounded'],16) and int(x['ulp'])==ulps(z[0],r[0][0])
    metrics,_=score(r,z,1,1)
    assert metrics['gamma_exceeded']==int(x['gamma_exceeded']) and equiv(compare(r,z,d)['quality_max_pass'],x['max_criterion'])
print(f'Verified {len(refs)} registered input matrices, {sum(len(g) for g in by.values())} execution rows, {len(scores)} distinct input/result scores, {len(ws)} exact witnesses and {len(gates)} verdicts.')
