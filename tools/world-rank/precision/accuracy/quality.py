"""Frozen companion direct-control screen and per-method CI verdict."""
import math
from collections import Counter
from fixtures import CASES, KINDS
from oracle import INF, MAG, units
from registration import SEEDS

FIELDS = ('control_class_delta', 'control_finite_pairs', 'control_zero_changed',
          'control_delta_max_abs', 'control_delta_rms_abs', 'control_delta_relmax', 'control_delta_relrms',
          'rms_abs', 'control_exact_max_abs', 'control_exact_rms_abs', 'half_ulp_max', 'half_ulp_rms',
          'quality_max_limit', 'quality_rms_limit', 'quality_max_pass', 'quality_rms_pass')
POLICY_SHA256='90521489d614a9faf26c7147e00809b5b167fb5550c37ef15cdbb8ff6ff6d400'


def _real(q):
    return math.ldexp(float(q), -298)


def _rms(qs, n):
    return _real(math.sqrt(qs/n))


def _kind(x):
    return 'finite' if (x&MAG)<INF else 'nan' if (x&MAG)>INF else x


def compare(reference, output, direct):
    counts = dict(control_class_delta=0, control_finite_pairs=0, control_zero_changed=0)
    maximum = dict(error=0, direct=0, half=0, delta=0, magnitude=0)
    ss = dict.fromkeys(maximum,0)
    eligible = len(reference)
    for (target,p,_), value, ctrl in zip(reference,output,direct,strict=True):
        vc,cc = _kind(value), _kind(ctrl)
        counts['control_class_delta'] += vc != cc
        if vc == cc == 'finite':
            v,d = units(value)<<149, units(ctrl)<<149
            counts['control_finite_pairs'] += 1
            counts['control_zero_changed'] += d==0 and v!=0
            for key, q in (('delta', abs(v-d)), ('magnitude',abs(d))):
                maximum[key]=max(maximum[key],q)
                ss[key]+=q*q
        if (target&MAG)>=INF or vc!='finite' or cc!='finite':
            eligible-=1
            continue
        # P is an integer in units of 2^-298; one half-ULP is at least 2^-150.
        half=1 << max(148,abs(p).bit_length()-25)
        for key,q in (('error',abs(v-p)),('direct',abs(d-p)),('half',half)):
            maximum[key]=max(maximum[key],q)
            ss[key]+=q*q
    results=dict.fromkeys(FIELDS,None) | counts
    pairs=counts['control_finite_pairs']
    if pairs:
        results['control_delta_max_abs']=_real(maximum['delta'])
        results['control_delta_rms_abs']=_rms(ss['delta'],pairs)
        if maximum['magnitude']:
            results['control_delta_relmax']=maximum['delta']/maximum['magnitude']
            results['control_delta_relrms']=math.sqrt(ss['delta']/ss['magnitude'])
        elif not maximum['delta']:
            results['control_delta_relmax']=results['control_delta_relrms']=0.
    if eligible==len(reference):
        max_limit=2*max(maximum['direct'],maximum['half'])
        sum_limit=4*max(ss['direct'],ss['half'])
        results.update(rms_abs=_rms(ss['error'],eligible), control_exact_max_abs=_real(maximum['direct']),
                       control_exact_rms_abs=_rms(ss['direct'],eligible),half_ulp_max=_real(maximum['half']),
                       half_ulp_rms=_rms(ss['half'],eligible),quality_max_limit=_real(max_limit),
                       quality_rms_limit=_rms(sum_limit,eligible),
                       quality_max_pass=maximum['error']<=max_limit,quality_rms_pass=ss['error']<=sum_limit)
    return results


def decision(rows):
    """Evaluate a single method across the complete frozen roster; missing data stays visible."""
    keys=Counter((r['case'],r['seed']) for r in rows)
    complete=(keys==Counter((c,s) for c in CASES for s in SEEDS) and
              all(r['returncode']==0 for r in rows))
    def group(predicate, check):
        selected=[r for r in rows if predicate(r)]
        want=Counter((c,s) for c in CASES if predicate(dict(case=c,category=KINDS[c])) for s in SEEDS)
        if Counter((r['case'],r['seed']) for r in selected)!=want or any(r['returncode'] for r in selected):
            return None
        return all(check(r) for r in selected)
    normal=lambda r:r['category']=='normal'
    verdict=dict(complete=complete,
        normal_envelope=group(normal, lambda r:r['class_fail']==r['gamma_exceeded']==0),
        hk_exact=group(lambda r:r['case'].startswith('hk-'),lambda r:r['wrong_rounded']==0),
        zero_overwrite=group(lambda r:r['category']=='zero',lambda r:r['wrong_rounded']==0),
        range_classes=group(lambda r:r['category']=='range',lambda r:r['class_fail']==0),
        exceptional_classes=group(lambda r:r['category']=='nonfinite',lambda r:r['class_fail']==0))
    verdict['accuracy_gate']=all(verdict.values())
    normal_rows=[r for r in rows if normal(r)]
    available=group(normal,lambda r:r.get('quality_max_pass') is not None) is True
    verdict['direct_quality']=all(r['quality_max_pass'] and r['quality_rms_pass'] for r in normal_rows) if available else None
    verdict['direct_quality_breaches']=[dict(case=r['case'],seed=r['seed'],max=r['quality_max_pass'],rms=r['quality_rms_pass'])
        for r in normal_rows if r.get('quality_max_pass') is not None and not (r['quality_max_pass'] and r['quality_rms_pass'])]
    verdict['matched_direct_accuracy']=verdict['accuracy_gate'] and verdict['direct_quality'] is True
    verdict['ordinary_route_counts']=dict(Counter(str(r['route']) for r in normal_rows))
    verdict['ordinary_component_route']=group(normal,lambda r:r['route']==1) if any(
        isinstance(r['route'],int) for r in rows) else None
    return verdict
