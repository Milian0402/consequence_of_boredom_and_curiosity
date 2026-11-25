"""Score one implementation against the frozen exact-dot panel, without timing."""
import argparse
import csv
import hashlib
import json
import math
import platform
import struct
import sys
from collections import Counter
from pathlib import Path

from fixtures import build, CASES, KINDS
from oracle import MAG, SIGN, INF, NAN, MODES, exact, model, units, ulps
from registration import SEEDS, verify, word_hash
from quality import FIELDS as QUALITY, POLICY_SHA256, compare, decision

COUNTS = ('outputs', 'exact_finite_dots', 'finite_targets', 'exact_zero', 'zero_scale',
          'nonfinite_targets', 'nan_targets', 'infinite_targets', 'unavoidable_overflow',
          'class_fail', 'finite_target_nonfinite', 'wrong_rounded', 'positive_to_negative',
          'negative_to_positive', 'zero_lost', 'abs_gt_002', 'abs_floor_gt_002',
          'rel_gt_1pct', 'well_conditioned', 'well_ulp_gt_2', 'gamma_exceeded')
MAXIMA = ('max_abs', 'max_rel', 'max_rho', 'max_ulp', 'frob_scaled', 'frob_relative')


def classify(u):
    if u & MAG < INF:
        return 'finite'
    if u & MAG > INF:
        return 'nan'
    return '-inf' if u & SIGN else '+inf'


def score(expected, observed, n, k):
    if len(expected) != len(observed):
        raise ValueError('Incomplete output')
    c = dict.fromkeys(COUNTS, 0)
    c['outputs'] = len(expected)
    maxima = {key: None for key in MAXIMA}
    witnesses = {}
    sqerr = sqscale = sqref = 0
    finite_pairs = 0
    for idx, ((target, p, scale), got) in enumerate(zip(expected, observed, strict=True)):
        def witness():
            return dict(row=idx//n, col=idx%n, target_bits=f'{target:08x}', observed_bits=f'{got:08x}',
                        exact_dot_units_2m298=p, absolute_sum_units_2m298=scale)
        def update(key, value):
            if maxima[key] is None or value > maxima[key]:
                maxima[key] = value
                witnesses[key] = witness()

        tc = classify(target)
        gc = classify(got)
        c['exact_finite_dots'] += p is not None
        if tc != 'finite':
            c['nonfinite_targets'] += 1
            c['nan_targets' if tc == 'nan' else 'infinite_targets'] += 1
            c['unavoidable_overflow'] += p is not None
        else:
            c['finite_targets'] += 1
            c['exact_zero'] += p == 0
            c['zero_scale'] += scale == 0
            floor = abs((units(target) << 149) - p)
            c['abs_floor_gt_002'] += 500*floor > 1 << 298
            c['well_conditioned'] += scale > 0 and 4*abs(p) >= scale

        if tc != gc:
            c['class_fail'] += 1
            c['wrong_rounded'] += 1
            witnesses.setdefault('first_class_failure', witness())
            if tc == 'finite':
                c['finite_target_nonfinite'] += 1
            continue
        if tc != 'finite':
            continue

        finite_pairs += 1
        output = units(got) << 149
        err = abs(output-p)
        distance = ulps(target, got)
        c['wrong_rounded'] += distance != 0
        c['zero_lost'] += (target & MAG) != 0 and (got & MAG) == 0
        c['positive_to_negative'] += p > 0 and output < 0
        c['negative_to_positive'] += p < 0 and output > 0
        c['abs_gt_002'] += 500*err > 1 << 298
        c['rel_gt_1pct'] += p != 0 and 100*err > abs(p)
        c['well_ulp_gt_2'] += scale > 0 and 4*abs(p) >= scale and distance > 2
        c['gamma_exceeded'] += err*((1<<24)-k) > k*scale
        update('max_abs', math.ldexp(float(err), -298))
        update('max_ulp', distance)
        if p:
            update('max_rel', err/abs(p))
        if scale:
            update('max_rho', err/scale)
        elif err:
            witnesses.setdefault('nonzero_at_zero_scale', witness())
        sqerr += err**2
        sqref += p**2
        sqscale += scale**2
    if finite_pairs and not c['finite_target_nonfinite']:
        maxima['frob_scaled'] = math.sqrt(sqerr/sqscale) if sqscale else (None if sqerr else 0.)
        maxima['frob_relative'] = math.sqrt(sqerr/sqref) if sqref else (None if sqerr else 0.)
    return c | maxima, witnesses


def run(args):
    register = verify()
    policy=Path(__file__).with_name('direct-control-policy.md')
    if hashlib.sha256(policy.read_bytes()).hexdigest()!=POLICY_SHA256:
        raise ValueError('Direct-control policy differs from the recorded freeze')
    if args.library:
        from native import Native
        engine = Native(args.library,args.control_library)
        modes = ('fp32-control',) + MODES
    else:
        engine = None
        modes = ('fp32-fma',) + MODES
    control_mode=modes[0]
    if args.modes:
        choices={str(x):x for x in modes}
        modes=tuple(choices[x.strip()] for x in args.modes.split(','))
        if len(set(modes))!=len(modes):
            raise ValueError('Repeated mode')
    base = Path(args.out)
    base.mkdir(parents=True, exist_ok=True)
    records = []
    for case in CASES:
        for seed in SEEDS:
            m,n,k,a,b = build(case, seed)
            targets = exact(a,b,m,n,k)
            if engine:
                direct,control_meta=engine.run(a,b,m,n,k,control_mode)
            else:
                direct=model(a,b,m,n,k,control_mode)
                control_meta=dict(returncode=0,route='software',reason='sequential-fp32-fma',primitive_calls=None,plans=[])
            for mode in modes:
                if mode==control_mode:
                    output,meta=direct,control_meta
                elif engine:
                    output, meta = engine.run(a,b,m,n,k,mode)
                else:
                    output = model(a,b,m,n,k,mode)
                    meta = dict(returncode=0,route='software',reason='sequential-fp32-fma',primitive_calls=None,plans=[])
                row = dict(case=case, category=KINDS[case], seed=seed, m=m, n=n, k=k, mode=str(mode), **meta)
                if meta['returncode'] == 0:
                    metrics, examples = score(targets, output, n, k)
                    metrics.update(compare(targets,output,direct))
                    canonical=[NAN if x&MAG>INF else 0 if x&MAG==0 else x for x in output]
                    row.update(metrics, witnesses=examples, output_sha256=word_hash(output), semantic_sha256=word_hash(canonical),
                               output_words_le_hex=struct.pack(f"<{len(output)}I", *output).hex(),
                               control_words_le_hex=struct.pack(f"<{len(direct)}I", *direct).hex())
                records.append(row)

    verdicts={str(mode):decision([r for r in records if r['mode']==str(mode)]) for mode in modes}
    result=dict(registration=register, backend=args.label, control_library_sha256=engine.control_digest if engine else None,
                  direct_policy_sha256=POLICY_SHA256,verdicts=verdicts,
                  library_sha256=engine.digest if engine else None, platform=platform.platform(),
                  python=sys.version.split()[0], cases=records)
    (base/'scores.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    (base/'verdicts.json').write_text(json.dumps(verdicts, indent=2)+'\n')
    columns = ('case','category','seed','m','n','k','mode','route','returncode','primitive_calls','reason','plans',
               'output_sha256','semantic_sha256') + COUNTS + MAXIMA + QUALITY
    with (base/'cases.csv').open('w',newline='') as f:
        w = csv.DictWriter(f,columns,extrasaction='ignore',lineterminator='\n')
        w.writeheader()
        for row in records:
            w.writerow(row | dict(plans=json.dumps(row['plans'],sort_keys=True,separators=(',',':'))))
    for mode in modes:
        group=[r for r in records if r['mode']==str(mode) and r['category']=='normal']
        print(mode,'ordinary complete',sum(r['returncode']==0 for r in group),'/',len(group),
              'class failures',sum(r.get('class_fail',0) for r in group),
              'finite pairs outside gamma',sum(r.get('gamma_exceeded',0) for r in group),
              'routes',dict(Counter(r['route'] for r in group)))


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    g=ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--model',action='store_true',help='Run explicit unrestricted arithmetic models')
    g.add_argument('--library',help='Run the actual shared library and its ordinary FP32 control')
    ap.add_argument('--control-library',help='Separate COB shared library for the actual direct FP32 comparator')
    ap.add_argument('--modes',help='Comma-separated native/model modes. A control is still measured for each case.')
    ap.add_argument('--label',required=True,help='Declare backend or control provenance')
    ap.add_argument('--out',required=True,help='Result directory for scores.json and cases.csv')
    run(ap.parse_args())
