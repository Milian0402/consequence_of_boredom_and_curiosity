"""Rank complete paired timings of the fixed native grouped-B panel."""
import argparse
import csv
import hashlib
import json
import math
import statistics as stats
from collections import Counter, defaultdict
from pathlib import Path

PANEL_SHA = '5e470c40de6e63b6debbb96afd146eda58de83e50ef3da1e2f19186a8990b4c4'
SEEDS = (267491683, 1893702713)
BASE, NEW = 'stream-incumbent', 'grouped-b'
DIRECT = ('openblas', 'mkl', 'mkl-jit', 'aocl', 'blasfeo', 'onednn-oneshot', 'libxsmm-dispatch')
RETAINED = ('onednn-cached-plan', 'libxsmm-retained')
ROSTER = ('openblas', 'mkl', 'mkl-jit', 'aocl', 'blasfeo', 'onednn-oneshot', 'onednn-cached-plan',
          'libxsmm-dispatch', 'libxsmm-retained', BASE, NEW)


def require(check, message):
    if not check:
        raise ValueError(message)


def rows(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def paired(reference, subject):
    ratios = [r / s for r, s in zip(reference, subject, strict=True)]
    return {
        'ratios': ratios, 'median': stats.median(ratios),
        'wins': sum(r > 1 for r in ratios),
        'seeds': [stats.median(ratios[:3]), stats.median(ratios[3:])],
    }


def aggregate(comparisons):
    if len(comparisons) != 8:
        return None
    medians = [c['median'] for c in comparisons]
    deletions = [stats.geometric_mean(medians[:i] + medians[i + 1:]) for i in range(8)]
    return {
        'geomean': stats.geometric_mean(medians),
        'median_wins': sum(v > 1 for v in medians),
        'wins_of_48': sum(c['wins'] for c in comparisons),
        'worst': min(medians),
        'seed_geomeans': [stats.geometric_mean(c['seeds'][j] for c in comparisons) for j in range(2)],
        'leave_one_layout_out': [min(deletions), max(deletions)],
        'repeatable_layout_leads': sum(c['wins'] >= 5 and c['median'] > 1.02 for c in comparisons),
    }


def qualify_samples(layout, records, page, exclusions):
    from verify.layout import verify_layout
    reasons = list(exclusions)
    expected = [('random', SEEDS[0]), ('random', SEEDS[1]), ('cancel', SEEDS[0]), ('scaled', SEEDS[1])]
    valid = [r for r in records if int(r['round']) < 0]
    warm = sorted((r for r in records if int(r['round']) >= 0), key=lambda r: int(r['round']))
    if (sorted((r['mode'], int(r['seed'])) for r in valid) != sorted(expected) or
            any(int(r['round']) != -1 or int(r['calls']) or int(r['place']) != -1 for r in valid)):
        reasons.append('missing, duplicate or invalid FP64 fixtures')
    if len(warm) != 6 or any(
            int(r['round']) != j or int(r['seed']) != SEEDS[j // 3] or int(r['calls']) != 32 or r['mode'] != 'random'
            for j, r in enumerate(warm[:6])):
        reasons.append('missing, duplicate or invalid matched timings')
    for r in records:
        try:
            verify_layout(layout, r, page)
            require(0 <= float(r['error']) <= 0.002, 'FP64 finite-output/max-absolute gate')
            require(0 < float(r['first_s']) < math.inf, 'invalid first-call time')
            require(int(r['threads']) == 1 and int(r['cpu']) >= 0, 'worker thread/CPU gate')
            value = float(r['per_call_s'])
            require((0 < value < math.inf) if int(r['round']) >= 0 else value == 0, 'invalid batch time')
        except ValueError as exc:
            reasons.append(str(exc))
    if reasons:
        return None, list(dict.fromkeys(reasons))
    return {
        'warm': [float(r['per_call_s']) for r in warm],
        'first_random_per_warm_worker': [float(r['first_s']) for r in warm],
        'first_qualification': [{'mode': r['mode'], 'seed': int(r['seed']), 'seconds': float(r['first_s'])} for r in valid],
        'max_error': max(float(r['error']) for r in records),
        'checked': sum(int(r['checked']) for r in records),
        'guards': sum(int(r['guards']) for r in records),
    }, []


def evaluate(layouts, raw, page, exclusions, roster_order):
    require(len(layouts) == 12 and sum(x['family'].startswith('fresh-') for x in layouts) == 8, 'panel size')
    require(set(roster_order) == set(ROSTER) and len(roster_order) == len(ROSTER), 'registered order')
    by_cell = defaultdict(list)
    identities = {(s['family'], s['layout']) for s in layouts}
    cpus = set()
    for r in raw:
        key = r['family'], r['layout'], r['method']
        require(key[:2] in identities and key[2] in ROSTER, f'unregistered raw cell {key}')
        by_cell[key].append(r)
        cpus.add(r['cpu'])
    require(len(cpus) == 1, f'single CPU across raw records: {cpus}')
    result = []
    for s in layouts:
        case = (s['family'], s['layout'])
        observed, failed = {}, {}
        for method in ROSTER:
            cell = (*case, method)
            data, reasons = qualify_samples(s, by_cell[cell], page, exclusions.get(cell, ()))
            if data is None:
                failed[method] = reasons
                continue
            observed[method] = data
        us = {m: stats.median(data['warm']) * 1e6 for m, data in observed.items()}
        winners = {}
        for tier, methods in [('one_shot', DIRECT), ('including_retained', (*DIRECT, *RETAINED)), ('whole', ROSTER)]:
            admitted = [m for m in methods if m in us]
            if admitted:
                winners[tier] = min(admitted, key=us.get)
        pairs = {
            m: {peer: paired(peer_data['warm'], observed[m]['warm']) for peer, peer_data in observed.items() if peer != m}
            for m in (BASE, NEW) if m in observed
        }
        result.append({'layout': s, 'methods': observed, 'exclusions': failed, 'winners': winners, 'median_us': us, 'pairs': pairs})
    primary = [r for r in result if r['layout']['family'].startswith('fresh-')]
    summaries = {}
    for subject in (BASE, NEW):
        for comparison in (BASE, 'one_shot', 'including_retained'):
            if subject == comparison:
                continue
            xs = []
            for r in primary:
                peer = r['winners'].get(comparison, comparison)
                if peer in r['pairs'].get(subject, {}):
                    xs.append(r['pairs'][subject][peer])
            summaries[f'{subject}/{comparison}'] = aggregate(xs)
    incumbent = summaries[f'{NEW}/{BASE}']
    clean_standalone = all(BASE in r['methods'] and NEW in r['methods'] for r in result)
    timing_gate = bool(incumbent and clean_standalone and incumbent['geomean'] >= 1.03 and
                       incumbent['median_wins'] >= 6 and min(incumbent['seed_geomeans']) > 1 and incumbent['worst'] >= 0.97)
    return dict(cpus=sorted(map(int, cpus)), cells=result, aggregates=summaries, standalone_timing_gate=timing_gate,
                qualified_method_layout_counts=dict(Counter(m for r in result for m in r['methods'])),
                tier_complete={label: all(set(ms) <= set(r['methods']) for r in primary)
                               for label, ms in [('one_shot', DIRECT), ('including_retained', (*DIRECT, *RETAINED))]})


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('raw', type=Path)
    cli.add_argument('output', type=Path)
    args = cli.parse_args()
    root = args.raw
    require(hashlib.sha256((root/'panel.csv').read_bytes()).hexdigest() == PANEL_SHA, 'unregistered panel')
    exclusions = defaultdict(list)
    admissions = root/'admissions.csv'
    if admissions.exists():
        for r in rows(admissions):
            if r['qualified'] == '0':
                exclusions[(r['family'],r['layout'],r['method'])].append(r['reason'])
    data = evaluate(rows(root/'panel.csv'), rows(root/'raw.csv'), int((root/'page-size.txt').read_text()), exclusions, ROSTER)
    args.output.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
