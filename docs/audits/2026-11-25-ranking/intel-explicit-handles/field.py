"""Paired ranking of standalone SGEMM against the original and expanded independent fields."""
from math import isfinite
from statistics import geometric_mean, median

COB = 'stream-incumbent'
DIRECT = ('openblas', 'mkl', 'mkl-jit', 'aocl', 'blasfeo', 'onednn-oneshot', 'libxsmm-dispatch')
RETAINED = ('onednn-cached-plan', 'libxsmm-retained')
ONESHOT, READY = 'mkl-handle-oneshot', 'mkl-handle-prepared'
ROSTER = (*DIRECT, *RETAINED, ONESHOT, READY, COB)
TIERS = {
    'original_one_shot': DIRECT,
    'original_including_retained': (*DIRECT, *RETAINED),
    'extended_one_shot': (*DIRECT, ONESHOT),
    'extended_including_retained': (*DIRECT, *RETAINED, ONESHOT, READY),
}
REGIMES = ('warm', 'first', 'first_with_setup')
SEEDS = (1587984119, 384761239)


def observations(data, regime):
    """Return six whole-call observations in seconds, in recorded seed/round order."""
    if regime == 'first_with_setup':
        return [first + prep for first, prep in zip(data['first'], data['prep'], strict=True)]
    if regime not in ('warm', 'first'):
        raise ValueError(f'Unknown timing regime {regime}')
    return list(data[regime])


def compare(subject, peer):
    """Use the same six worker seeds/rounds for both the raw and the paired ratios."""
    if len(subject) != 6 or len(peer) != 6:
        raise ValueError('Expected six matched subject and peer observations')
    if not all(isfinite(x) and x > 0 for x in (*subject, *peer)):
        raise ValueError('Invalid time in a qualified comparison')
    raw_subject, raw_peer = median(subject), median(peer)
    ratios = [p / s for p, s in zip(peer, subject, strict=True)]
    return {'subject_median_us': raw_subject*1e6, 'peer_median_us': raw_peer*1e6,
            'raw_ratio': raw_peer/raw_subject, 'paired_median': median(ratios),
            'ratios': ratios, 'wins': sum(x > 1 for x in ratios),
            'seeds': [median(ratios[:3]), median(ratios[3:])]}


def rank_case(cell):
    methods = cell['methods']
    for name, data in methods.items():
        for key in ('warm', 'first', 'prep'):
            if len(data[key]) != 6 or not all(isfinite(t) for t in data[key]):
                raise ValueError(f'Six finite {key} values required for {name}')
        if min(data['warm']) <= 0 or min(data['first']) <= 0 or min(data['prep']) < 0:
            raise ValueError(f'Invalid call or preparation duration for {name}')
        if name != READY and any(data['prep']):
            raise ValueError(f'Unexpected separate preparation duration for {name}')
    ranking = {}
    for regime in REGIMES:
        times = {n: observations(data, regime) for n, data in methods.items()}
        raw = {n: median(values) for n, values in times.items()}
        tiers = {}
        for tier, roster in TIERS.items():
            admitted = [m for m in roster if m in raw]
            best = min(admitted, key=raw.get) if admitted else None
            result = compare(times[COB], times[best]) if best and COB in times else None
            tiers[tier] = {'method': best, 'comparison': result,
                           'complete': len(admitted) == len(roster), 'admitted': len(admitted)}
        ranking[regime] = tiers
    return ranking


def summarize_comparisons(samples):
    comparisons = [c for c in samples if c is not None]
    if not comparisons:
        return None
    # Six-shape headlines require all six matching subject/leader comparisons.
    # Partial ranks and qualification coverage remain visible in the case rows.
    complete_six = len(comparisons) == 6
    return {
        'geomean': geometric_mean(c['paired_median'] for c in comparisons) if complete_six else None,
        'raw_wins': sum(c['raw_ratio'] > 1 for c in comparisons),
        'paired_wins': sum(c['paired_median'] > 1 for c in comparisons),
        'round_wins': sum(c['wins'] for c in comparisons),
        'seed_geomeans': [geometric_mean(c['seeds'][j] for c in comparisons) if complete_six else None for j in (0, 1)],
        'ranked_layouts': len(comparisons),
    }


def summarize(cases, families, regime, tier):
    original = tier.replace('extended_', 'original_')
    ranks = [cases[f][regime][tier] for f in families]
    totals = summarize_comparisons([r['comparison'] for r in ranks])
    if totals is None:
        return None
    changes = {'changed_leaders': 0, 'changed_raw_wins': 0, 'changed_paired_wins': 0}
    comparables = 0
    for family, rank in zip(families, ranks, strict=True):
        prev = cases[family][regime][original]
        a, b = rank['comparison'], prev['comparison']
        if a is None or b is None:
            continue
        comparables += 1
        changes['changed_leaders'] += rank['method'] != prev['method']
        changes['changed_raw_wins'] += (a['raw_ratio'] > 1) != (b['raw_ratio'] > 1)
        changes['changed_paired_wins'] += (a['paired_median'] > 1) != (b['paired_median'] > 1)
    return {**totals, 'complete_layouts': sum(r['complete'] for r in ranks),
            'change_comparable_layouts': comparables, **changes}


def handle_comparisons(cells, families, comparator):
    """The stock comparator time is the numerator and the explicit method time is the denominator."""
    handles = {}
    for cell in cells:
        methods = cell['methods']
        handles[cell['layout']['family']] = {
            r: {handle: compare(observations(methods[handle], r), observations(methods[comparator], r))
                if handle in methods and comparator in methods else None for handle in (ONESHOT, READY)}
            for r in REGIMES
        }
    return {
        'cases': handles,
        'aggregates': {r: {h: summarize_comparisons([handles[f][r][h] for f in families])
                           for h in (ONESHOT, READY)} for r in REGIMES},
    }


def rank_fields(cells):
    """Select a real qualified peer independently by layout, timing regime and field."""
    if len(cells) != 11:
        raise ValueError('Expected eleven registered physical layouts')
    cases = {}
    for cell in cells:
        family = cell['layout']['family']
        if family in cases:
            raise ValueError(f'Duplicate layout ID {family}')
        cases[family] = rank_case(cell)
    families = [f for f in cases if f.startswith('fresh-')]
    if len(families) != 6:
        raise ValueError('Expected six previously registered target layouts')
    return {
        'cases': cases,
        'aggregates': {r: {tier: summarize(cases, families, r, tier) for tier in TIERS} for r in REGIMES},
        'handles_vs_mkl': handle_comparisons(cells, families, 'mkl'),
        'handles_vs_mkl_direct_call': handle_comparisons(cells, families, 'mkl-jit'),
    }
