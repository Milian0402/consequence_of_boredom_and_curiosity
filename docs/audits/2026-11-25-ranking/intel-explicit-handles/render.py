"""Human-readable and flat-table views of validated native explicit-handle scores."""
import csv
import re
from collections import Counter, defaultdict
from math import ceil
from pathlib import Path
from statistics import geometric_mean, median

from field import COB, ONESHOT, READY, REGIMES, ROSTER, SEEDS, TIERS, compare, observations


NAMES = {
    COB: 'Standalone COB', 'mkl': 'oneMKL', 'mkl-jit': 'oneMKL direct-call',
    'openblas': 'OpenBLAS', 'aocl': 'AOCL', 'blasfeo': 'BLASFEO',
    'onednn-oneshot': 'oneDNN one-shot', 'onednn-cached-plan': 'oneDNN cached',
    'libxsmm-dispatch': 'LIBXSMM dispatch', 'libxsmm-retained': 'LIBXSMM retained',
    ONESHOT: 'Explicit one-shot', READY: 'Explicit prepared',
}
TIERS_SHORT = {
    'original_one_shot': 'Original one-shot (7)', 'extended_one_shot': 'Expanded one-shot (8)',
    'original_including_retained': 'Original incl. retained (9)',
    'extended_including_retained': 'Expanded incl. retained (11)',
}
REGIME_NAMES = {'warm': 'Warm', 'first': 'First execution', 'first_with_setup': 'Setup + first'}
GROUPS = (('original_one_shot', 'extended_one_shot'),
          ('original_including_retained', 'extended_including_retained'))
BASELINES = ('mkl', 'mkl-jit')
GEOMETRY = 'family layout m n k lda ldb ldc ar ac br bc cr cc'.split()


def public_text(value):
    if isinstance(value, (list, tuple)):
        return ' · '.join(public_text(item) for item in value)
    value = ' '.join(str(value).split())
    return re.sub(r'/(?:Users|home|private|tmp|var)/(?:[^\s,;]+)', '[local path]', value)


def cell_text(value):
    return public_text(value).replace('|', r'\|').replace('<', '&lt;').replace('>', '&gt;')


def table(lines, headers, rows):
    lines.extend(['', '| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join('---' for _ in headers) + ' |'])
    lines.extend('| ' + ' | '.join(str(v) for v in row) + ' |' for row in rows)


def ordered(scores):
    return sorted(scores, key=lambda s: (s['receipt']['run'], s['receipt']['replica']))


def principal(cell):
    return cell['layout']['family'].startswith('fresh-')


def label(cell):
    g = cell['layout']
    return f"{g['family']} · {g['m']} × {g['n']} × {g['k']}"


def timing(cell, name, regime):
    data = cell['methods'].get(name)
    return observations(data, regime) if data else None


def match(cell, subject, peer, regime):
    a, b = timing(cell, subject, regime), timing(cell, peer, regime)
    return compare(a, b) if a and b else None


def summary_rows(score):
    rows = []
    for regime in REGIMES:
        for tiers in GROUPS:
            for tier in tiers:
                a = score['rankings']['aggregates'][regime][tier]
                if not a or a['ranked_layouts'] == 0:
                    rows.append([REGIME_NAMES[regime], TIERS_SHORT[tier], *['—'] * 7])
                    continue
                n = a['ranked_layouts']
                effect = '—' if a['geomean'] is None else f"{a['geomean']:.4f}×"
                seeds = a['seed_geomeans']
                seed_text = '—' if not seeds or None in seeds else f"{seeds[0]:.4f} / {seeds[1]:.4f}"
                changes = 'reference'
                if tier.startswith('extended'):
                    counts = ' / '.join(str(a[k]) for k in ('changed_leaders', 'changed_raw_wins', 'changed_paired_wins'))
                    changes = f"{counts} of {a['change_comparable_layouts']}"
                rows.append([REGIME_NAMES[regime], TIERS_SHORT[tier], effect, f"{a['raw_wins']}/{n}",
                             f"{a['paired_wins']}/{n}", f"{a['round_wins']}/{n*6}", seed_text,
                             f"{a['complete_layouts']}/6", changes])
    return rows


def leader_text(info):
    method = info.get('method')
    if not method:
        return 'No qualified leader'
    p = info.get('comparison')
    if p is None:
        return f"{NAMES[method]} · COB unavailable"
    return f"{NAMES[method]} {p['peer_median_us']:.2f} µs · {p['paired_median']:.4f}×"


def leader_change(before, after):
    if before.get('method') == after.get('method'):
        return leader_text(after)
    return leader_text(before) + ' → ' + leader_text(after)


def leader_tables(lines, score):
    lines += ['', '### Fastest independent method on each layout', '',
              'Each entry gives the original leader → the expanded leader, with its median and its paired '
              'time ratio against COB. A single entry means the same leader won both fields. Each regime '
              'selects its leaders from the measurements in that regime.']
    for regime in REGIMES:
        rows = []
        for c in score['cells']:
            refs = score['rankings']['cases'][c['layout']['family']][regime]
            a = timing(c, COB, regime)
            rows.append([label(c), f'{median(a)*1e6:.2f}' if a else '—',
                         *[leader_change(refs[old], refs[new]) for old, new in GROUPS]])
        lines += ['', f'#### {REGIME_NAMES[regime]}']
        table(lines, ['Physical layout · M × N × K', 'COB µs', 'One-shot leaders (7 → 8)',
                      'Retained-inclusive leaders (9 → 11)'], rows)


def handle_table(lines, score):
    lines += ['', '### Explicit handle execution and setup', '',
              'All times are medians in microseconds. “Setup + first” takes the median of six within-worker '
              'sums. The single-call one-shot time includes create, obtain pointer, execute and destroy. '
              'Each prepared first and warm execution checks that the requested geometry matches the '
              'cached geometry and reuses the worker’s handle. Prepared destruction occurs '
              'after the measured products and is outside the setup-inclusive interval.']
    status = Counter((m, c['methods'][m]['api_status']) for c in score['cells'] for m in (ONESHOT, READY)
                     if m in c['methods'])
    summaries = []
    for m in (ONESHOT, READY):
        summaries.append(f"{NAMES[m]} had {status[m, 'MKL_JIT_SUCCESS']} generated and "
                         f"{status[m, 'MKL_NO_JIT']} callable fallback layouts")
    lines += ['', '. '.join(summaries) + '.']
    rows = []
    for c in score['cells']:
        one, ready = c['methods'].get(ONESHOT), c['methods'].get(READY)
        def us(values):
            return f'{median(values)*1e6:.2f}' if values else '—'
        statuses = [d['api_status'] if d else 'excluded' for d in (one, ready)]
        if statuses[0] == statuses[1]:
            statuses = statuses[:1]
        rows.append([c['layout']['family'], ' / '.join(statuses),
                     us(one['warm']) if one else '—', us(one['first']) if one else '—',
                     us(ready['warm']) if ready else '—', us(ready['prep']) if ready else '—',
                     us(ready['first']) if ready else '—', us(timing(c, READY, 'first_with_setup'))])
    table(lines, ['Layout', 'One-shot / prepared API status', 'One-shot warm', 'One-shot first',
                  'Prepared warm', 'Setup', 'Ready first', 'Setup + first'], rows)


def direct_table(lines, score):
    lines += ['', '### Explicit methods against oneMKL controls', '',
              'These ratios divide the control time by the explicit method’s time. Values above one favor '
              'the explicit method. Each entry gives the geometric mean of six principal-layout paired '
              'medians, followed by the count of layouts whose paired median favors the explicit method.']
    rows = []
    cases = [c for c in score['cells'] if principal(c)]
    for subject in (ONESHOT, READY):
        for peer in BASELINES:
            row = [NAMES[subject], NAMES[peer]]
            for regime in REGIMES:
                pairs = [match(c, subject, peer, regime) for c in cases]
                medians = [p['paired_median'] for p in pairs if p]
                value = (f"{geometric_mean(medians):.4f}× · {sum(x > 1 for x in medians)}/6"
                         if len(medians) == 6 else f"Incomplete ({len(medians)}/6 layouts)")
                row.append(value)
            rows.append(row)
    table(lines, ['Explicit method', 'Control', *[REGIME_NAMES[r] for r in REGIMES]], rows)


def generated_handle_example(lines, score):
    c = next((c for c in score['cells'] if principal(c) and
              c['methods'].get(READY, {}).get('api_status') == 'MKL_JIT_SUCCESS'), None)
    if c is None:
        return
    ready = c['methods'][READY]
    setup = ready['prep']
    lines += ['', '### Generated prepared handle on one principal layout', '',
              f"{label(c)} was the first principal layout in panel order whose prepared call returned "
              f"MKL_JIT_SUCCESS. Creating the handle and obtaining its callable before the first product "
              f"took a median {median(setup)*1e6:.3f} µs in the six measured workers."]
    ordinary = timing(c, 'mkl', 'warm')
    if ordinary is None:
        lines.append('Ordinary oneMKL lacked a qualified warm comparison on this layout.')
        return
    elapsed = ready['warm']
    savings = [base - prepared for base, prepared in zip(ordinary, elapsed, strict=True)]
    seed_savings = [median(savings[:3]), median(savings[3:])]
    wins = sum(v > 0 for v in savings)
    lines.append(f"The warm medians were {median(ordinary)*1e6:.3f} µs for ordinary oneMKL and "
                 f"{median(elapsed)*1e6:.3f} µs for the prepared public call. Subtracting prepared time "
                 f"from ordinary time in each matched round gave a median {median(savings)*1e6:+.3f} µs "
                 f"per product, with {wins}/6 positive savings. The seed-block medians were "
                 f"{seed_savings[0]*1e6:+.3f} and {seed_savings[1]*1e6:+.3f} µs.")
    if median(savings) <= 0:
        lines.append('The median warm execution supplied no positive saving to recover setup.')
        return
    if wins != 6:
        lines.append('The observed saving changed sign or reached zero across rounds, so a reuse count '
                     'would depend on which round supplied the saving.')
        return
    # A descriptive sensitivity check, applied separately within each seed's three matched rounds.
    for start in (0, 3):
        baseline, prepared = ordinary[start:start+3], elapsed[start:start+3]
        spread = max(max(baseline) - min(baseline), max(prepared) - min(prepared))
        if any(b - p <= max(b * .01, spread) for b, p in zip(baseline, prepared, strict=True)):
            lines.append('At least one round’s saving was within 1% of ordinary time or within the '
                         'observed time range of one method in that seed block. The modeled reuse count '
                         'would be sensitive to that small difference.')
            return
    counts = [prep / saving for prep, saving in zip(setup, savings, strict=True)]
    lines.append(f"At the observed warm rates, a setup-only model gives approximately "
                 f"{ceil(median(counts))} products to recover preparation (individual rounds "
                 f"{min(map(ceil, counts))}–{max(map(ceil, counts))}). Each ratio divides the "
                 "prepared worker’s setup time by that round’s per-product warm saving. The median and "
                 "range endpoints are rounded up to whole products. This models repeated warm products "
                 "at the same geometry and excludes "
                 "the first execution’s transient cost, library initialization and eventual destruction.")


def exclusions(lines, score):
    grouped = defaultdict(list)
    for c in score['cells']:
        for method, reason in c['exclusions'].items():
            grouped[method, public_text(reason)].append(c['layout']['family'])
    total = len(score['cells']) * len(ROSTER)
    count = sum(len(c['methods']) for c in score['cells'])
    lines += ['', f'{count}/{total} registered method/layout combinations qualified.']
    if grouped:
        table(lines, ['Excluded method', 'Layouts', 'Recorded reason'],
              [[NAMES.get(m, m), ', '.join(fs), cell_text(reason)] for (m, reason), fs in grouped.items()])
    identity = score.get('identity', {})
    if score.get('isa') and identity.get('release'):
        isa = {'avx2': 'AVX2', 'avx512': 'AVX-512'}.get(score['isa'], score['isa'])
        lines += ['', f"The installed COB release `{identity['release']}` used its {cell_text(isa)} kernel "
                  f"on this host. Its ordinary worker SHA-256 was `{identity['ordinary'][COB]}`."]
    n = score.get('numerical', {})
    if 'raw_fp64_outputs' in n:
        lines += ['', f"The numerical checks examined {n['raw_fp64_outputs']:,} FP64-reference output values "
                  f"and {n['checked_parent_storage']:,} guard or padding locations. The largest recorded "
                  f"absolute error was {n['max_absolute_error']:.9g}. Native workers completed "
                  f"{n['complete_warm_calls']:,} warm multiplies in {n['warm_batches']:,} batches."]


def make_report(results):
    """Render validated results without pooling CPU assignments."""
    jobs = ordered(results)
    if not jobs:
        raise ValueError('At least one validated native job is required')
    lines = ['# Standalone COB against explicit oneMKL handles', '',
             'Two explicit oneMKL API policies extend the independent field. The one-shot variant constructs '
             'and destroys a handle for each multiplication. The prepared variant constructs once per worker '
             'and reuses its callable. The subject is the installed standalone COB release performing a '
             'single-threaded row-major FP32 SGEMM with α = 1 and β = 0. Each host was scored separately.',
             '', 'The overview covers six principal layouts reused from the preceding study. The table '
             'compares the original and expanded retained-inclusive fields. The speed ratio is the geometric '
             'mean of the six layout-level paired medians. A ratio above one favors COB.']
    rows = []
    for score in jobs:
        receipt = score['receipt']
        cpu = cell_text(score['model'])
        name = f"{cpu}, replica {receipt['replica']}"
        if receipt.get('job_url'):
            name = f"[{name}]({receipt['job_url']})"
        for regime in REGIMES:
            agg = score['rankings']['aggregates'][regime]
            old, new = [agg[tier] for tier in GROUPS[1]]
            def gm(value):
                return '—' if not value or value['geomean'] is None else f"{value['geomean']:.4f}×"
            wins = (f"{new['raw_wins']}/{new['ranked_layouts']}" if new and new['ranked_layouts'] else '—')
            changed = '—' if not new else f"{new['changed_leaders']}/{new['change_comparable_layouts']}"
            rows.append([name, REGIME_NAMES[regime], gm(old), gm(new), wins, changed])
    table(lines, ['Native job', 'Regime', 'Original 9', 'Expanded 11', 'COB raw wins after expansion',
                  'Leader changes'], rows)
    lines += ['', 'All eleven physical layouts, including three historical anchors and two cropped views, '
              'preserve their dimensions and strides. The rerun used new input draws from seeds '
              f'{SEEDS[0]} and {SEEDS[1]}. Each qualified method supplied three workers per seed. Warm samples '
              'time 32 complete products per worker on resident data. A first execution is the worker’s first '
              'complete product after initialization. Only the explicitly prepared method sets up its handle '
              'before that timer. “Setup + first” adds its measured construction and pointer-lookup time to '
              'the same worker’s first execution. Process startup, library initialization and eventual '
              'prepared-handle destruction are outside this interval. All other policies retain their '
              'observed first-call times. The six matched rounds repeat two input draws on one VM per job, '
              'so small timing differences carry no estimate of variation across machines.',
              '', 'The original seven one-shot methods are OpenBLAS, ordinary sequential oneMKL, its '
              'JIT-capable direct-call macro, AOCL, BLASFEO, oneDNN one-shot and LIBXSMM per-call dispatch. '
              'The original retained-inclusive field also admits oneDNN cached-plan and retained LIBXSMM. '
              'The expanded one-shot field adds the explicit one-shot policy. The expanded retained-inclusive '
              'field admits both explicit policies. MKL_JIT_SUCCESS records a generated implementation and '
              'MKL_NO_JIT records an admitted callable fallback.']
    lines.extend(['', '### Original assignments', '',
                  'The original AMD EPYC 7763 and Intel Xeon Platinum 8573C jobs both ended before timing '
                  'because the adapter build lacked the oneMKL SDK path manifest. Their receipts and native '
                  'checks are preserved separately. The results below come from the two newly assigned replacement '
                  'jobs after correction of the manifest export. '
                  '[AMD job](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45708206524/job/139535589638) · '
                  '[Intel job](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45708206524/job/139535589668) · '
                  '[Original assignment record](initial-attempt/README.md)'])
    for score in jobs:
        replica = score['receipt']['replica']
        lines += ['', f"## {cell_text(score['model'])} · replica {replica}", '',
                  'Raw wins compare COB’s median time with the independent leader’s median. Paired wins use '
                  'the median of the six within-round leader/COB ratios. The seed columns give the '
                  'three-round seed results aggregated across the principal layouts. The final column counts '
                  'layouts whose leader, raw COB win, or paired COB win changed when the two policies '
                  'entered their respective fields.']
        table(lines, ['Regime', 'Independent field', 'Paired geomean', 'COB raw wins', 'COB paired wins',
                      'COB round wins', 'Seed 1 / 2', 'Complete fields', 'Changes (leader / raw / paired)'],
              summary_rows(score))
        leader_tables(lines, score)
        handle_table(lines, score)
        direct_table(lines, score)
        generated_handle_example(lines, score)
        exclusions(lines, score)
    lines += ['', 'The [comparison CSV](comparison.csv) records each qualified method’s absolute measurements and all six ratios '
              'for every COB comparison, along with the four explicit-versus-control comparisons per layout. '
              'The ready-handle rows include setup and first execution from each matched worker. '
              'The recorded timings describe these processors, geometries and call policies.',
              '', 'From the extracted evidence directory, run `python3 -B packet.py .` to replay the '
              'verification, rankings, report and CSV.']
    repo = 'https://github.com/Milian0402/consequence_of_boredom_and_curiosity'
    for revision in dict.fromkeys(s['run'].get('benchmark', '') for s in jobs):
        if re.fullmatch(r'[a-f0-9]{40}', revision):
            lines += ['', f'[Benchmark source]({repo}/tree/{revision}/tools/world-rank/handles) · '
                      f'[Native workflow]({repo}/blob/{revision}/.github/workflows/native-gemm.yml#L543)']
    return '\n'.join(lines) + '\n'


def policy(method):
    if method == READY:
        return 'explicit-prepared-before-first'
    if method == ONESHOT:
        return 'explicit-create-execute-destroy-per-call'
    if method == 'onednn-cached-plan':
        return 'cached-plan'
    if method == 'libxsmm-retained':
        return 'retained-kernel'
    return 'one-shot'


def write_comparisons(path, results):
    """Write qualified and excluded roster entries, with per-round times and pairing."""
    dims = GEOMETRY
    fields = ['run', 'replica', 'cpu', 'vendor', 'operation', *dims, 'principal', 'regime',
              'subject', 'subject_policy', 'subject_api_status', 'subject_qualified', 'subject_exclusion',
              'peer', 'peer_policy', 'peer_api_status', 'peer_qualified', 'peer_exclusion',
              *[f'{tier}_leader' for tier in TIERS], *[f'{tier}_complete' for tier in TIERS],
              'subject_median_us', 'peer_median_us', 'ratio_of_medians', 'median_of_paired_ratios',
              'paired_wins', 'seed1_paired_median', 'seed2_paired_median']
    for j in range(1, 7):
        fields.extend(f'{f}_{j}' for f in ('round', 'seed', 'subject_us', 'peer_us', 'peer_over_subject',
                                            'subject_prep_us', 'peer_prep_us'))
    with Path(path).open('w', encoding='utf8', newline='') as out:
        writer = csv.DictWriter(out, fields, lineterminator='\n')
        writer.writeheader()
        for score in ordered(results):
            for c in score['cells']:
                source, error = c['methods'], c['exclusions']
                for regime in REGIMES:
                    ranks = score['rankings']['cases'][c['layout']['family']][regime]
                    common = dict(run=score['receipt']['run'], replica=score['receipt']['replica'],
                                  cpu=score['model'], vendor=score['vendor'], operation='single-thread FP32 overwrite SGEMM',
                                  **{k: c['layout'][k] for k in dims}, principal=int(principal(c)), regime=regime)
                    for tier, result in ranks.items():
                        common[f'{tier}_leader'] = result.get('method') or ''
                        common[f'{tier}_complete'] = int(result['complete'])
                    pairs = [(COB, peer) for peer in ROSTER]
                    pairs += [(m, p) for m in (ONESHOT, READY) for p in BASELINES]
                    for subject, peer in pairs:
                        row = dict(common)
                        for name, method in [('subject', subject), ('peer', peer)]:
                            data = source.get(method)
                            row[name] = method
                            row[name + '_policy'] = policy(method)
                            row[name + '_api_status'] = data.get('api_status', '') if data else ''
                            row[name + '_qualified'] = int(data is not None)
                            row[name + '_exclusion'] = public_text(error.get(method, ''))
                            if data is None:
                                continue
                            times = timing(c, method, regime)
                            row[name + '_median_us'] = median(times) * 1e6
                            for j, value in enumerate(times, 1):
                                row[f'{name}_us_{j}'] = value * 1e6
                                if method == READY:
                                    row[f'{name}_prep_us_{j}'] = data['prep'][j-1] * 1e6
                        p = match(c, subject, peer, regime)
                        if p:
                            row.update(ratio_of_medians=p['raw_ratio'], median_of_paired_ratios=p['paired_median'],
                                       paired_wins=p['wins'], seed1_paired_median=p['seeds'][0],
                                       seed2_paired_median=p['seeds'][1])
                            for j, value in enumerate(p['ratios'], 1):
                                row[f'peer_over_subject_{j}'] = value
                        for j in range(6):
                            row[f'round_{j+1}'] = j
                            row[f'seed_{j+1}'] = SEEDS[j//3]
                        writer.writerow(row)
