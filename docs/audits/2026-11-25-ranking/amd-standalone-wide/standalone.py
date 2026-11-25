"""Recompute secondary standalone COB comparisons from the two sealed AMD scores."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import geometric_mean, median

COB = 'stream-incumbent'
ONE_SHOT = ('openblas', 'mkl', 'mkl-jit', 'aocl', 'blasfeo', 'onednn-oneshot', 'libxsmm-dispatch')
RETAINED = ('onednn-cached-plan', 'libxsmm-retained')
ASSISTED = ('host-forward', 'host-split8', 'host-prefix8')
TIERS = {'one-shot': ONE_SHOT, 'retained-inclusive': ONE_SHOT + RETAINED}
TIMINGS = {'Warm': 'warm', 'First': 'first_random_per_warm_worker'}
SEEDS = (1047891527, 1936510481)
MODES = (('random', SEEDS[0]), ('random', SEEDS[1]), ('cancel', SEEDS[0]), ('scaled', SEEDS[1]))
SCORES = ('e5f9ea15ce5accc977360ef06cc47964f79c180b3b23bf3059a1d2bf7d6b14d9',
          '70cf6989253545fc7a5542feb4b482022bacd6fde2b0e5e1c1b94a1a4dacb297')
RELEASE = 'a30d71f0aaba4bcceb9ec5c530cffd63e6967cd6'
NAMES = {'openblas': 'OpenBLAS', 'mkl': 'oneMKL', 'mkl-jit': 'oneMKL direct-call',
         'aocl': 'AOCL', 'blasfeo': 'BLASFEO', 'onednn-oneshot': 'oneDNN one-shot',
         'onednn-cached-plan': 'oneDNN cached', 'libxsmm-dispatch': 'LIBXSMM dispatch',
         'libxsmm-retained': 'LIBXSMM retained'}


def load_jobs(root):
    jobs = []
    for replica, digest in enumerate(SCORES, 1):
        contents = (root / f'replica-{replica}' / 'score.json').read_bytes()
        if hashlib.sha256(contents).hexdigest() != digest:
            raise ValueError(f'Original score digest differs on replica {replica}')
        job = json.loads(contents)
        if job['receipt']['replica'] != replica or job['identity']['release'] != RELEASE:
            raise ValueError('Original replica or production release identity differs')
        jobs.append(job)
    return jobs


def evaluate(job):
    rows, selected = [], {}
    for cell in job['cells']:
        g = cell['layout']
        methods = cell['methods']
        excluded = cell['exclusions']
        for timing, key in TIMINGS.items():
            subject = methods[COB][key]
            leaders = {tier: min((m for m in roster if m in methods), key=lambda m: median(methods[m][key]))
                       for tier, roster in TIERS.items()}
            for peer in (*ONE_SHOT, *RETAINED, COB, *ASSISTED):
                peer_data = methods.get(peer)
                role = 'standalone' if peer == COB else 'vendor-assisted' if peer in ASSISTED else 'independent'
                row = dict(run=job['receipt']['run'], replica=job['receipt']['replica'], cpu=job['model'],
                           cob_isa=job['isa'], operation='single-thread row-major FP32 overwrite SGEMM',
                           source_release=job['identity']['release'], cob_binary_sha256=job['identity']['ordinary'][COB],
                           **g, timing=timing, standalone=COB, peer=peer, peer_role=role,
                           peer_plan='retained' if peer in RETAINED else 'one-shot',
                           peer_binary_sha256=job['identity']['ordinary'].get(peer, ''),
                           qualified=int(peer_data is not None), exclusion=' | '.join(excluded.get(peer, [])),
                           fastest_one_shot=leaders['one-shot'], fastest_including_retained=leaders['retained-inclusive'],
                           cob_median_us=median(subject)*1e6)
                for j, value in enumerate(subject):
                    row[f'round_{j+1}'] = j
                    row[f'seed_{j+1}'] = SEEDS[j//3]
                    row[f'cob_us_{j+1}'] = value*1e6
                if peer_data is not None:
                    times = peer_data[key]
                    ratios = [p / b for p, b in zip(times, subject, strict=True)]
                    seed1, seed2 = median(ratios[:3]), median(ratios[3:])
                    row.update(peer_median_us=median(times)*1e6, ratio_of_raw_medians=median(times)/median(subject),
                               median_of_paired_ratios=median(ratios), rounds_favoring_cob=sum(x > 1 for x in ratios),
                               seed1_paired_median=seed1, seed2_paired_median=seed2)
                    for j, (time, ratio) in enumerate(zip(times, ratios, strict=True), 1):
                        row[f'peer_us_{j}'] = time*1e6
                        row[f'peer_over_cob_{j}'] = ratio
                for method, name in ((COB, 'cob'), (peer, 'peer')):
                    if method not in methods:
                        continue
                    obs = methods[method]
                    row[f'{name}_max_absolute_error'] = obs['max_error']
                    if timing == 'First':
                        fixtures = {(f['mode'], f['seed']): f['seconds'] for f in obs['first_qualification']}
                        for mode, seed in MODES:
                            row[f'{name}_qualification_first_{mode}_{seed}_us'] = fixtures[mode, seed]*1e6
                for tier, winner in leaders.items():
                    if peer == winner:
                        selected[g['family'], timing, tier] = row
                        # The original score used warm-selected peers for its first-call summaries.
                        # This check uses the original peer-specific measurements in both regimes.
                        field = 'pairs' if timing == 'Warm' else 'cold_pairs'
                        if cell[field][COB][peer]['ratios'] != ratios:
                            raise ValueError('Paired observations differ from the original score')
                rows.append(row)
    return rows, selected


def table(lines, headings, values):
    lines.extend(['', '| ' + ' | '.join(headings) + ' |', '| ' + ' | '.join('---' for _ in headings) + ' |'])
    lines.extend('| ' + ' | '.join(str(x) for x in row) + ' |' for row in values)


def summary(selected, families, regime, tier):
    rows = [selected[f,regime,tier] for f in families]
    ratios = [r['median_of_paired_ratios'] for r in rows]
    return [f"{geometric_mean(ratios):.4f}×", f"{sum(r['ratio_of_raw_medians']>1 for r in rows)}/{len(rows)}",
            f"{sum(r>1 for r in ratios)}/{len(rows)}", f"{sum(r['rounds_favoring_cob'] for r in rows)}/{6*len(rows)}",
            f"{geometric_mean(r['seed1_paired_median'] for r in rows):.4f}×",
            f"{geometric_mean(r['seed2_paired_median'] for r in rows):.4f}×"]


def leader(row):
    return f"{NAMES[row['peer']]} {row['peer_median_us']:.2f} µs"


def measures(row):
    return f"{row['ratio_of_raw_medians']:.4f} / {row['median_of_paired_ratios']:.4f}"


def report(jobs, results, csv_name):
    lines = ['# Standalone COB on two AMD EPYC processors', '',
             'On six fresh wide SGEMMs, standalone COB had the lowest warm median in the qualified '
             'field on five shapes on the EPYC 7763 and zero on the EPYC 9V45. '
             'For the first call after initialization, it had the lower raw median on six and five shapes, '
             'respectively. Its paired first-call median ratio favored COB on all six shapes on each CPU.',
             '', 'The two jobs measured different processors and selected different COB kernels. These are '
             'exploratory results from an existing panel registered for an Intel vendor-assisted experiment. '
             'Each AMD assignment provides one job on its CPU model.',
             '', '## Six fresh shapes', '',
             'Each aggregate is the geometric mean of six equally weighted layout ratios. The table counts '
             'wins using two calculations. “Raw” compares each method’s median time over six observations. '
             '“Paired” divides peer time by COB time within the same seed and round, then takes the median. '
             'Ratios above one favor COB. The two seed columns repeat the paired calculation using the '
             'three rounds in each seed block.']
    values = []
    for job, selected in zip(jobs, results, strict=True):
        fresh = [c['layout']['family'] for c in job['cells'] if c['layout']['family'].startswith('fresh-')]
        for timing in TIMINGS:
            for tier in TIERS:
                values.append([job['model'].split()[2], timing, tier, *summary(selected, fresh, timing, tier)])
    table(lines, ['CPU', 'Call', 'Qualified independent field', 'Paired geomean', 'Raw case wins',
                  'Paired case wins', 'Paired round wins', 'Seed 1', 'Seed 2'], values)
    lines.extend(['', 'The one-shot field comprised OpenBLAS, sequential oneMKL, its JIT-capable direct-call '
                  'adapter, AOCL, BLASFEO, strict FP32 oneDNN one-shot and LIBXSMM per-call dispatch. '
                  'The retained-inclusive field also admitted cached-plan oneDNN and LIBXSMM with a retained '
                  'kernel pointer. For each CPU, layout, regime and tier, the peer was the method with the '
                  'lowest raw median in that regime. First-call comparisons therefore use first-call leaders.',
                  '', 'Workers were pinned to one logical CPU. Each warm observation averages 32 completed '
                  'public multiplications with resident inputs and output. A first observation times the '
                  'first complete multiplication in a fresh process after library initialization, including '
                  'any on-demand plan or kernel creation inside that call. Both regimes use the same six '
                  f'random workers, three apiece for seeds {SEEDS[0]} and {SEEDS[1]}.'])
    for job, selected in zip(jobs, results, strict=True):
        families = [c['layout']['family'] for c in job['cells']]
        isa = {'avx512':'AVX-512','avx2':'AVX2'}[job['isa']]
        lines.extend(['', f"## {job['model']} · {isa}", '',
                      f"Each of the eleven native traces recorded one public call and one {isa} kernel call. "
                      f"The timed ordinary COB worker had SHA-256 `{job['identity']['ordinary'][COB]}`. "
                      f"[Native job]({job['receipt']['job_url']})"])
        for regime in TIMINGS:
            lines.extend(['', f"### {regime} calls", ''])
            two = any(selected[f,regime,'one-shot']['peer'] != selected[f,regime,'retained-inclusive']['peer']
                      for f in families)
            lines.append('Each ratio cell gives raw / paired. ' +
                         ('The second peer column also admits retained plans.' if two else
                          'Both independent fields selected the same leader on every layout.'))
            values = []
            for f in families:
                one, both = [selected[f,regime,tier] for tier in TIERS]
                row = [f, f"{one['m']} × {one['n']} × {one['k']}", f"{one['cob_median_us']:.2f}", leader(one), measures(one)]
                if two:
                    row.extend([leader(both), measures(both)])
                values.append(row)
            headers = ['Layout', 'M × N × K', 'COB µs', 'Fastest one-shot' if two else 'Fastest independent', 'Raw / paired']
            if two:
                headers.extend(['Fastest incl. retained', 'Raw / paired'])
            table(lines, headers, values)
        if job['isa'] == 'avx512':
            r = selected['fresh-high-depth','First','one-shot']
            lines.extend(['', f"At 112 × 3581 × 337, AOCL’s first-call median of {r['peer_median_us']:.2f} µs "
                          f"was lower than COB’s {r['cob_median_us']:.2f} µs. COB finished first in "
                          f"{r['rounds_favoring_cob']}/6 matched rounds. The ratio of raw medians was "
                          f"{r['ratio_of_raw_medians']:.5f}× and the median paired ratio was "
                          f"{r['median_of_paired_ratios']:.5f}×. This layout accounts for the "
                          'difference between five fresh first-call wins by raw median and six by paired median.'])
    lines.extend(['', '## Coverage and provenance', '',
                  'Both jobs had all nine independent methods qualified on all six fresh shapes. '
                  'On the historical 137 × 2179 × 347 layout, BLASFEO exited with signal 11 on both jobs '
                  'and its time is missing. The other independent comparisons on that layout use eight '
                  'qualified methods. The “control-cropped” labels identify the original experiment’s two '
                  'cropped views. COB executed its ordinary standalone path there.',
                  '', 'The subject was the installed single-threaded `cob_sgemm_rowmajor` overwrite operation '
                  'from the same tuned source release on both machines. The original score replays check '
                  'ordinary binary identity, native dispatch, FP64-reference accuracy, input protection, '
                  'output padding and guards. The three oneMKL-assisted wrappers ran as forwarding controls '
                  'on AMD. Their measurements are available in the CSV, outside the independent field. '
                  f'[Standalone source at {RELEASE[:8]}](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/tree/{RELEASE})',
                  '', 'The CSV gives all six round/seed IDs, COB and peer times, ratios, binary identities, '
                  'qualification status and independent leaders for each registered method, layout and regime. '
                  'It also preserves the four dedicated first-call fixture observations, which use distinct '
                  'qualification workers and are excluded from the first-call ranking. '
                  f'[All comparisons]({csv_name})'])
    return '\n'.join(lines)+'\n'


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf8', newline='') as out:
        writer = csv.DictWriter(out, fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evidence', type=Path, help='extracted original prefix-eight packet')
    parser.add_argument('report', type=Path)
    parser.add_argument('comparisons', type=Path)
    args = parser.parse_args()
    jobs = load_jobs(args.evidence)
    rows = []
    selected = []
    for job in jobs:
        measurements, leaders = evaluate(job)
        rows.extend(measurements)
        selected.append(leaders)
    args.report.write_text(report(jobs, selected, args.comparisons.name), encoding='utf8')
    write_csv(args.comparisons, rows)


if __name__ == '__main__':
    main()
