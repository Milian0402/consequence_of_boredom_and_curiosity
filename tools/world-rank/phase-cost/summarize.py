"""Validate and summarize paired wall and phase measurements from phase-worker."""
import argparse
import csv
import math
from pathlib import Path
from statistics import median

CASES = (
    (84, 353, 447, 3),
    (131, 577, 661, 5),
    (112, 1744, 357, 0),
    (216, 927, 619, 1),
    (65, 2161, 529, 7),
    (128, 4096, 256, 0),
)
KEY = ('m', 'n', 'k', 'pad')
PHASES = ('pack_a', 'pack_b', 'apply')
COUNTS = tuple(phase + '_calls' for phase in PHASES)
DURATIONS = ('control_ns', 'profile_ns', *(phase + '_ns' for phase in PHASES))
ERRORS = ('control_max_abs', 'profile_max_abs')
FIELDS = (*KEY, 'round', 'first', 'calls', *DURATIONS, *COUNTS, 'route_calls', 'nr', *ERRORS)
ROUNDS = 6
CALLS = 3


def load(path):
    cases = {case: {} for case in CASES}
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(FIELDS):
            raise ValueError(f'{path}: unexpected CSV header {reader.fieldnames}')
        for raw in reader:
            try:
                if None in raw or any(value is None for value in raw.values()):
                    raise ValueError('incorrect number of fields')
                row = {key: int(value) for key, value in raw.items()
                       if key not in ('first', *ERRORS)}
                for key in ERRORS:
                    row[key] = float(raw[key])
                    if not math.isfinite(row[key]) or not 0 <= row[key] < 0.002:
                        raise ValueError(f'{key} is outside [0, 0.002)')
                case = tuple(row[key] for key in KEY)
                if case not in cases:
                    raise ValueError(f'unexpected case {case}')
                number = row['round']
                if number not in range(1, ROUNDS + 1) or number in cases[case]:
                    raise ValueError(f'invalid or duplicate round {number} for {case}')
                if raw['first'] != ('control' if number % 2 else 'profile'):
                    raise ValueError('counterbalanced method order differs from protocol')
                if row['calls'] != CALLS or row['route_calls'] != CALLS:
                    raise ValueError(f'each batch must cover {CALLS} calls on the measured route')
                if any(row[key] <= 0 for key in (*DURATIONS, *COUNTS, 'nr')):
                    raise ValueError('nonpositive duration, phase count or tile width')
                nr = row['nr']
                if nr not in (16, 32):
                    raise ValueError('unexpected x86 blocked tile width')
                depth_panels = CALLS * ((row['k'] + 255) // 256)
                column_panels = (row['n'] + 8 * nr - 1) // (8 * nr)
                if tuple(row[key] for key in COUNTS) != (depth_panels, depth_panels * column_panels, depth_panels * column_panels):
                    raise ValueError('phase counts do not match the 256-deep, eight-tile panel schedule')
                used = sum(row[key + '_ns'] for key in PHASES)
                if used > row['profile_ns']:
                    raise ValueError('exclusive phase timers exceed profiled wall time')
                row['remaining_ns'] = row['profile_ns'] - used
                cases[case][number] = row
            except (TypeError, ValueError) as error:
                raise ValueError(f'{path}:{reader.line_num}: {error}') from error
    for case, rounds in cases.items():
        if len(rounds) != ROUNDS:
            raise ValueError(f'{path}: expected {ROUNDS} distinct rounds for {case}, got {sorted(rounds)}')
        for key in (*COUNTS, 'nr'):
            if len({row[key] for row in rounds.values()}) != 1:
                raise ValueError(f'{path}: {key} changes across repeats of {case}')
    return cases


def summarize(case, rounds):
    rows = list(rounds.values())
    result = dict(zip(KEY, case))
    result.update(pairs=len(rows), calls_per_method_per_pair=CALLS, nr=rows[0]['nr'])
    for key in COUNTS:
        result[key + '_per_gemm'] = rows[0][key] / CALLS
    for method in ('control', 'profile'):
        result[method + '_median_ns_per_gemm'] = median(row[method + '_ns'] / CALLS for row in rows)
        result[method + '_max_abs_error'] = max(row[method + '_max_abs'] for row in rows)
    ratios = [row['profile_ns'] / row['control_ns'] for row in rows]
    result['median_profile_over_control'] = median(ratios)
    result['min_profile_over_control'] = min(ratios)
    result['max_profile_over_control'] = max(ratios)
    for phase in (*PHASES, 'remaining'):
        result[phase + '_median_pct'] = median(100 * row[phase + '_ns'] / row['profile_ns'] for row in rows)
    # The denominator is the wall time of the very same instrumented batch.
    # Combining before taking medians preserves the A/B relationship.
    result['packing_combined_median_pct'] = median(
        100 * (row['pack_a_ns'] + row['pack_b_ns']) / row['profile_ns'] for row in rows)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('raw_csv', type=Path)
    parser.add_argument('summary_csv', type=Path)
    args = parser.parse_args()
    if args.raw_csv.resolve() == args.summary_csv.resolve():
        parser.error('input and output paths must differ')
    try:
        cases = load(args.raw_csv)
        summaries = [summarize(case, rounds) for case, rounds in cases.items()]
        with args.summary_csv.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
            writer.writeheader()
            for row in summaries:
                writer.writerow({key: format(value, '.12g') if isinstance(value, float) else value
                                 for key, value in row.items()})
    except (OSError, ValueError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
