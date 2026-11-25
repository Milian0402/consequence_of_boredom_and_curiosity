"""Rank complete paired timings of the fixed native explicit handle panel."""
import csv
import math

PANEL_SHA = '2b4d6871262a4d038533b6cab819e819c3cff1eaf3ef606678b2ad6e30520ed5'
SEEDS = (1587984119, 384761239)
BASE = 'stream-incumbent'
HANDLE1,HANDLEP='mkl-handle-oneshot','mkl-handle-prepared'
DIRECT = ('openblas', 'mkl', 'mkl-jit', 'aocl', 'blasfeo', 'onednn-oneshot', 'libxsmm-dispatch')
RETAINED = ('onednn-cached-plan', 'libxsmm-retained')
ROSTER = ('openblas', 'mkl', 'mkl-jit', 'aocl', 'blasfeo', 'onednn-oneshot', 'onednn-cached-plan',
          'libxsmm-dispatch', 'libxsmm-retained', BASE, HANDLE1,HANDLEP)


def require(check, message):
    if not check:
        raise ValueError(message)


def rows(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


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
        'first': [float(r['first_s']) for r in warm],
        'first_qualification': [{'mode': r['mode'], 'seed': int(r['seed']), 'seconds': float(r['first_s'])} for r in valid],
        'max_error': max(float(r['error']) for r in records),
        'checked': sum(int(r['checked']) for r in records),
        'guards': sum(int(r['guards']) for r in records),
    }, []


