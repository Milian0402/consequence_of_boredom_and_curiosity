"""Replay the archived M5 experiments using only Python and the local COB checkout."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay(script, *args):
    return subprocess.check_output([sys.executable, '-B', str(HERE / script), *map(str, args)], text=True)


def independent_field(rows, summary):
    peers = ('accelerate', 'maxmulsk', 'kleidiai')
    times = {(r['shape'], r['method'], r['seed'], r['round']): float(r['seconds']) for r in rows}
    checked = 0
    for shape, result in summary.items():
        methods = result['methods']
        fastest = min((name for name in peers if methods[name]['accuracy_pass']),
                      key=lambda name: methods[name]['median_seconds'])
        for name in methods:
            if name.startswith('fp16-'):
                ratios = [times[shape, fastest, seed, rep] / seconds
                          for (idx, method, seed, rep), seconds in times.items()
                          if idx == shape and method == name]
                require(len(ratios) == 12 and max(ratios) <= 1, f'independent control at {shape}/{name}')
                checked += len(ratios)
    return checked


def main():
    manifest = (HERE / 'SHA256SUMS').read_text().splitlines()
    for line in manifest:
        digest, file = line.split('  ', 1)
        require(sha(HERE / file) == digest, f'manifest mismatch in {file}')

    print(replay('numerics/replay.py').strip())
    replay('numerics/exact-components.py')
    independent_pairs = 0
    for panel in ('results', 'higher'):
        directory = HERE / 'timing' / panel
        actual = json.loads(replay('timing/score.py', directory))
        expected = json.loads((directory / 'summary.json').read_bytes())
        require(actual == expected, f'{panel} recorded score mismatch')
        with (directory / 'time.csv').open(newline='') as stream:
            independent_pairs += independent_field(list(csv.DictReader(stream)), expected)
    require(independent_pairs == 288, 'incomplete comparison with independent competitors')
    print('Verified both six-layout timing fields, 864 warm observations and 288 comparisons with independent FP32 controls.')
    json.loads(replay('timing/audit.py'))

    ceiling = json.loads(replay('apple/ceiling/summarize.py'))
    require(ceiling == json.loads((HERE / 'apple/ceiling/summary.json').read_bytes()), 'instruction replay')
    identities = json.loads((HERE / 'apple/receipts/identity.json').read_bytes())
    for name, digest in identities['source'].items():
        require(sha(HERE / 'apple' / name) == digest, f'timed Apple source mismatch in {name}')
    protected = json.loads((HERE / 'apple/qa/native-m5.json').read_bytes())
    lib_digest = identities['measured_library_sha256']
    require(protected['library_sha256'] == lib_digest, 'protected check used different native library')
    for panel in ('results', 'higher'):
        identity = json.loads((HERE / 'timing' / panel / 'identity.json').read_bytes())
        require(identity['libraries']['apple-fp16.dylib'] == lib_digest, 'timed lower-plane library')
    with (HERE / 'numerics/data/identities.csv').open(newline='') as stream:
        high = next(row['composite_sha256'] for row in csv.DictReader(stream)
                    if row['implementation'] == 'sme-0')
    require(high == json.loads((HERE / 'timing/higher/identity.json').read_text())['qualified_dylib_sha256'],
            'higher precision timing and numerical studies used different libraries')
    with (HERE / 'apple/qa/integrated-corpus.csv').open(newline='') as stream:
        records = list(csv.DictReader(stream))
    ordinary = [r for r in records if r['category'] == 'normal']
    require(len(records) == 76 and len(ordinary) == 36, 'integrated corpus coverage')
    require(all(r['accepted'] == '1' and r['match_common_output_bits'] == '1' for r in ordinary),
            'integrated raw/three results differ from recorded common frontend')
    require(sum(m['cells'] for m in protected['modes']) == 27932, 'native protected output count')
    print('Verified six register-only pairs, ordinary integrated equality, protected-check receipts and native library identity.')


if __name__ == '__main__':
    main()
