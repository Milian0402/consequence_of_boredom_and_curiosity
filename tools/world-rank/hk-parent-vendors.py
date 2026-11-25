"""Verify external Winograd qualification and record the two COB hybrids' actual leaf calls."""
import argparse
import csv
import json
from pathlib import Path

DIMS = ('m', 'n', 'k', 'pad')


def read(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def dims(row):
    return tuple(int(row[key]) for key in DIMS)


def qualify(directory, name):
    expected = {'fastmmw-768': 49, 'fastmmw-1536': 7}[name]
    panel = {dims(row) for row in read(directory/'panel.csv')}
    rows = [row for row in read(directory/'fastmmw-routes.csv') if row['implementation'] == name]
    if len(rows) != len(panel) or {dims(row) for row in rows} != panel:
        raise ValueError(f'Incomplete route evidence for {name}')
    for row in rows:
        if (row['passed'], row['leaf'], int(row['leaf_calls']), int(row['padded'])) != ('1', 'openblas', expected, 1):
            raise ValueError(f'Failed {name} qualification for {dims(row)}')


def hybrids(directory, native):
    ranking = json.loads((directory/'ranking.json').read_text())
    reference = {dims(case) for case in read(directory/'panel.csv')}
    first, second = ranking['seeds']
    required = {(shape, mode, block, seed) for shape in reference
                for mode, block, seed in [('random',1,first),('random',2,second),
                                          ('cancel',0,first),('scaled',0,first)]}
    checks = read(directory/'accuracy.csv')
    fields = ['implementation', 'leaf', 'loaded_library', *DIMS, 'mode', 'block', 'seed',
              'leaf_calls', 'expected_leaf_calls', 'checked', 'passed']
    invalid = []
    with (directory/'hybrid-leaves.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for leaf in ('openblas', native):
            name = 'hybrid-' + leaf
            variant, library = ranking['implementations'][name]
            if variant != 'hybrid-next' or Path(library).name != leaf+'.so':
                raise ValueError(f'Unexpected actual hybrid or loaded DSO for {name}: {variant} {library}')
            seen = set()
            for row in checks:
                if row['implementation'] != name:
                    continue
                shape = dims(row)
                m, n, _, _ = shape
                # All three frozen shapes select two levels; cropped right/bottom
                # fringes make one additional call per nonempty strip.
                expected = 49 + int(m % 4 != 0) + int(n % 4 != 0)
                key = shape, row['mode'], int(row['block']), int(row['seed'])
                detail = dict(part.split('=',1) for part in row['detail'].split(',') if '=' in part)
                calls = int(detail['hybrid_vendor_calls'])
                passed = (key in required and key not in seen and calls == expected and
                          row['verified'] == '1' and row['threads'] == '1' and int(row['checked']) == m*n)
                record = {field:row[field] for field in (*DIMS,'mode','block','seed','checked')}
                writer.writerow(dict(record, implementation=name, leaf=leaf, loaded_library=library,
                                     leaf_calls=calls, expected_leaf_calls=expected, passed=int(passed)))
                seen.add(key)
                if not passed:
                    invalid.append((name,key))
            if seen != required:
                invalid.append((name,'missing/extra accuracy cases'))
    if invalid:
        raise ValueError(f'Incomplete hybrid-leaf proof: {invalid}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('qualified','hybrids'))
    parser.add_argument('directory', type=Path)
    parser.add_argument('name')
    args = parser.parse_args()
    try:
        (qualify if args.action == 'qualified' else hybrids)(args.directory,args.name)
    except (KeyError, ValueError, OSError) as error:
        parser.exit(1,f'{error}\n')
