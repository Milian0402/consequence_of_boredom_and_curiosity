"""Verify the published measurements with the Python 3 standard library."""
import csv
import hashlib
import itertools
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent
ARCHES = ('x86_64', 'aarch64')
SEEDS = (73194653, 28417069)
TRIALS = tuple(itertools.product((1, 2), (1, 2, 3)))
FIXTURES = {('random', 1, SEEDS[0]), ('random', 2, SEEDS[1]),
            ('scaled', 0, SEEDS[0]), ('cancel', 0, SEEDS[0])}
PEERS = {'direct', 'external_strassen', 'external_winograd'}


def require(condition, evidence):
    if not condition:
        raise ValueError(f'Failed check: {evidence}')


def read_table(name):
    with (ROOT / (name + '.csv')).open(newline='') as stream:
        return list(csv.DictReader(stream))


def index(rows, keys):
    result = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        require(key not in result, ('duplicate', key))
        result[key] = row
    return result


def check_files():
    entries = (ROOT / 'SHA256SUMS').read_text().splitlines()
    seen = set()
    for line in entries:
        digest, name = line.split('  ')
        require(re.fullmatch(r'[A-Za-z0-9_.-]+', name) and name not in seen, name)
        require(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name)
        seen.add(name)
    needed = {'README.md', 'verify.py', 'provenance.json', 'panel.csv', 'builds.csv',
              'methods.csv', 'coverage.csv', 'summary.csv', 'timings.csv', 'accuracy.csv',
              'release-routes.csv', 'prototype-routes.csv', 'standalone-routes.csv',
              'fastmmw-routes.csv', 'hybrid-leaves.csv'}
    require(seen == needed, 'checksum manifest membership')


def check_routes(methods, panel):
    shape_to_family = {tuple(r[k] for k in ('m','n','k','pad')): name for (name,), r in panel.items()}
    for filename, roles in [('release-routes', {'original_release', 'production_winograd'}),
                            ('prototype-routes', {'public_winograd'}),
                            ('standalone-routes', {'standalone'}),
                            ('fastmmw-routes', {'external_winograd'}),
                            ('hybrid-leaves', {'worker_winograd'})]:
        expected = set()
        hybrid = filename == 'hybrid-leaves'
        for (arch, method), profile in methods.items():
            if profile['role'] in roles and profile['status'] == 'included':
                for (family,) in panel:
                    for fixture in FIXTURES if hybrid else {('route', 0, 0)}:
                        expected.add((arch, method, family, fixture))
        seen = set()
        for row in read_table(filename):
            arch, method = row['arch'], row['implementation']
            shape = tuple(row[k] for k in ('m','n','k','pad'))
            m, n, k, _ = map(int, shape)
            family = shape_to_family[shape]
            fixture = (row['mode'], int(row['block']), int(row['seed'])) if hybrid else ('route', 0, 0)
            key = (arch, method, family, fixture)
            require(key in expected and key not in seen and row['passed'] == '1', key)
            seen.add(key)
            if hybrid:
                count = 49 + (m % 4 != 0) + (n % 4 != 0)
                require(int(row['checked']) == m*n and int(row['leaf_calls']) == int(row['expected_leaf_calls']) == count, key)
                continue
            require(int(row['output_elements']) == m*n and all(row[x] == '1' for x in
                    ('timed_binary_exact','trace_binary_exact','timed_inputs_unchanged','trace_inputs_unchanged')), key)
            if filename == 'standalone-routes':
                require(row['observed'] == row['expected'], key)
            elif filename == 'fastmmw-routes':
                cutoff = int(method.split('-')[-1])
                levels = 0
                while min(m, n, k) > cutoff * (2 ** levels):
                    levels += 1
                require(int(row['leaf_calls']) == 7**levels and int(row['padded']) == int(any(d % 2**levels for d in (m,n,k))), key)
            else:
                align = method.endswith('-aligned-two')
                tiles = (64,64,32) if align else (4,4,4)
                cm, cn, ck = [d - d % tile for d, tile in zip((m,n,k), tiles)]
                dims = tuple(int(row['core_'+d]) for d in 'mnk')
                areas = [name for name, needed in [('depth', align and k != ck), ('right', n != cn), ('bottom', m != cm)] if needed]
                require(dims == (cm,cn,ck) and row['edge_regions'] == '|'.join(areas) and
                        int(row['beta0']) == int(row['expected_beta0']) == 49 + (m != cm) + (n != cn) and
                        int(row['beta1']) == int(row['expected_beta1']) == int(align and k != ck) and
                        row['core_calls'] == '49' and row['threads'] == row['internal_core_shapes_ok'] == row['edge_pointers_ok'] == '1', key)
                if filename == 'release-routes':
                    require(row['linked_worker_exact'] == row['linked_worker_inputs_unchanged'] == '1' and int(row['linked_worker_outputs_checked']) == m*n, key)
        require(seen == expected, (filename, 'route coverage'))


def main():
    check_files()
    meta = json.loads((ROOT/'provenance.json').read_text())
    require(tuple(meta['seeds']) == SEEDS and meta['maximum_absolute_error'] == .002, 'protocol')
    require(hashlib.sha256((ROOT/'panel.csv').read_bytes()).hexdigest() == meta['panel_sha256'], 'registered panel')
    panel = index(read_table('panel'), ['family'])
    methods = index(read_table('methods'), ['arch','implementation'])
    coverage = index(read_table('coverage'), ['arch','family','implementation'])
    summary = index(read_table('summary'), ['arch','family'])
    eligible = {key for key,r in methods.items() if r['status'] == 'included'}
    expect = {(arch,family,method) for arch,method in eligible for family, in panel}
    require(set(coverage) == expect and set(summary) == {(a,f) for a,f,_ in expect}, 'roster')
    builds = index(read_table('builds'), ['arch','implementation'])
    require(set(builds) == {key for key in eligible if methods[key]['role'] in ('production_winograd','original_release')}, 'compiled workers')
    for (_,method), row in builds.items():
        commit = meta['previous_revision'] if method == 'release-previous' else meta['production_revision']
        require(row['commit'] == commit and row['fixture_commit'] == meta['fixture_revision'] and
                row['before_cmake_archive_sha256'] == row['cmake_archive_sha256'] and
                all(re.fullmatch('[a-f0-9]{64}',row[k]) for k in ('cmake_archive_sha256','make_object_sha256','worker_sha256')), method)
    check_routes(methods, panel)
    timings = defaultdict(dict)
    fixtures = defaultdict(set)
    affinity = defaultdict(set)
    total = 0
    for r in read_table('accuracy'):
        a,f,n = r['arch'],r['family'],r['implementation']
        shape = panel[f,]
        key = a,f,n
        rec = r['mode'],int(r['block']),int(r['seed'])
        require(key in expect and rec in FIXTURES and rec not in fixtures[key], (key,rec))
        fixtures[key].add(rec)
        cells = int(r['checked'])
        total += cells
        affinity[a].add(r['cpu'])
        err = float(r['max_abs_error'])
        lo,hi = float(r['signed_min_error']),float(r['signed_max_error'])
        require(cells == int(shape['m'])*int(shape['n']) and 0 <= err <= .002 and lo <= hi and
                math.isclose(max(abs(lo),abs(hi)),err,rel_tol=1e-10) and math.isfinite(float(r['checksum'])) and
                r['nonfinite'] == r['guard_changes'] == r['input_changes'] == '0' and r['verified'] == r['threads'] == '1' and
                0 < float(r['first_sgemm_seconds']) < math.inf and (not n.startswith('onednn-') or r['onednn_impl']), (key,rec))
    roster = {a:[n for (arch,n),profile in methods.items() if arch==a and profile['status']=='included'] for a in ARCHES}
    family_order = [f for f, in panel]
    for r in read_table('timings'):
        a,f,n = r['arch'],r['family'],r['implementation']
        key = a,f,n
        b,j = int(r['block']),int(r['round'])
        sec = float(r['seconds'])
        require(key in expect and (b,j) in TRIALS and (b,j) not in timings[key] and
                int(r['seed']) == SEEDS[b-1] and 0 < sec < math.inf and r['calls'] == r['threads'] == r['resident'] == '1', (key,b,j))
        names = roster[a]
        rotation = (family_order.index(f)+j-1) % len(names)
        order = names[rotation:] + names[:rotation]
        if (b+j) % 2:
            order.reverse()
        require(int(r['position']) == order.index(n)+1, (key,'counterbalanced position'))
        timings[key][b,j] = sec
        affinity[a].add(r['cpu'])
    require(all(len(affinity[a]) == 1 for a in ARCHES), 'CPU affinity')
    require(set(timings) == set(fixtures) == expect, 'complete timed and numerical cells')
    for key in expect:
        r = coverage[key]
        require(set(timings[key]) == set(TRIALS) and fixtures[key] == FIXTURES and
                r['warm_pairs'] == '6' and r['verified_cases'] == '4' and r['qualified'] == '1', key)
    for a in ARCHES:
        new_odds = []
        fixed = meta['hosts'][a]['fixed']
        for f, in panel:
            r = summary[a,f]
            times = {n:median(timings[a,f,n].values()) for n in roster[a]}
            def best(names):
                return min(names, key=lambda n:(times[n],n))
            oneshot = [n for n in times if n != 'onednn-cached-plan']
            independent = best(n for n in oneshot if methods[a,n]['role'] in PEERS)
            release = best(n for n in oneshot if methods[a,n]['role']=='production_winograd')
            winner = best(oneshot)
            ratios = [timings[a,f,independent][trial]/timings[a,f,fixed][trial] for trial in TRIALS]
            wins = sum(x>1 for x in ratios)
            require(r['field_winner']==winner and r['fastest_release']==release and r['fixed']==fixed and
                    r['independent']==independent and r['winner_if_cached_included']==best(times) and
                    r['paired_speedup']==f'{median(ratios):.7f}' and int(r['wins'])==wins and r['pairs']=='6', (a,f,'summary'))
            for method,col in [(winner,'field_median_ms'),(release,'release_median_ms'),(fixed,'fixed_median_ms'),(independent,'independent_median_ms')]:
                require(r[col]==f'{times[method]*1000:.4f}', (a,f,col))
            if f in ('fresh-deep-odd','fresh-wide-odd'):
                new_odds.append(median(ratios)>1 and wins>=5)
        require(len(new_odds)==2 and all(new_odds)==meta['hosts'][a]['meets_two_odd_rule'], (a,'fixed-policy rule'))
    print(f'Verified {len(builds)} release builds, {len(expect)} method/layout combinations, '
          f'{sum(map(len,timings.values()))} timings, {sum(map(len,fixtures.values()))} numerical fixtures and {total:,} checked output cells.')


if __name__ == '__main__':
    main()
