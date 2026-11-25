"""Attribute the pinned HK15 parent and its children using paired, native calls."""
from array import array
import csv
import ctypes as ct
import json
import math
import os
from pathlib import Path
import random
from statistics import median
import sys
import time

build, out = map(Path, sys.argv[1:3])
smoke = len(sys.argv) > 3 and sys.argv[3] == 'smoke'
phases = ('hk_forms', 'hk_child', 'hk_output', 'fused_body', 'fused_pack_a', 'fused_pack_b',
          'fused_apply_short', 'fused_apply_long', 'fused_merge', 'fused_edges',
          'base_body', 'base_pack_a', 'base_pack_b', 'base_apply_short', 'base_apply_long')
P = ct.POINTER(ct.c_float)
U = ct.c_uint64 * len(phases)
arguments = [ct.c_int] * 3 + [P, ct.c_int, P, ct.c_int, P, ct.c_int]
modes = ('control', 'profile')


class Matrix:
    def __init__(self, rows, cols, pad):
        self.rows = rows
        self.cols = cols
        self.ld = cols + pad
        self.stride = 4 * self.ld
        self.memory = ct.create_string_buffer(rows * self.stride + 256 + 128)
        self.address = (ct.addressof(self.memory) + 127) & ~127
        self.pointer = ct.cast(self.address, P)
        self.postguard = b'\xa7' * 256
        ct.memmove(self.address + self.stride * rows, self.postguard, 256)

    def image(self, values, pad):
        return array('f', values).tobytes() + array('f', [pad] * (self.ld-self.cols)).tobytes()

    def fill(self, images):
        for i in range(self.rows):
            ct.memmove(self.address+i*self.stride, images[i % len(images)], self.stride)

    def check(self, images):
        for i in range(self.rows):
            if ct.string_at(self.address+i*self.stride, self.stride) != images[i % len(images)]:
                raise ValueError(f'Incorrect output, padding or mutated input at row {i}')
        if ct.string_at(self.address+self.rows*self.stride, 256) != self.postguard:
            raise ValueError('Trailing guard changed')


def setup(m, n, k, pad):
    a, b, c = Matrix(m,k,pad), Matrix(k,n,pad), Matrix(m,n,pad)
    # All products are multiples of 2^-8; the independent FP64 oracle is exact
    # for these bounded inputs. 8 row and 11 column patterns vary across K.
    rng = random.Random(173019)
    av = [[rng.randrange(-16,17)/16 for _ in range(k)] for _ in range(8)]
    bv = [[rng.randrange(-16,17)/16 for _ in range(11)] for _ in range(k)]
    ai = [a.image(row, math.nan) for row in av]
    bi = [b.image((row * ((n+10)//11))[:n], math.nan) for row in bv]
    ci = []
    for row in av:
        ref = [sum(row[p]*bv[p][j] for p in range(k)) for j in range(11)]
        ci.append(c.image((ref * ((n+10)//11))[:n], 123456.0))
    a.fill(ai)
    b.fill(bi)
    c.fill([c.image([math.nan]*n, 123456.0)])
    args = (m,n,k,a.pointer,a.ld,b.pointer,b.ld,c.pointer,c.ld)
    return args, (a,b,c), (ai,bi,ci)


def affinity():
    if not smoke:
        if len(os.sched_getaffinity(0)) != 1 or len(list(Path('/proc/self/task').iterdir())) != 1:
            raise ValueError('Expected exactly one CPU and one OS thread')


def library(route, mode):
    handle = ct.CDLL(str((build/f'{route}-{mode}.so').resolve()), mode=os.RTLD_LOCAL|os.RTLD_NOW)
    handle.cob_sgemm_rowmajor.argtypes = arguments
    handle.cob_sgemm_rowmajor.restype = None
    if mode == 'profile':
        handle.hk_profile_reset.argtypes = []
        handle.hk_profile_read.argtypes = [U, U]
        handle.hk_profile_reset.restype = handle.hk_profile_read.restype = None
    return handle


def exclusive(ns, whole):
    # Inputs/output and child calls are sibling spans. Child bodies contain their
    # respective pack/apply times. Each residual includes timer instrumentation.
    result = dict(outer_forms=ns['hk_forms'], outer_output=ns['hk_output'],
                  outer_other=whole - ns['hk_forms']-ns['hk_output']-ns['hk_child'],
                  child_setup_and_other=ns['hk_child']-ns['fused_body']-ns['base_body'])
    for body, children in [('fused', phases[4:10]), ('base', phases[11:])]:
        result.update((name, ns[name]) for name in children)
        result[body+'_other'] = ns[body+'_body']-sum(ns[k] for k in children)
    if min(result.values()) < 0 or sum(result.values()) != whole:
        raise ValueError('Non-additive nested timers')
    return result


def main():
    cases = [(288,432,432,3)] if smoke else [(1920,2880,2880,0), (2048,2880,3264,3)]
    rounds = 2 if smoke else 4
    handles = {(r,t):library(r,t) for r in ('parent960','legacy1024') for t in modes}
    affinity()
    summaries = []
    with (out/'raw.csv').open('w', newline='') as stream:
        fields = ['m','n','k','pad','route','pair','first','control_ns','profile_ns','profile_over_control']
        fields += [p+s for s in ('_ns','_calls') for p in phases]
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for case in cases:
            args, matrices, images = setup(*case)
            for route in ('legacy1024','parent960'):
                control, profiled = [handles[route,t] for t in modes]
                for handle in (control, profiled):
                    handle.cob_sgemm_rowmajor(*args)
                    matrices[2].check(images[2])
                records = []
                for i in range(rounds):
                    elapsed = {}
                    order = modes if i % 2 == 0 else modes[::-1]
                    for name in order:
                        lib = handles[route,name]
                        if name == 'profile':
                            lib.hk_profile_reset()
                        t = time.perf_counter_ns()
                        lib.cob_sgemm_rowmajor(*args)
                        elapsed[name] = time.perf_counter_ns()-t
                        if name == 'profile':
                            ns, counts = U(), U()
                            lib.hk_profile_read(ns, counts)
                        matrices[2].check(images[2])
                        affinity()
                    by_ns, by_count = dict(zip(phases,ns)), dict(zip(phases,counts))
                    expected_fused = 15 if route == 'parent960' else 0
                    if by_count['hk_child'] != 15 or by_count['fused_body'] != expected_fused or by_count['base_body'] != 15-expected_fused:
                        raise ValueError(f'Unexpected successful dispatch counts: {by_count}')
                    active = 'fused' if expected_fused else 'base'
                    if not all(by_count[active+s] for s in ('_pack_a','_pack_b','_apply_long')):
                        raise ValueError(f'Missing active panel measurements: {by_count}')
                    detail = exclusive(by_ns,elapsed['profile'])
                    row = dict(zip(('m','n','k','pad'),case), route=route, pair=i+1, first=order[0],
                               control_ns=elapsed['control'],profile_ns=elapsed['profile'],
                               profile_over_control=elapsed['profile']/elapsed['control'])
                    row.update((p+'_ns',by_ns[p]) for p in phases)
                    row.update((p+'_calls',by_count[p]) for p in phases)
                    writer.writerow(row)
                    stream.flush()
                    records.append((row,detail))
                overhead = median(r['profile_over_control'] for r,_ in records)
                summaries.append(dict(shape=case, route=route, pairs=rounds, profile_over_control=overhead,
                    within_five_percent=0.95 <= overhead <= 1.05,
                    profile_ratio_range=[min(r['profile_over_control'] for r,_ in records),max(r['profile_over_control'] for r,_ in records)],
                    median_control_ms=median(r['control_ns'] for r,_ in records)/1e6,
                    median_profile_ms=median(r['profile_ns'] for r,_ in records)/1e6,
                    median_exclusive_percent={key:median(100*d[key]/r['profile_ns'] for r,d in records)
                                               for key in records[0][1]}))
            for matrix, image in zip(matrices,images):
                matrix.check(image)
    (out/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')


if __name__ == '__main__':
    main()
