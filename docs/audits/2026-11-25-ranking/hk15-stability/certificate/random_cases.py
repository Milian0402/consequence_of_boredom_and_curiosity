"""Deterministic synthetic tests and an independent numeric oracle for the HK guard."""
import math
from random import Random
import struct


def _f32(x):
    try:
        return struct.unpack('=f', struct.pack('=f', x))[0]
    except OverflowError:
        return math.copysign(math.inf, x)


def _dot(x, y):
    acc = 0.0
    for u, v in zip(x, y):
        acc += u*v
    return acc


def _centers(size, count):
    return [(2*i+1)*size//(2*count) for i in range(count)]


def expected_policy(A, B):
    """Classify logical FP32 inputs. 0 is regular, 1 is classical, 2 is wide."""
    m, k, n = len(A), len(B), len(B[0])
    if max(n, k) > 3842:
        return 2
    a = [list(map(_f32, row)) for row in A]
    b = [list(map(_f32, row)) for row in B]
    if not m or not k or not n or any(len(row) != k for row in a) or any(len(row) != n for row in b):
        raise ValueError('Expected two nonempty conformable rectangular matrices')
    for matrix in (a, b):
        if any(x != 0 and not 2.0**-48 <= abs(x) <= 2.0**48 for row in matrix for x in row):
            return 2
    bt = list(zip(*b))
    for matrix, columns, depths in ((a, list(zip(*a)), a), (b, bt, bt)):
        row_max = [max(map(abs, row)) for row in matrix]
        peak = max(row_max)
        smallest = min(min(row_max), min(max(map(abs, col)) for col in columns))
        if smallest == 0 or 4*smallest < peak:
            return 1
        if any(4*sum(abs(x) >= peak/8 for x in depth) < 3*k for depth in depths):
            return 1
    if min(m, n) >= 16 and k >= 64:
        points = _centers(k, 64)
        rhs = [[b[p][j] for p in points] for j in _centers(n, 16)]
        norms = [_dot(y, y) for y in rhs]
        hits = 0
        for i in _centers(m, 16):
            x = [a[i][p] for p in points]
            norm = _dot(x, x)
            for y, rhs_norm in zip(rhs, norms):
                dot = _dot(x, y)
                if 4.0*dot*dot > norm*rhs_norm:
                    hits += 1
                    if hits == 4:
                        return 1
    return 0


def _dense(rng, rows, cols, scale):
    return [[rng.choice((-1, 1)) * scale * rng.randrange(128, 257)/256
             for _ in range(cols)] for _ in range(rows)]


def _pair(rng, m, n, k):
    return (_dense(rng, m, k, 2.0**rng.randint(-35, 35)),
            _dense(rng, k, n, 2.0**rng.randint(-35, 35)))


def _case(name, family, a, b):
    return name, family, a, b, expected_policy(a, b)


def cases():
    rng = Random(45167813)
    for q in range(18):
        m, n, k = (rng.randint(12, 96) for _ in range(3))
        a, b = _pair(rng, m, n, k)
        yield _case(f'dense-scale-{q:02}', 'random-dense', a, b)

    rng = Random(11482379)
    for q in range(8):
        m, n, k = rng.randint(15, 58), rng.randint(15, 58), 35+q
        for side in ('a', 'b'):
            a, b = _pair(rng, m, n, k)
            selected = rng.randrange(m if side == 'a' else n)
            matrix = a if side == 'a' else b
            peak = max(max(map(abs, row)) for row in matrix)
            ordered = rng.sample(range(k), k)
            needed = (3*k+3)//4
            for accepted in (False, True):
                live = needed if accepted else needed-1
                for pos, p in enumerate(ordered):
                    mag = peak if pos == 0 else peak*(.125 if pos < live else .0625)
                    value = -mag if p%2 else mag
                    if side == 'a':
                        a[selected][p] = value
                    else:
                        b[p][selected] = value
                # Each yielded pair is an independent fixture even if the caller materializes the generator.
                yield _case(f'support-{q}-{side}-{live}', 'random-depth-boundary',
                            [row[:] for row in a], [row[:] for row in b])

    rng = Random(29831753)
    for axis in ('a-row', 'a-col', 'b-row', 'b-col'):
        for q, factor in enumerate((.25-2.0**-26, .25, .25+2.0**-25)):
            m, n, k = 34, 37, 42
            a, b = _dense(rng, m, k, 1), _dense(rng, k, n, 1)
            mat = a if axis[0]=='a' else b
            mat[0][0] = 1
            if axis.endswith('row'):
                selected = rng.randrange(1, len(mat))
                mat[selected] = [rng.choice((-factor, factor)) for _ in mat[selected]]
            else:
                selected = rng.randrange(1, len(mat[0]))
                for row in mat:
                    row[selected] = rng.choice((-factor, factor))
            yield _case(f'axis-{axis}-{q}', 'random-axis-boundary', a, b)

    rng = Random(16590707)
    magnitudes = [float('nan'), math.inf, -math.inf, 2.0**-149, -2.0**-126,
                  2.0**-48-2.0**-72, 2.0**-48, 2.0**48, 2.0**48+2.0**25, -31.0]
    for q in range(30):
        m, n, k = rng.randint(35, 75), rng.randint(35, 75), rng.randint(67, 104)
        a, b = _dense(rng, m, k, 1), _dense(rng, k, n, 1)
        side = 'a' if q%2 else 'b'
        mat = a if side=='a' else b
        if q//10 == 0:
            i, j = 0, 0
        elif q//10 == 1:
            i, j = len(mat)-1, len(mat[0])-1
        else:
            row_samples = _centers(len(mat), 16 if side=='a' else 64)
            col_samples = _centers(len(mat[0]), 64 if side=='a' else 16)
            i = rng.choice(sorted(set(range(len(mat))) - set(row_samples)))
            j = rng.choice(sorted(set(range(len(mat[0]))) - set(col_samples)))
        mat[i][j] = magnitudes[q%10]
        if q%5 == 1:
            # A classical condition on the other operand must not mask a wide input.
            other = b if side=='a' else a
            other[0] = [0.0]*len(other[0])
        yield _case(f'spike-{side}-{q:02}', 'random-range-or-prominence-spike', a, b)

    rng = Random(18942731)
    for q in range(15):
        m, n, k = rng.randrange(13, 55), rng.randrange(15, 55), (63,64,65,79,96)[q%5]
        a, b = _pair(rng, m, n, k)
        direction = [rng.choice((-1., 1.)) for _ in range(k)]
        fraction = (.05, .30, .55, .80, 1.0)[q//3]
        for row in a:
            for p in range(k):
                if rng.random() < fraction:
                    row[p] = direction[p]*abs(row[p])
        for p, row in enumerate(b):
            for j in range(n):
                if rng.random() < fraction:
                    row[j] = direction[p]*abs(row[j])
        yield _case(f'coherence-{q:02}', 'random-sign-coherence', a, b)
