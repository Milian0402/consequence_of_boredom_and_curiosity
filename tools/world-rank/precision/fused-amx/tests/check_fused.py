"""Check the real packer, and when the OS grants AMX, the real whole-call kernel."""
import argparse
import ctypes as ct
import json
import math
from pathlib import Path
import platform
from random import Random
import struct

F = ct.c_float
U16 = ct.c_uint16
U32 = ct.c_uint32
FP = ct.POINTER(F)
WP = ct.POINTER(U16)
I = ct.c_int
S = ct.c_size_t
GEMM = [I, I, I, FP, I, FP, I, FP, I]
OUTPUT_GUARD = 0x7FC5A61F
INPUT_GUARD = 0x7FC45BCD
WORD_GUARD = 0xA5D3


def f32(x):
    return struct.unpack('=f', struct.pack('=f', x))[0]


def from_bits(bits):
    return struct.unpack('=f', struct.pack('=I', bits))[0]


def rne(x):
    # Round on the mathematical BF16 lattice. Python integer round uses ties-to-even.
    if x == 0:
        return 0x8000 if math.copysign(1, x) < 0 else 0
    exponent = max(-126, math.frexp(abs(x))[1] - 1)
    significand = round(math.ldexp(abs(x), 7-exponent))
    rounded = math.copysign(math.ldexp(significand, exponent-7), x)
    return struct.unpack('=I', struct.pack('=f', rounded))[0] >> 16


def split(x):
    components = []
    for _ in range(3):
        component = rne(x)
        components.append(component)
        x = f32(x - from_bits(component << 16))
    assert x == 0, 'the finite binary32 input was not exactly decomposed'
    return components


def bind(lib, name, argtypes, result):
    fn = getattr(lib, name)
    fn.argtypes, fn.restype = argtypes, result
    return fn


def view(rows, cols, pad, offset, bits):
    ld = cols+pad
    storage = (U32 * (offset+rows*ld+19))()
    for t in range(len(storage)):
        storage[t] = bits
    pointer = ct.cast(ct.byref(storage, 4*offset), FP)
    return storage, pointer, ld


def pack_checks(lib):
    size = bind(lib, 'amx_fused6_pack_words', [I, I, I], S)
    pack = bind(lib, 'amx_fused6_pack', [I, I, I, FP, I, WP], I)
    special = [0, 0x80000000, 0x30800000, 0xb0800000, 0x4e800000, 0xce800000,
               0x3f807fff, 0x3f808000, 0x3f808001, 0x3f817fff, 0x3f818000,
               0x3f818001, 0xbf808000, 0xbf818000, 0x3fc36c11, 0xbf912345]
    assert [rne(from_bits(raw)) for raw in special[6:14]] == [
        0x3f80, 0x3f80, 0x3f81, 0x3f81, 0x3f82, 0x3f82, 0xbf80, 0xbf82]
    assert all(size(*invalid)==0 for invalid in ((3,2,2),(0,0,1),(1,1,-1),(1,2**31-1,2**31-1)))
    checked = 0
    flushed_cases = 0
    for right in (0, 1):
        for rows, cols, pad, offset in ((1, 1, 0, 0), (13, 19, 2, 3), (32, 32, 0, 0),
                                        (35, 67, 11, 7), (66, 41, 5, 2)):
            storage, ptr, ld = view(rows, cols, pad, offset, INPUT_GUARD)
            rng = Random(197030 + right*83 + rows)
            for i in range(rows):
                for j in range(cols):
                    q = i*cols+j
                    if q < len(special):
                        raw = special[q]
                    else:
                        raw = (rng.getrandbits(1)<<31) | (rng.randrange(98, 157)<<23) | rng.getrandbits(23)
                    storage[offset+i*ld+j] = raw
            original = bytes(storage)
            k = rows if right else cols
            panel_count = math.ceil((cols if right else rows)/16)
            depth_count = math.ceil(k/32)
            plane = panel_count*depth_count*512
            words = size(right, rows, cols)
            assert words == 3*plane, (right, rows, cols, words)
            destination = (U16 * (words+100))()
            for q in range(len(destination)):
                destination[q] = WORD_GUARD
            skip = 32+(-ct.addressof(destination)//2)%32
            dst = ct.cast(ct.byref(destination, 2*skip), WP)
            assert ct.addressof(dst.contents)%64 == 0
            assert pack(right, rows, cols, ptr, ld, dst) == 0
            assert bytes(storage) == original
            assert all(v == WORD_GUARD for v in (*destination[:skip], *destination[skip+words:]))
            # Walk the destination in its documented stream order, including zeros at every tail.
            q = 0
            for part in range(3):
                for panel in range(panel_count):
                    for depth in range(depth_count):
                        for t in range(512):
                            if right:
                                pair, place = divmod(t, 32)
                                i, j = 32*depth+2*pair+place%2, 16*panel+place//2
                            else:
                                i, j = 16*panel+t//32, 32*depth+t%32
                            expected = split(ptr[i*ld+j])[part] if i < rows and j < cols else 0
                            assert dst[q] == expected, (right, rows, cols, part, i, j, hex(dst[q]), hex(expected))
                            q += 1
            assert q == words
            checked += q
        invalid = (0x307fffff, 0xb07fffff, 0x4e800001, 0xce800001, 1, 0x80000001,
                   0x007fffff, 0x00800000, 0x7f7fffff, 0x7f800000, 0xff800000,
                   0x7fc10000, 0x7f800001, 0xffc11111)
        rows, cols = 7, 13
        storage, ptr, ld = view(rows, cols, 5, 3, INPUT_GUARD)
        for i in range(rows):
            for j in range(cols):
                ptr[i*ld+j] = 1
        out = (U16 * size(right, rows, cols))()
        for raw in invalid:
            storage[3+(rows-1)*ld+cols-1] = raw
            original = bytes(storage)
            assert pack(right, rows, cols, ptr, ld, out) == 1, (right, hex(raw))
            assert bytes(storage) == original
        old = bind(lib, 'test_enable_ftz_daz', [], I)()
        if old >= 0:
            try:
                for raw, expected in ((1, 1), (0x80000001, 1), (0x30800001, 0), (0x80000000, 0)):
                    storage[3+(rows-1)*ld+cols-1] = raw
                    assert pack(right, rows, cols, ptr, ld, out) == expected, (right, hex(raw), 'FTZ/DAZ')
                    flushed_cases += 1
            finally:
                bind(lib, 'test_restore_mxcsr', [I], None)(old)
    return {'packed_words_checked': checked, 'rejected_range_or_nonfinite_inputs': 2*len(invalid),
            'x86_ftz_daz_classifications_checked': flushed_cases}


def cases():
    rng = Random(239104)
    for m, n, k, kind, padded in ((1, 1, 1, 'integer', False),
                                  (15, 17, 31, 'random', True),
                                  (16, 16, 32, 'integer', False),
                                  (17, 19, 33, 'random', True),
                                  (13, 21, 66, 'cancellation', True),
                                  (31, 35, 65, 'random', True),
                                  (129, 33, 5, 'integer', True)):
        if kind == 'integer':
            values = lambda: rng.randrange(-4, 5)
        else:
            values = lambda: f32(rng.uniform(-1, 1))
        a = [[values() for _ in range(k)] for _ in range(m)]
        b = [[values() for _ in range(n)] for _ in range(k)]
        if kind == 'cancellation':
            for p in range(1, k, 2):
                for row in a:
                    row[p] = row[p-1]
                for j in range(n):
                    b[p][j] = -b[p-1][j]
        yield kind, padded, a, b

    # For each dot, the leading and first correction cancel exactly. The three
    # cases expose the three separate second-degree terms without large outputs
    # obscuring a missing contribution. All nonzero inputs satisfy the domain.
    x = from_bits(0x3f806d55)
    h, u, _ = [from_bits(v << 16) for v in split(x)]
    yield 'left-low', True, [[x, -h, -u]], [[1.], [1.], [1.]]
    yield 'right-low', True, [[1., 1., 1.]], [[x], [-h], [-u]]
    yield 'mid-mid', True, [[x, -x, -h, h]], [[x], [h], [x], [h]]


def scalar_outputs(a, b):
    parts_a = [[[from_bits(raw << 16) for raw in split(x)] for x in row] for row in a]
    parts_b = [[[from_bits(raw << 16) for raw in split(x)] for x in row] for row in b]
    rows = []
    for i in range(len(a)):
        result = []
        for j in range(len(b[0])):
            ab = [(parts_a[i][p], parts_b[p][j]) for p in range(len(b))]
            truth = math.fsum(a[i][p]*b[p][j] for p in range(len(b)))
            terms = [math.fsum(left[l]*right[r] for left, right in ab
                               for l in range(3) for r in range(3) if l+r == degree)
                     for degree in range(3)]
            six = math.fsum(terms)
            omitted = math.fsum(abs(left[l]*right[r]) for left, right in ab
                                for l, r in ((2, 1), (1, 2), (2, 2)))
            assert abs(six-truth) <= omitted + 3e-14*max(1, abs(truth)), (i, j, six, truth)
            assert abs(six-truth) < 0.002
            result.append((truth, six, terms))
        rows.append(result)
    return rows


def product_case(multiply, route, products, kind, padded, a, b, expected, trace=None):
    m, n, k = len(a), len(b[0]), len(b)
    offsets = (3, 7, 11) if padded else (0, 0, 0)
    pads = (9, 5, 13) if padded else (0, 0, 0)
    a_buf, ap, lda = view(m, k, pads[0], offsets[0], INPUT_GUARD)
    b_buf, bp, ldb = view(k, n, pads[1], offsets[1], INPUT_GUARD)
    c_buf, cp, ldc = view(m, n, pads[2], offsets[2], OUTPUT_GUARD)
    for i in range(m):
        for p in range(k):
            ap[i*lda+p] = a[i][p]
    for p in range(k):
        for j in range(n):
            bp[p*ldb+j] = b[p][j]
    original_a, original_b = bytes(a_buf), bytes(b_buf)
    before = [trace(i) for i in range(3)] if trace else []
    assert multiply(m, n, k, ap, lda, bp, ldb, cp, ldc) == 0
    assert route() == 1 and products() == 6
    assert bytes(a_buf) == original_a and bytes(b_buf) == original_b
    if trace:
        tiles = math.ceil(m/16)*math.ceil(n/16)
        kblocks = tiles*math.ceil(k/32)
        assert [trace(i)-before[i] for i in range(3)] == [tiles, kblocks, 6*kblocks]
    for index, word in enumerate(c_buf):
        relative = index-offsets[2]
        if relative < 0 or relative//ldc >= m or relative%ldc >= n:
            assert word == OUTPUT_GUARD, (kind, 'guard', index)
    worst = 0
    for i in range(m):
        for j in range(n):
            truth, six, terms = expected[i][j]
            actual = cp[i*ldc+j]
            assert math.isfinite(actual) and abs(actual-truth) <= 0.002, (kind, i, j, actual, truth)
            if kind in ('integer', 'cancellation'):
                assert actual == truth, (kind, actual, truth)
            if kind in ('left-low', 'right-low', 'mid-mid'):
                assert abs(math.fsum(terms[:2])) == 0 and abs(six) > 1e-6
                # The aggregation has at most five roundings on these tiny probes.
                assert abs(actual-six) <= 5e-10, (kind, actual, six)
            worst = max(worst, abs(actual-truth))
    return worst


def runtime_checks(lib, manifests, *, bench=False, trace=False):
    init = bind(lib, 'amx_fused6_init', [], I)
    compiled = bind(lib, 'cob_amx_six_compiled', [], I)()
    if bench:
        fn = bind(lib, 'precision_sgemm', [I]+GEMM, I)
        multiply = lambda *args: fn(6, *args)
        prefix = 'precision'
    else:
        multiply = bind(lib, 'amx_fused6_sgemm', GEMM, I)
        prefix = 'amx_fused6'
    route = bind(lib, prefix+'_last_route', [], I)
    products = bind(lib, prefix+'_last_products', [], I)
    reason = bind(lib, prefix+'_last_reason', [], ct.c_char_p)
    trace_counts = bind(lib, 'cob_amx_six_trace', [I], ct.c_uint64) if trace else None
    one = (F*1)(1)
    out = (U32*1)(OUTPUT_GUARD)
    assert multiply(0, 1, 1, None, 0, None, 0, None, 0) == 0
    assert multiply(1, 0, 1, None, 0, None, 0, None, 0) == 0
    zero_k = (U32*8)(*([OUTPUT_GUARD]*8))
    assert multiply(2, 2, 0, None, 0, None, 0, ct.cast(ct.byref(zero_k, 4), FP), 3) == 0
    assert list(zero_k) == [OUTPUT_GUARD, 0, 0, OUTPUT_GUARD, 0, 0, OUTPUT_GUARD, OUTPUT_GUARD]
    assert reason() == b'zero-k' and route() == 0
    assert multiply(1, -1, 1, one, 1, one, 1, ct.cast(out, FP), 1) == -1
    assert multiply(1, 1, 1, one, 0, one, 1, ct.cast(out, FP), 1) == -1 and out[0] == OUTPUT_GUARD
    if bench:
        assert fn(9, 1, 1, 1, one, 1, one, 1, ct.cast(out, FP), 1) == -1
        assert out[0] == OUTPUT_GUARD and reason() == b'unsupported-component-count'
    supported = init() == 0
    if not supported:
        rc = multiply(1, 1, 1, one, 1, one, 1, ct.cast(out, FP), 1)
        assert rc == -2 and route() == 0 and products() == 0 and out[0] == OUTPUT_GUARD, (rc, route(), products())
        if trace:
            assert [trace_counts(i) for i in range(3)] == [0, 0, 0]
        return {'amx_compute': 'SKIP', 'compiled_amx': bool(compiled), 'reason': reason().decode()}
    assert compiled and platform.system() == 'Linux' and platform.machine().lower() in ('x86_64', 'amd64')
    worst = max(product_case(multiply, route, products, *sample, trace_counts) for sample in manifests)
    # Confirm FP64 rescue classification and unchanged guard words on both sides.
    control_count = None if bench else bind(lib, 'test_fused_rescue_calls', [], I)
    for sub in (0x30000000, 0x4f000000, 0x7fc10001):
        for unsafe_left in (True, False):
            a = (U32*1)(sub if unsafe_left else 0x3f800000)
            b = (U32*1)(0x3f800000 if unsafe_left else sub)
            c = (U32*3)(OUTPUT_GUARD, OUTPUT_GUARD, OUTPUT_GUARD)
            before = control_count() if control_count else 0
            assert multiply(1, 1, 1, ct.cast(a, FP), 1, ct.cast(b, FP), 1, ct.cast(ct.byref(c, 4), FP), 1) == 0
            assert route() == 2 and products() == 0 and reason() == b'fp64-range-rescue'
            assert control_count is None or control_count() == before+1
            expected = from_bits(sub)
            assert c[0] == OUTPUT_GUARD and c[2] == OUTPUT_GUARD
            assert (math.isnan(from_bits(c[1])) if math.isnan(expected) else from_bits(c[1]) == expected)
    large = float(2**114)
    a = (F*3)(large, large, 1.)
    b = (F*3)(large, -large, .75)
    result = (F*1)(float('nan'))
    assert multiply(1, 1, 3, a, 3, b, 1, result, 1) == 0
    assert result[0] == .75 and route() == 2 and products() == 0 and reason() == b'fp64-range-rescue'
    return {'amx_compute': 'PASS', 'compiled_amx': True, 'native_cells': sum(len(a)*len(b[0]) for _,_,a,b,_ in manifests),
            'native_max_abs_error': worst, 'native_fp64_rescues': 7, 'trace_instrumented': trace}


def main():
    if not __debug__:
        raise RuntimeError('Run this validation with Python assertions enabled.')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('library', type=Path)
    parser.add_argument('--trace', type=Path)
    parser.add_argument('--plugin', type=Path, help='installed benchmark DSO with precision_sgemm(mode=6)')
    parser.add_argument('--require-native', action='store_true', help='fail if this host skips AMX execution')
    args = parser.parse_args()
    library = ct.CDLL(str(args.library.resolve()))
    report = pack_checks(library)
    manifests = []
    for kind, padded, a, b in cases():
        expected = scalar_outputs(a, b)
        if kind in ('left-low', 'right-low', 'mid-mid'):
            truth, six, terms = expected[0][0]
            assert math.fsum(terms[:2]) == 0 and abs(six) > 1e-6
        manifests.append((kind, padded, a, b, expected))
    errors = [abs(six-truth) for _,_,_,_,output in manifests for row in output for truth,six,_ in row]
    report['scalar_six_term_model'] = {'cells': len(errors), 'max_abs_error': max(errors),
                                       'cases': len(manifests)}
    report['ordinary'] = runtime_checks(library, manifests)
    report['benchmark_abi'] = runtime_checks(library, manifests, bench=True)
    if args.trace:
        report['instrumented'] = runtime_checks(ct.CDLL(str(args.trace.resolve())), manifests, trace=True)
    if args.plugin:
        report['installed_benchmark_abi'] = runtime_checks(ct.CDLL(str(args.plugin.resolve())), manifests, bench=True)
    print(json.dumps(report, indent=2))
    if args.require_native:
        assert all(value['amx_compute'] == 'PASS' for value in report.values()
                   if isinstance(value, dict) and 'amx_compute' in value), 'AMX execution was required'


if __name__ == '__main__':
    main()
