"""Deterministic row-major binary32 SGEMM fixtures.

CASES fixes evaluation order. KINDS distinguishes ordinary finite tests, finite
input range/overflow tests, and deliberate nonfinite inputs. build(case, seed)
returns (m, n, k, A_bits, B_bits), with flat lists of unsigned binary32 words.
Only integer arithmetic constructs the operands. There are no generated files,
floating-point library dependencies, expected results or evaluations on import.
"""
from operator import index


# name, category, M, N, K, PRNG domain. The paired positive cases share a domain.
_SPECS = (
    ('signed-full-mantissa', 'normal', 9, 11, 67, 1),
    ('positive-correlated', 'normal', 7, 13, 65, 2),
    ('positive-correlated-scale12', 'normal', 7, 13, 65, 2),
    ('bf16-rounding-boundaries', 'normal', 7, 11, 37, 3),
    ('same-sign-ulp-residue', 'normal', 9, 7, 257, 4),
    ('reciprocal-exponents', 'range', 7, 11, 97, 5),
    ('hk-exact-dyadics', 'normal', 11, 7, 63, 6),
    ('hk-complementary-scale', 'normal', 7, 9, 33, 17),
    ('hk-near-zero', 'normal', 5, 13, 33, 7),
    ('omitted-component-products', 'normal', 7, 9, 17, 8),
    ('bf16-subnormal-boundaries', 'range', 7, 9, 31, 9),
    ('fp32-subnormal-scaled', 'range', 5, 11, 29, 10),
    ('bf16-conversion-overflow', 'range', 7, 9, 19, 11),
    ('avoidable-intermediate-overflow', 'range', 5, 9, 17, 12),
    ('true-fp32-overflow', 'range', 7, 11, 19, 13),
    ('quiet-nan', 'nonfinite', 5, 9, 23, 14),
    ('signed-infinities', 'nonfinite', 7, 9, 27, 15),
    ('zero-times-infinity', 'nonfinite', 5, 11, 17, 16),
    ('zero-depth', 'zero', 3, 5, 0, 18),
)
CASES = tuple(s[0] for s in _SPECS)
KINDS = {s[0]: s[1] for s in _SPECS}
_PARAMETERS = {s[0]: s[2:] for s in _SPECS}

_MASK = (1 << 64) - 1
_SIGN = 1 << 31
_FRAC = (1 << 23) - 1
_ONE = 0x3f800000
_INF = 0x7f800000


class _Words:
    """SplitMix64 with explicit unsigned wraparound."""
    def __init__(self, seed, domain):
        self.state = (index(seed) ^ (domain * 0xd6e8feb86659fd93)) & _MASK

    def next(self):
        self.state = (self.state + 0x9e3779b97f4a7c15) & _MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xbf58476d1ce4e5b9) & _MASK
        z = ((z ^ (z >> 27)) * 0x94d049bb133111eb) & _MASK
        return z ^ (z >> 31)

    def sign(self):
        return (self.next() >> 63) << 31


def _normal(e, fraction=0, sign=0):
    """Encode a normal with unbiased exponent e and explicit fraction bits."""
    return sign | ((e + 127) << 23) | fraction


def _dyadic(coefficient, power):
    """Encode coefficient * 2**power, for the small exact normal values here."""
    if coefficient == 0:
        return 0
    mag = abs(coefficient)
    top = mag.bit_length() - 1
    return _normal(power + top, (mag << (23 - top)) & _FRAC, _SIGN if coefficient < 0 else 0)


def _random_normals(count, rng):
    words = []
    for _ in range(count):
        x = rng.next()
        words.append(_normal(int((x >> 32) % 8) - 4, x & _FRAC, (x >> 63) << 31))
    return words


def build(case, seed):
    """Construct independent operand arrays for one case and integer seed."""
    m, n, k, domain = _PARAMETERS[case]
    rng = _Words(seed, domain)
    a = [0] * (m*k)
    b = [0] * (k*n)

    if case == 'zero-depth':
        return m, n, k, a, b
    if case == 'signed-full-mantissa':
        a = _random_normals(m*k, rng)
        b = _random_normals(k*n, rng)

    elif case.startswith('positive-correlated'):
        # Each depth coordinate has a shared coarse positive profile, with
        # row/column-specific lower mantissa bits. Scale only A by exactly 2^12.
        profile = [rng.next() & 0x7ff000 for _ in range(k)]
        scale = 12 if case.endswith('scale12') else 0
        a = [_normal(scale, profile[p] | (rng.next() & 0xfff)) for i in range(m) for p in range(k)]
        b = [_normal(0, profile[p] | (rng.next() & 0xfff)) for p in range(k) for j in range(n)]

    elif case == 'bf16-rounding-boundaries':
        # Low halves just below, at and just above a BF16 midpoint. Alternate
        # retained-bit parity every three entries; magnitude and sign vary.
        def boundary(t):
            high = ((rng.next() % 63) << 1) | ((t // 3) & 1)
            low = (0x7fff, 0x8000, 0x8001)[t % 3]
            return _normal((t % 7) - 3, (high << 16) | low, rng.sign())
        a = [boundary(t) for t in range(m*k)]
        b = [boundary(t+1) for t in range(k*n)]

    elif case == 'same-sign-ulp-residue':
        # Every adjacent pair contributes a positive one-ULP difference times
        # the same positive multiplier. Zero-pad the last depth coordinate.
        for p in range(0, k-1, 2):
            for i in range(m):
                x = _normal(0, 1 + rng.next() % _FRAC)
                a[i*k+p], a[i*k+p+1] = x, (x-1) | _SIGN
            for j in range(n):
                x = _normal(0, rng.next() & _FRAC)
                b[p*n+j] = b[(p+1)*n+j] = x

    elif case == 'reciprocal-exponents':
        # Magnitudes span 224 binades across depth. Opposite exponents keep
        # exact individual product magnitudes in [1,4).
        exponents = (-112, 112, -95, 95, -64, 64, -29, 29, -1, 1, 0)
        shift = rng.next() % len(exponents)
        for p in range(k):
            e = exponents[(p + shift) % len(exponents)]
            for i in range(m):
                a[i*k+p] = _normal(e, rng.next() & _FRAC, rng.sign())
            for j in range(n):
                b[p*n+j] = _normal(-e, rng.next() & _FRAC, rng.sign())

    elif case == 'hk-exact-dyadics':
        # Bounded signed dyadics; the entire ordinary dot is exactly
        # representable in FP32, including any order of summing its terms.
        a = [_dyadic(int(rng.next() % 33)-16, -5) for _ in a]
        b = [_dyadic(int(rng.next() % 33)-16, -5) for _ in b]

    elif case == 'hk-complementary-scale':
        # Complementary depth factors keep |A_ip B_pj| constant across depth.
        # Tiny rows and columns cross, with a zero row and zero column.
        # Alternating depth signs leave exactly one signed product (K odd).
        row_signs = [rng.sign() for _ in range(m)]
        col_signs = [rng.sign() for _ in range(n)]
        for p in range(k):
            group = (17*p) % 7 < 3
            for i in range(1,m):
                a[i*k+p] = _normal((-18 if i%2 else 0) - 8*group, 0, row_signs[i] ^ (_SIGN if p%2 else 0))
            for j in range(1,n):
                b[p*n+j] = _normal((-18 if j%3 else 0) - 8*(not group), 0, col_signs[j])

    elif case == 'hk-near-zero':
        # Keep opposite contributions far apart. Each pair cancels exactly
        # mathematically, leaving only the last small signed dyadic product.
        h = k // 2
        for p in range(h):
            for i in range(m):
                x = _dyadic(1 + rng.next() % 31, -4) | rng.sign()
                a[i*k+p], a[i*k+p+h] = x, x ^ _SIGN
            for j in range(n):
                x = _dyadic(1 + rng.next() % 31, -4) | rng.sign()
                b[p*n+j] = b[(p+h)*n+j] = x
        for i in range(m):
            a[i*k+k-1] = _dyadic(1+rng.next()%7, -15) | rng.sign()
        for j in range(n):
            b[(k-1)*n+j] = _dyadic(1+rng.next()%7, -16) | rng.sign()

    elif case == 'omitted-component-products':
        # x = 1 + 2^-9 + 2^-18 splits exactly into three BF16 components.
        # Each dot is a 4-product expansion of (x - a0)*(x - b0).
        # Choosing a0/b0 as 1 or 1+2^-9 gives mid/mid, mid/low and low/low
        # residuals. Individual scalar products use both signs and exponents.
        mid, low = 1 << (23-9), 1 << (23-18)
        for i in range(m):
            e = int(rng.next()%5)-2
            x, ref = _normal(e, mid|low), _normal(e, mid if i%2 else 0)
            sign = rng.sign()
            a[i*k:i*k+4] = [x^sign, ref^sign^_SIGN, x^sign^_SIGN, ref^sign]
        for j in range(n):
            e = int(rng.next()%5)-2
            x, ref = _normal(e, mid|low), _normal(e, mid if j%2 else 0)
            sign = rng.sign()
            for p, val in enumerate((x, x, ref, ref)):
                b[p*n+j] = val ^ sign

    elif case in ('bf16-subnormal-boundaries', 'fp32-subnormal-scaled'):
        # A spans FP32 min-normal and BF16 subnormal/tie transitions or the
        # FP32 subnormal lattice. B's large finite binary powers rescale
        # products into the normal range, exposing a flushed input.
        if case == 'bf16-subnormal-boundaries':
            mags = (0x007fffff, 0x00800000, 0x00800001, 0x007f0000, 0x007f7fff,
                    0x007f8000, 0x007f8001, 0x00007fff, 0x00008000, 0x00008001,
                    0x0000ffff, 0x00010000, 0x00010001, 0x00018000, 0x00018001)
        else:
            mags = (1, 2, 3, 7, 17, 127, 255, 257, 1023, 2049, 4095, 8193, 16385)
        shift = rng.next() % len(mags)
        a = [mags[(t+shift)%len(mags)] | rng.sign() for t in range(m*k)]
        b = [_normal(118 + rng.next()%5, 0, rng.sign()) for _ in b]

    elif case in ('bf16-conversion-overflow', 'true-fp32-overflow'):
        # One nonzero A per row limits the exact product. Near-maximum A
        # rounds to BF16 infinity. B=±1/8 yields a finite exact FP32 result;
        # B=±2 puts the exact result beyond the FP32 overflow boundary.
        pos = rng.next()%k
        for i in range(m):
            a[i*k+pos] = (0x7f7fffff - rng.next()%1024) | rng.sign()
        b = [_normal(-3 if case.startswith('bf16-') else 1, 0, rng.sign()) for _ in b]

    elif case == 'avoidable-intermediate-overflow':
        # Finite products of order 2^160 cancel in exact pairs. The last
        # coordinate leaves a small signed finite result in every output.
        for p in range(0, k-1, 2):
            for i in range(m):
                x = _normal(100, rng.next() & _FRAC, rng.sign())
                a[i*k+p], a[i*k+p+1] = x, x ^ _SIGN
            for j in range(n):
                x = _normal(60, rng.next() & _FRAC, rng.sign())
                b[p*n+j] = b[(p+1)*n+j] = x
        for i in range(m):
            a[i*k+k-1] = _ONE | rng.sign()
        for j in range(n):
            b[(k-1)*n+j] = _normal(-3, 0, rng.sign())

    else:
        # Retain unaffected finite columns and/or rows as internal controls.
        a, b = _random_normals(m*k, rng), _random_normals(k*n, rng)
        p = rng.next()%k
        if case == 'quiet-nan':
            a[p] = 0x7fc00000 | (rng.next() & 0x3fffff)
            b[p*n+2] = 0xffc00000 | (rng.next() & 0x3fffff)
        else:
            for i in range(m):
                signs = (_ONE, _ONE|_SIGN) if case == 'signed-infinities' else (0, _SIGN, _ONE, _ONE|_SIGN)
                a[i*k+p] = signs[i % len(signs)]
            for j in range(n):
                if j % 3 != 2:
                    b[p*n+j] = _INF | (_SIGN if j%3==1 else 0)
    return m, n, k, a, b
