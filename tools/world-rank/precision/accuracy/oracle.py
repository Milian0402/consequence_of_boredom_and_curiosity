"""Integer dyadic oracles and explicit IEEE round-to-nearest-even software controls."""
from functools import lru_cache

SIGN = 1 << 31
MAG = SIGN - 1
INF = 0x7f800000
NAN = 0x7fc00000
MODES = (1, 3, 6, 9, 16)
TERMS = {
    1: ((0,0),),
    3: ((0,1),(1,0),(0,0)),
    6: ((0,2),(1,1),(2,0),(0,1),(1,0),(0,0)),
    9: ((2,2),(1,2),(2,1),(0,2),(1,1),(2,0),(0,1),(1,0),(0,0)),
}


@lru_cache(maxsize=32768)
def units(u):
    """Finite FP32 bits as a signed integer in units of 2^-149."""
    exponent, fraction = (u >> 23) & 255, u & 0x7fffff
    if exponent == 255:
        raise ValueError('Expected a finite input')
    value = ((0x800000 + fraction) << (exponent-1)) if exponent else fraction
    return -value if u & SIGN else value


def round_fp32(value, exp):
    """Correctly round signed integer value times 2**exp; canonical +0 when exact."""
    if not value:
        return 0
    sign = SIGN if value < 0 else 0
    n = abs(value)
    shift = max(n.bit_length()-24, -149-exp)
    if shift > 0:
        q, rem = divmod(n, 1 << shift)
        half = 1 << (shift-1)
        n = q + (rem > half or (rem == half and q & 1))
    else:
        n <<= -shift
    exp += shift
    if not n:
        return sign
    top = n.bit_length()-1
    e = top + exp
    if e >= 128:
        return sign | INF
    if e < -126:
        return sign | (n << (exp+149))
    mantissa = n >> (top-23) if top >= 23 else n << (23-top)
    return sign | ((e+127)<<23) | (mantissa & 0x7fffff)


def add(x,y):
    ax, ay = x&MAG, y&MAG
    if ax >= INF or ay >= INF:
        if ax>INF or ay>INF or (ax==ay==INF and x!=y):
            return NAN
        return x if ax==INF else y
    if x==y==SIGN:
        return SIGN
    return round_fp32(units(x)+units(y), -149)


def product_class(x,y):
    ax, ay = x&MAG,y&MAG
    if ax>INF or ay>INF or (ax==0 and ay==INF) or (ay==0 and ax==INF):
        return NAN
    if ax==INF or ay==INF:
        return ((x^y)&SIGN)|INF
    return None


def fma(x,y,z):
    special = product_class(x,y)
    if special is not None:
        return add(special,z)
    if z&MAG >= INF:
        return NAN if z&MAG > INF else z
    if x&MAG == 0 or y&MAG == 0:
        return add((x^y)&SIGN,z)
    return round_fp32(units(x)*units(y)+(units(z)<<149), -298)


def bf(x):
    """BF16 RNE, including gradual underflow, returned as widened FP32 bits."""
    if x&MAG > INF:
        return (x & 0xffff0000) | 0x00400000
    return (x + 0x7fff + ((x>>16)&1)) & 0xffff0000


@lru_cache(maxsize=32768)
def triple(x):
    parts = []
    r = x
    for i in range(3):
        parts.append(bf(r))
        if i<2:
            r=add(r, parts[-1]^SIGN)
    return tuple(parts)


def exact(A,B,m,n,k):
    """Return exact signed and unsigned magnitude sums in 2^-298 units or IEEE class."""
    outputs = []
    aa = [units(x) if x&MAG < INF else None for x in A]
    bb = [units(x) if x&MAG < INF else None for x in B]
    for i in range(m):
        for j in range(n):
            total = magnitude = 0
            exceptional = None
            for p in range(k):
                ia, ib = i*k+p, p*n+j
                x,y = aa[ia],bb[ib]
                if x is None or y is None:
                    term=product_class(A[ia], B[ib])
                    exceptional = term if exceptional is None else add(exceptional,term)
                else:
                    term=x*y
                    total+=term
                    magnitude+=abs(term)
            if exceptional is not None:
                outputs.append((exceptional,None,None))
            else:
                outputs.append((round_fp32(total,-298),total,magnitude))
    return outputs


def model(A,B,m,n,k,mode):
    if mode == 'fp32-fma':
        return [dot(A[i*k:(i+1)*k],B[j::n]) for i in range(m) for j in range(n)]
    a = [triple(x) for x in A]
    b = [triple(x) for x in B]
    result = []
    for i in range(m):
        for j in range(n):
            c = 0
            terms = TERMS[6 if mode == 16 else mode]
            groups = (terms[:3], terms[3:5], terms[5:]) if mode==16 else tuple((t,) for t in terms)
            for group in groups:
                v = 0
                for left,right in group:
                    for p in range(k):
                        v = fma(a[i*k+p][left], b[p*n+j][right], v)
                c = v if len(groups) == 1 else add(c,v)
            result.append(c)
    return result


def dot(x,y):
    c=0
    for a,b in zip(x,y,strict=True):
        c=fma(a,b,c)
    return c


def ulps(a,b):
    """Adjacent representable finite FP32 values, coalescing signed zeros."""
    def order(v):
        return -(v&MAG) if v&SIGN else v
    return abs(order(a)-order(b))
