"""Check the ordinary BLIS entry and inspect actual SVE microkernel calls in a separate binary."""
import ctypes as C
import mmap
from pathlib import Path
import sys

LIBC = C.CDLL(None, use_errno=True)
LIBC.mprotect.argtypes = (C.c_void_p, C.c_size_t, C.c_int)
FP = C.POINTER(C.c_float)
M, N, K = 97, 137, 131
LDA, LDB, LDC = K + 11, N + 17, N + 23
SENTINEL = -873.25


def parent(rows, cols, stride, offset, fn):
    count = offset + rows * stride + 29
    mem = mmap.mmap(-1, count * 4)
    buf = (C.c_float * count).from_buffer(mem)
    for j in range(count):
        buf[j] = SENTINEL
    for i in range(rows):
        for j in range(cols):
            buf[offset + i * stride + j] = fn(i, j)
    ptr = C.cast(C.byref(buf, 4 * offset), FP)
    return mem, buf, ptr


def readonly(mem, buf):
    if LIBC.mprotect(C.addressof(buf), len(mem), mmap.PROT_READ):
        raise OSError(C.get_errno(), 'mprotect read-only parent')


def run(path, diagnostic):
    dso = C.CDLL(str(path))
    assert dso.rank_vendor_init() == 0, 'BLIS runtime/architecture/one-thread check failed'
    dso.rank_vendor_sgemm.argtypes = (C.c_int, C.c_int, C.c_int, FP, C.c_int, FP, C.c_int, FP, C.c_int)
    dso.rank_vendor_sgemm.restype = None
    if diagnostic:
        dso.rank_blis_snapshot.argtypes = (C.POINTER(C.c_uint64),)
        dso.rank_blis_snapshot.restype = None
    for generation in (0, 1):
        def a_int(i, p):
            return ((i * 7 + p * 5 + generation * 3) % 17) - 8
        def b_int(p, j):
            return ((p * 13 + j * 3 + generation * 7) % 19) - 9
        am, ab, a = parent(M, K, LDA, 11, lambda i, p: a_int(i, p) / 8)
        bm, bb, b = parent(K, N, LDB, 21, lambda p, j: b_int(p, j) / 16)
        cm, cb, c = parent(M, N, LDC, 25, lambda i, j: float('nan'))
        readonly(am, ab)
        readonly(bm, bb)
        dso.rank_vendor_sgemm(M, N, K, a, LDA, b, LDB, c, LDC)
        # The generated A rows and B columns repeat every 17 and 19 indices.
        # Their integer dot products are exactly representable at this depth.
        oracle = [[sum(a_int(i, p) * b_int(p, j) for p in range(K)) / 128
                   for j in range(19)] for i in range(17)]
        checked = 0
        for idx, value in enumerate(cb):
            i, j = divmod(idx - 25, LDC)
            if 0 <= i < M and j < N:
                reference = oracle[i % 17][j % 19]
                assert value == reference, (path.name, i, j, value, reference)
                checked += 1
            else:
                assert value == SENTINEL, ('output guard', path.name, idx)
        assert checked == M*N
        if diagnostic:
            stats = (C.c_uint64 * 2)()
            dso.rank_blis_snapshot(stats)
            assert stats[0] > 0 and stats[1] >= M*N*K, ('missing optimized microkernel volume', list(stats))
        print(f'{path.name} fixture={generation} exact_outputs={checked} output_guards={len(cb)-checked} readonly_input_parents=2')


if __name__ == '__main__':
    directory = Path(sys.argv[1]).resolve()
    run(directory/'blis.so', False)
    run(directory/'blis-trace.so', True)
