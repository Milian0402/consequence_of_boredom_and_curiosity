"""Exercise the public HK15 entry and its fallback without a large timing run.

Pass a patched COB checkout. A temporary fixture reduces only the two size
bounds. It counts entries and selected leaves, and injects allocation failure.
The exact production shape predicate is separately tested at its true bounds.
"""
import ctypes as ct
import math
import os
from pathlib import Path
import platform
from random import Random
import subprocess
import sys
import tempfile

repo = Path(sys.argv[1]).resolve() if len(sys.argv)>1 else next(
    path for path in Path(__file__).resolve().parents if (path/'src/gemm.c').is_file())
src = repo/'src'
P = ct.POINTER(ct.c_float)
args = [ct.c_int]*3+[P,ct.c_int,P,ct.c_int,P,ct.c_int]
with tempfile.TemporaryDirectory() as folder:
    tmp=Path(folder)
    code=(src/'gemm.c').read_text()
    hk=(src/'cob_hk15.inc').read_text()
    shape_start=hk.index('static int cob_hk15_shape')
    original_predicate=hk[shape_start:hk.index('\n}',shape_start)+2]
    gate=original_predicate.replace('static int cob_hk15_shape','int hk_test_shape')
    schedule=next((repo/p for p in ('third_party/fast-matmul/hk233.inc','tools/world-rank/finite/hk233.inc')
                   if (repo/p).is_file()))
    hk=hk.replace(original_predicate,original_predicate.replace('768','8').replace('1280','32'))
    old_bind='= cob_sgemm_rowmajor_base;'
    assert hk.count(old_bind)==1
    hk=hk.replace(old_bind,'= hk_test_leaf;')
    alloc='cob_aligned_alloc(128, (ac+bc+pc)*sizeof(float))'
    assert hk.count(alloc)==1
    hk=hk.replace(alloc,'hk_test_noalloc ? (++hk_test_failures, NULL) : '+alloc)
    schedule_include='#include "../third_party/fast-matmul/hk233.inc"'
    assert hk.count(schedule_include)==1
    hk=hk.replace(schedule_include,f'#include "{schedule}"')
    inspector='''
int hk_test_leaves, hk_test_entries, hk_test_noalloc, hk_test_failures;
static void hk_test_leaf(int m,int n,int k,const float* a,int lda,const float* b,int ldb,float* c,int ldc)
{
    ++hk_test_leaves;
    cob_sgemm_rowmajor_base(m,n,k,a,lda,b,ldb,c,ldc);
}
'''
    code=code.replace('#include "cob_hk15.inc"',inspector+gate+hk)
    hook='    if (cob_hk15_try(m, n, k, a, lda, b, ldb, c, ldc)) return;'
    assert code.count(hook)==1
    code=code.replace(hook,'    ++hk_test_entries;\n'+hook)
    code+='''
int hk_test_try(int m,int n,int k,const float* a,int lda,const float* b,int ldb,float* c,int ldc)
{ return cob_hk15_try(m,n,k,a,lda,b,ldb,c,ldc); }
void hk_test_base(int m,int n,int k,const float* a,int lda,const float* b,int ldb,float* c,int ldc)
{ cob_sgemm_rowmajor_base(m,n,k,a,lda,b,ldb,c,ldc); }
'''
    cfile=tmp/'fixture.c'; cfile.write_text(code)
    out=tmp/('hk-test.dylib' if platform.system()=='Darwin' else 'hk-test.so')
    flags=['-O2','-std=c11','-fPIC','-DCOB_ENABLE_HK15=1','-DCOB_ENABLE_PORTABLE_STRASSEN=1',
           '-DCOB_ENABLE_RECURSIVE_STRASSEN=1','-DCOB_ENABLE_BLOCKED_LEAF=1',
           '-DCOB_DISABLE_APPLE_AMX=1','-DCOB_DISABLE_APPLE_SME=1']
    if platform.machine().lower() in ('arm64','aarch64'): flags+=['-DCOB_USE_NEON=1']
    linker='-dynamiclib' if platform.system()=='Darwin' else '-shared'
    subprocess.run([os.environ.get('CC','cc'),*flags,'-I'+str(repo/'include'),'-I'+str(src),
                    linker,str(cfile),'-lm','-o',str(out)],check=True)
    lib=ct.CDLL(str(out))
    lib.cob_sgemm_rowmajor.argtypes=args;lib.cob_sgemm_rowmajor.restype=None
    lib.hk_test_base.argtypes=args;lib.hk_test_base.restype=None
    lib.hk_test_try.argtypes=args
    lib.hk_test_shape.argtypes=[ct.c_int]*3
    leaves=ct.c_int.in_dll(lib,'hk_test_leaves')
    entries=ct.c_int.in_dll(lib,'hk_test_entries')
    noalloc=ct.c_int.in_dll(lib,'hk_test_noalloc')
    failed_allocations=ct.c_int.in_dll(lib,'hk_test_failures')
    for shape,expected in [
        ((1536,2304,2304),1),((2560,3840,3840),1),((1920,2880,2880),1),
        ((2432,3648,3648),1),((2048,2880,3264),1),((1920,2304,2304),1),
        ((1922,2304,2304),0),((1534,2304,2304),0),((2562,3840,3840),0),
        ((1536,2303,2304),0),((1536,2304,2303),0),((1537,2304,2304),0)]:
        assert lib.hk_test_shape(*shape)==expected,(shape,expected)
    rng = Random(38421)
    dense_rng = Random(4897)
    measurements = []
    uniform_routes = {0: 0, 15: 0}

    def matrix(count):
        return (ct.c_float * count)(*([713.25] * count))

    def dense_value():
        magnitude = dense_rng.uniform(0.5, 1)
        return magnitude if dense_rng.getrandbits(1) else -magnitude

    def regular_inputs(m, n, k, a, lda, b, ldb):
        # Use logical elements only. This oracle applies to the bounded random inputs below.
        a_rows = [[abs(a[i*lda+p]) for p in range(k)] for i in range(m)]
        b_rows = [[abs(b[p*ldb+j]) for j in range(n)] for p in range(k)]
        a_cols = list(zip(*a_rows))
        b_cols = list(zip(*b_rows))
        for rows, cols, depths in ((a_rows, a_cols, a_rows), (b_rows, b_cols, b_cols)):
            peak = max(map(max, rows))
            if any(4 * max(line) < peak for line in (*rows, *cols)):
                return False
            if any(4 * sum(8*x >= peak for x in line) < 3*k for line in depths):
                return False
        return True

    def check_product(m, n, k, a, lda, b, ldb, expected_leaves, allocation_failure=False):
        ldc = n + 5
        c = matrix(m*ldc + 11)
        for i in range(m):
            for j in range(n):
                c[i*ldc+j] = float('nan')
        originals = bytes(a), bytes(b)
        leaves.value = entries.value = failed_allocations.value = 0
        noalloc.value = int(allocation_failure)
        lib.cob_sgemm_rowmajor(m, n, k, a, lda, b, ldb, c, ldc)
        assert (leaves.value, entries.value) == (expected_leaves, 1), (m, n, k, expected_leaves, leaves.value)
        expect_failure = allocation_failure and regular_inputs(m, n, k, a, lda, b, ldb)
        assert failed_allocations.value == int(expect_failure)
        worst = 0.0
        for i in range(m):
            for j in range(n):
                ref = sum(a[i*lda+p] * b[p*ldb+j] for p in range(k))
                err = abs(c[i*ldc+j] - ref)
                assert math.isfinite(err) and err <= .002, (m, n, k, i, j, err)
                worst = max(worst, err)
            assert list(c[i*ldc+n:(i+1)*ldc]) == [713.25] * (ldc-n)
        assert list(c[-11:]) == [713.25] * 11 and originals == (bytes(a), bytes(b))
        measurements.append((m*n, worst))

    for m,n,k,route in [(32,48,48,15),(44,60,72,15),(16,24,24,15),(64,96,96,15),
                        (33,48,48,0),(32,49,48,0),(32,48,49,0),(32,63,48,0),
                        (14,24,24,0),(66,96,96,0),(32,48,48,0)]:
        for legacy in (True, False):
            for cancel in (False, True):
                lda, ldb = k+1, n+3
                a, b = matrix(m*lda+11), matrix(k*ldb+11)
                sample = (lambda: rng.uniform(-1, 1)) if legacy else dense_value
                for i in range(m):
                    for p in range(k):
                        a[i*lda+p] = a[i*lda+p-1] if cancel and p%2 else sample()
                for p in range(k):
                    for j in range(n):
                        b[p*ldb+j] = -b[(p-1)*ldb+j] if cancel and p%2 else sample()
                expected = route
                if legacy and route and not regular_inputs(m, n, k, a, lda, b, ldb):
                    expected = 0
                check_product(m, n, k, a, lda, b, ldb, expected,
                              allocation_failure=(m,n,k,route) == (32,48,48,0))
                if legacy and route:
                    uniform_routes[leaves.value] += 1

    # Sweep the four possible K residues while holding geometry, operand peaks,
    # and the number of logical terms fixed within each accepted/rejected pair.
    m, n = 28, 42
    for k in (48, 45, 42, 39):
        lda, ldb = k+1, n+3
        required = (3*k+3)//4
        for operand in ('A', 'B'):
            a, b = matrix(m*lda+11), matrix(k*ldb+11)
            for i in range(m):
                for p in range(k):
                    a[i*lda+p] = dense_value()
            for p in range(k):
                for j in range(n):
                    b[p*ldb+j] = dense_value()
            for strong_count in (required, required-1):
                for p in range(k):
                    # Preserve a full-size maximum in every affected line.
                    value = 1.0 if p == 0 else (0.5 if p < strong_count else 0.0625)
                    if operand == 'A':
                        a[(m-2)*lda+p] = value
                    else:
                        b[p*ldb+n-3] = value
                check_product(m, n, k, a, lda, b, ldb, 15 if strong_count == required else 0)
    noalloc.value=0
    m,n,k=32,48,48
    buffers=[matrix(48*53+11) for _ in range(3)]
    a,b,c=buffers
    for operands in [(None,49,b,51,c,53),(a,49,None,51,c,53),(a,49,b,51,None,53),
                     (a,47,b,51,c,53),(a,49,b,47,c,53),(a,49,b,51,c,47),
                     (a,49,b,51,a,53),(a,49,b,51,b,53)]:
        leaves.value=0
        assert not lib.hk_test_try(m,n,k,*operands) and leaves.value==0
    for alias in (0,1):
        a,b=matrix(48*53+11),matrix(48*53+11)
        for v in (a,b):
            for i in range(len(v)): v[i]=rng.uniform(-1,1)
        a2,b2=type(a).from_buffer_copy(a),type(b).from_buffer_copy(b)
        c,c2=(a,a2) if alias==0 else (b,b2)
        leaves.value=0;entries.value=0
        lib.cob_sgemm_rowmajor(m,n,k,a,49,b,51,c,53)
        lib.hk_test_base(m,n,k,a2,49,b2,51,c2,53)
        assert (leaves.value,entries.value)==(0,1) and bytes(c)==bytes(c2)
    checked = sum(count for count, _ in measurements)
    worst = max(error for _, error in measurements)
    print(f'12 production geometry checks; {len(measurements)} FP64-checked public calls; {checked} FP64 cells; max error {worst:.9g}')
    print(f'Uniform fixtures on eligible geometries: {uniform_routes[15]} used 15 leaves and {uniform_routes[0]} used the fallback.')
    print('Verified 15 counted base leaves per accepted call, one public entry, forced-allocation fallback, null/stride/alias guards and untouched padding.')
