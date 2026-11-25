from pathlib import Path
import subprocess as sp,shutil,sys,difflib
h=Path(__file__).resolve().parent;repo=h/'source';old=h.parents[1]/'frontier-wave/apple-thin';leaf='third_party/maxmulsk/strassen_leaf.c'
original=(repo/leaf).read_text();start=original.index('__arm_locally_streaming __arm_new("za") static int\ncob_wide_run(');stop=original.index('int cob_sgemm_sme_medium_wide',start);pt=original.index('    for (int j = 0;',start);tail=original.index('    return 1;',pt)
body=original[start:pt].replace('cob_wide_run(', 'compact_run(')
anchor='    float *packed = aligned_alloc(64, (a_size + (size_t)k * 64) * sizeof(*packed));'
assert original.count(anchor)==1
for mode in sys.argv[1:]:
 trace=mode.endswith('-trace');variant=mode.removesuffix('-trace')
 extra=(h/'vertical-kernel.c').read_text() if variant=='vertical' else ''
 check='uint64_t compact_completed;\nuint64_t compact_observed(void){return compact_completed;}\n' if trace else ''
 s=original[:stop]+extra+body+(h/(variant+'.c')).read_text()+original[tail:stop]+check+original[stop:]
 frag='''    if (lda == k && ldb == n && ldc == n) {
        float *packed = aligned_alloc(64, (a_sizeEXTRA) * sizeof(*packed));
        if (!packed) return 0;
        const int done = compact_run(m, n, k, a, lda, b, ldb, c, ldc, packed, depth, packed + a_size);
        free(packed);
        TRACEreturn done;
    }
'''.replace('EXTRA',' + (size_t)k * 256' if variant=='superpanel' else '').replace('TRACE','compact_completed += done;\n        ' if trace else '')
 s=s.replace(anchor,frag+anchor)
 (h/(mode+'-leaf.c')).write_text(s)
 sp.run(['cc','-std=c11','-O3','-DCOB_ENABLE_APPLE_MEDIUM_WIDE=1','-mcpu=apple-m4','-march=armv8.7-a+sme+sme2','-c',h/(mode+'-leaf.c'),'-o',h/(mode+'-leaf.o')],check=True)
 sp.run(['cc','-O3','-dynamiclib','-I'+str(repo/'include'),old/'adapter.c',old/'package-on/CMakeFiles/cob_matmul.dir/src/gemm.c.o',h/(mode+'-leaf.o'),'-o',h/(mode+'.dylib')],check=True)
 if not trace:
  (h/(mode+'.patch')).write_text(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),'a/'+leaf,'b/'+leaf)))
