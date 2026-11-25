from pathlib import Path
import subprocess as sp
h=Path(__file__).resolve().parent;s=h/'candidate-source'
src=(s/'third_party/maxmulsk/strassen_leaf.c').read_text()
key='    const int done = cob_wide_run('
assert src.count(key)==1 and 'frontier_calls' not in src
src=src.replace('int cob_sgemm_sme_medium_wide','static uint64_t frontier_taken;\nuint64_t frontier_calls(void) { return frontier_taken; }\nint cob_sgemm_sme_medium_wide').replace('    free(packed);\n    return done;', '    free(packed);\n    frontier_taken += done;\n    return done;')
(h/'trace.c').write_text(src)
for cmd in (['cc','-O3','-std=c11','-mcpu=apple-m4','-march=armv8.7-a+sme+sme2','-c',h/'trace.c','-o',h/'trace.o'],['cc','-dynamiclib','-O3','-I'+str(s/'include'),h/'adapter.c',h/'candidate-build/CMakeFiles/cob_matmul.dir/src/gemm.c.o',h/'trace.o','-o',h/'trace.dylib'],['cc','-O3','-std=c11',h/'guards.c','-o',h/'guards']):
 sp.run(cmd,check=True)
for mode in ('candidate','trace'):
 with (h/('checks-'+mode+'.csv')).open('w') as f:sp.run([h/'guards',h/(mode+'.dylib')],stdout=f,check=True)
