"""Restore the Git-verified b218bef release and apply the frozen stream-B delta."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

here=Path(__file__).resolve().parent
repo,build,out=(Path(p).resolve() for p in sys.argv[1:])
baseline=build/'release'
candidate=build/'candidate'
subprocess.run([sys.executable,here.parent/'release-replay/prepare-sources.py',repo,baseline],check=True)
subprocess.run([sys.executable,here.parent/'release-public/apply-test-backport.py',baseline],check=True)
def files(tree):
    return {p.relative_to(tree).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in tree.rglob('*') if p.is_file()}
spec=json.loads((here.parent/'medium/source.json').read_bytes())
registration=json.loads((here/'preregistration.json').read_bytes())
before=files(baseline)
if before!=spec['release_files'] or spec['revision']!=registration['production_revision']:
    raise ValueError('release source inventory differs')
for filename in ('stream-b.patch','panel.csv','protocol.md','check.c'):
    if hashlib.sha256((here/filename).read_bytes()).hexdigest()!=registration['files'][filename]:
        raise ValueError('wrong frozen input '+filename)
shutil.copytree(baseline,candidate)
subprocess.run(['patch','-p1','-i',here/'stream-b.patch'],cwd=candidate,check=True)
after=files(candidate)
if {n for n in set(before)|set(after) if before.get(n)!=after.get(n)}!={'src/gemm.c','src/cob_stream_b.inc'}:
    raise ValueError('unexpected patch scope')
(out/'sources.json').write_text(json.dumps(dict(base=spec['revision'],patch=registration['files']['stream-b.patch'],original=before,candidate=after),indent=2,sort_keys=True)+'\n')
# Instrument only already-successful dispatch returns in separate trace objects.
for tree in (baseline,candidate):
    copy=tree.with_name(tree.name+'.trace')
    shutil.copytree(tree,copy)
    p=copy/'src/gemm.c'
    s=p.read_text()
    for n,call in [('old_medium','cob_sgemm_x86_medium_panel(m, n, k, a, lda, b, ldb, c, ldc)) return;'),
                   ('blocked','if (cob_sgemm_one_shot_blocked(m, n, k, a, lda, b, ldb, c, ldc)) return;')]:
        if s.count(call)!=1:raise ValueError('trace change not uniquely bound to '+n)
        s=s.replace(call,call.replace(' return;',' { ++stream_trace_'+n+'; return; }'))
    p.write_text('int stream_trace_old_medium, stream_trace_blocked;\n'+s)
