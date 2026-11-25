"""Apply the forward-only patch to the archived preimage and repeat its static-provider proof."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


base = Path(__file__).resolve().parent
proof = json.loads((base/'correctness.json').read_text())
patch = base/'openblas-proof-fix.patch'
old = base/'vendor-release.measured.sh'
if (digest(old), digest(patch)) != (proof['measured_sha256'], proof['proposed_patch_sha256']):
    raise ValueError('The recorded source or proposed patch changed')
with tempfile.TemporaryDirectory() as tmp:
    top = Path(tmp)
    suite = top/'tools/world-rank/ragged-hk'
    suite.mkdir(parents=True)
    shutil.copyfile(old,suite/'vendor-release.sh')
    for options in (['--check'], []):
        subprocess.run(['git','apply','--no-index',*options,patch],cwd=top,check=True)
    if {p.name:digest(p) for p in suite.iterdir()} != proof['source_files_sha256']:
        raise ValueError('Reconstructed source differs from the tested version')
    subprocess.run([sys.executable,'-B',suite/'test-openblas-link.py'],check=True)
    for method, result in proof['methods'].items():
        evidence = base/method
        subprocess.run([sys.executable,'-B',suite/'check-openblas-link.py',
                        f'/work/product/{method}-install/lib/libcob_matmul.a','/tmp/openblas-sdk/lib/libopenblas.a',
                        evidence/'ordinary-link.log',evidence/'api-symbols.txt'],check=True)
        symbols = {s.split()[-1] for s in (evidence/'api-symbols.txt').read_text().splitlines()}
        if 'sgemm_' in symbols:
            raise ValueError('Original Fortran-symbol predicate would unexpectedly succeed')
        identities = dict(reversed(l.split()) for l in (evidence/'identities.sha256').read_text().splitlines())
        for r in result['workers'].values():
            if identities[r['executable']] != r['sha256'] or r['checked'] != 2049*2051 or r['threads'] != 1 or not 0 <= r['max_error'] <= .002:
                raise ValueError('Worker identity, one-thread execution or full accuracy receipt is inconsistent')
        w = result['workers']
        if (w['ordinary']['output_sha256'] != w['trace']['output_sha256'] or
            identities['libopenblas.a'] != proof['provider_archive_sha256']):
            raise ValueError('Guarded result digest or provider digest mismatch')
        lines = (evidence/'vendor-route.txt').read_text().splitlines()
        count = result['total_vendor_calls']
        if (result['core_leaves'] != 49 or len(lines) != 2+count or
            lines[1] != f"RAGGED_VENDOR_ROUTE,{count},{count},0,{result['vendor']}" or
            'core_calls=49,core_overwrites=49' not in lines[0] or 'provider_ok=1' not in lines[0]):
            raise ValueError('Actual calls do not support the recorded decomposition')
print('Verified the exact patch, six negative mutations, both real static call origins and both numerical receipts.')
