"""Freeze/verify input byte identities independently of any backend or oracle."""
import hashlib
import json
import struct
import sys
from pathlib import Path
from fixtures import CASES, KINDS, build

HERE = Path(__file__).resolve().parent
SEEDS = (0x51D9A47B, 0xA37C096D)
REGISTER = HERE / 'registration.json'
REGISTER_SHA256 = 'ec4bea33f96d3d7d6c13ff87a7f8cfa8e95a221662b1c708263867b70a4cbfc5'


def word_hash(words):
    data = struct.pack(f'<{len(words)}I', *words)
    return hashlib.sha256(data).hexdigest()


def manifest():
    sources = {p: hashlib.sha256((HERE/p).read_bytes()).hexdigest() for p in ('fixtures.py', 'protocol.md')}
    entries = []
    for case in CASES:
        for seed in SEEDS:
            m,n,k,a,b = build(case, seed)
            assert len(a)==m*k and len(b)==k*n
            entries.append(dict(case=case, category=KINDS[case], seed=seed, m=m, n=n, k=k,
                                a_sha256=word_hash(a), b_sha256=word_hash(b)))
    return json.dumps(dict(sources=sources, inputs=entries), indent=2)+'\n'


def verify():
    if REGISTER.read_text() != manifest():
        raise ValueError('Frozen fixture registration differs from regenerated inputs or protocol')
    digest=hashlib.sha256(REGISTER.read_bytes()).hexdigest()
    if digest != REGISTER_SHA256:
        raise ValueError('Input registration differs from the recorded freeze')
    return digest


if __name__ == '__main__':
    if sys.argv[1:] == ['--freeze'] and not REGISTER.exists():
        with REGISTER.open('x') as f:
            f.write(manifest())
    elif sys.argv[1:] not in ([], ['--freeze']):
        raise SystemExit('Usage: python3 -B registration.py [--freeze]')
    print(verify())
