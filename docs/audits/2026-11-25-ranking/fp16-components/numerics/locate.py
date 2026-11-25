"""Locate the existing repository evaluator and verify the immutable numerical sources."""
import hashlib
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
def reference(repo=None):
    bases=[Path(repo).resolve()] if repo else list(HERE.parents)
    for base in bases:
        path=base/'tools/world-rank/precision/accuracy'
        if (path/'registration.json').is_file():
            for line in (HERE/'data/reference.sha256').read_text().splitlines():
                expected,name=line.split('  ',1)
                if hashlib.sha256((path/name).read_bytes()).hexdigest()!=expected:
                    raise ValueError(f'Numerical source differs: {name}')
            sys.path.insert(0,str(path))
            return path
    raise ValueError('Pass --repo with the source repository root')
