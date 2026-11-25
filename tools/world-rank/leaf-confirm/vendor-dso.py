"""Bind each installed release worker to the digest-pinned sequential LP64 Linux provider."""
import hashlib
from pathlib import Path
import re
import subprocess
import sys

EXPECTED = {
    'armpl': {'libarmpl_lp64.so': '42ad9482d2eeddb85ceddb127a8a88975def8491a77af254e20f05fd0b388cf9'},
    'mkl': {
        'libmkl_intel_lp64.so.3': 'eb5a6748ec4e85dc07bd4dbf860af21e86f26647dbba837a94338a8289bf1faa',
        'libmkl_sequential.so.3': '812f56b8de337f2bc591426462232a4b114a443b284dabfb985e9af0c39f32be',
        'libmkl_core.so.3': '2b7ed4833447281eec1edee574115f663169dec03d38c7cd2ed8fe2db8c0861e'},
}


def dynamic(path, tag):
    result = subprocess.check_output(['readelf', '-dW', path], text=True)
    return re.findall(r'\(' + tag + r'\)[^\n]*?\[([^\]]+)\]', result)


def verify(provider, directory, workers):
    for name, expected in EXPECTED[provider].items():
        library = (directory/name).resolve(strict=True)
        with library.open('rb') as f:
            digest = hashlib.sha256()
            for block in iter(lambda: f.read(1024*1024), b''):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f'{name}: shared object differs from pinned {provider} package')
        soname = dynamic(library,'SONAME')
        if soname and soname != [name]:
            raise ValueError(f'{name}: unexpected DT_SONAME {soname}')
        for worker in workers:
            needed = dynamic(worker,'NEEDED')
            if needed.count(name) != 1:
                raise ValueError(f'{worker.name}: expected one bare DT_NEEDED for {name}, got {needed}')
            ldd = subprocess.check_output(['ldd',worker],text=True)
            entries = re.findall(r'^\s*' + re.escape(name) + r' => (\S+) \(',ldd,re.M)
            if len(entries)!=1 or Path(entries[0]).resolve(strict=True)!=library:
                raise ValueError(f'{worker.name}: {name} did not resolve to its pinned SDK file: {entries}')
            print(f'worker={worker.name} needed={name} sdk_soname={soname[0] if soname else "none"} same_file=1 sha256={expected}')


if __name__=='__main__':
    provider, sdk, *executables = sys.argv[1:]
    verify(provider, Path(sdk), [Path(f) for f in executables])
