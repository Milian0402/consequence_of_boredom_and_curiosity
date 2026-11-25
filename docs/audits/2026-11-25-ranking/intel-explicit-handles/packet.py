"""Replay per-host scoring and verify data against an official artifact when included."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile

from evidence import validate
from failed import initial
from raw import require
from render import make_report, write_comparisons


def sha(data):
    return hashlib.sha256(data).hexdigest()


def same_json(a,b):
    return json.dumps(a,sort_keys=True,separators=(',',':'),allow_nan=False)==json.dumps(b,sort_keys=True,separators=(',',':'),allow_nan=False)


def run_packet(p, archive=None):
    receipt=json.loads((p/'github.json').read_text())
    members=json.loads((p/'members.json').read_text())
    require(receipt['status']=='success' and receipt['zipped_crc_passed'] and len(receipt['benchmark_sha'])==40, 'incomplete GitHub artifact')
    require(set(members)=={f.name for f in (p/'raw').iterdir() if f.is_file()}, 'incomplete portable manifest')
    for filename,details in members.items():
        require('/' not in filename and sha((p/'raw'/filename).read_bytes())==details['sha256'], 'member digest '+filename)
    if archive is not None:
        require(sha(archive.read_bytes())==receipt['artifact_sha256'] and archive.stat().st_size==receipt['bytes'], 'official ZIP digest')
        with zipfile.ZipFile(archive) as z:
            require(not z.testzip() and len([i for i in z.infolist() if not i.is_dir()])==receipt['members'], 'official ZIP CRC/count')
            for details in members.values():
                require(sha(z.read(details['member']))==details['sha256'], 'official ZIP payload '+details['member'])
    source_hashes={}
    for line in (p/'raw'/'handle-source.sha256').read_text().splitlines():
        h,name=line.split(None,1)
        require(name.startswith('source/') and '..' not in Path(name).parts,'unsafe source manifest')
        require(sha((p.parent/'registration'/name).read_bytes())==h,'source used for measured native adapter '+name)
        source_hashes[name]=h
    require({'source/handle.c','source/handle.h','source/observe.c','source/build.sh','source/routes.py'}<=set(source_hashes),'incomplete handle source coverage')
    score=validate(p/'raw',receipt)
    require(same_json(score,json.loads((p/'score.json').read_text())), 'recomputed/packaged evidence differs')
    return score


def main():
    cmd=argparse.ArgumentParser(description=__doc__)
    cmd.add_argument('kit',type=Path,nargs='?',default=Path(__file__).resolve().parent)
    a=cmd.parse_args(); kit=a.kit
    with (kit/'files.csv').open() as handle: ledger=list(csv.DictReader(handle))
    actual={f.relative_to(kit).as_posix() for f in kit.rglob('*') if
            f.is_file() and f.name!='files.csv' and '__pycache__' not in f.parts and 'originals' not in f.relative_to(kit).parts}
    require({r['path'] for r in ledger}==actual, 'missing or additional portable files')
    for record in ledger:
        require(sha((kit/record['path']).read_bytes())==record['sha256'], 'portable file checksum '+record['path'])
    for n in (1,2):
        p=kit/'initial-attempt'/f'replica-{n}'
        proof=initial(p.with_suffix('.zip'),p.with_suffix('.json'))
        require(same_json(proof,json.loads((p.parent/(p.name+'-summary.json')).read_text())),'initial failure replay')
    hosts=sorted(kit.glob('replica-*')); require(hosts, 'no hosts')
    scores=[]
    for host in hosts:
        archive=kit/'originals'/(host.name+'.zip')
        s=run_packet(host,archive if archive.exists() else None)
        scores.append(s)
        print(f"{host.name}: {s['model']} · {s['qualified_count']}/132 method-layout cells · {s['numerical']['complete_warm_calls']} checked whole warm calls")
    require((kit/'README.md').read_text()==make_report(scores),'rendered README differs from numerical replay')
    with tempfile.TemporaryDirectory() as tmp:
        regenerated=Path(tmp)/'comparison.csv'
        write_comparisons(regenerated,scores)
        require(regenerated.read_bytes()==(kit/'comparison.csv').read_bytes(),'reported table differs from numeric replay')
    print(f'Verified {len(ledger)} portable files and {len(hosts)} independent host replays.')


if __name__=='__main__': main()
