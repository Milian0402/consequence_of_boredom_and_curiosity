"""Extract the four public UCI operand arrays from the original SHA-pinned ZIP."""
import argparse
import csv
import hashlib
import io
from pathlib import Path
import re
import subprocess
import zipfile

PIN = '0d7b054fea010270e9b3f06411c654c5e59547732ad626381980baffe0a23fb0'

def prepare(z, out):
    def counts(name):
        rows = list(csv.reader(io.TextIOWrapper(z.open(name), encoding='ascii')))
        return [bytes(map(int, row[:64])) for row in rows if len(row) == 65]
    test, train = counts('optdigits.tes'), counts('optdigits.tra')
    assert len(test)==1797 and len(train)==3823
    (out/'digits_A.u8').write_bytes(b''.join(test))
    (out/'digits_B.u8').write_bytes(b''.join(map(bytes, zip(*train))))
    for split, rows in [('windep',1797), ('tra',1934)]:
        data = subprocess.run(['gzip','-dc'], input=z.read('optdigits-orig.'+split+'.Z'),
                              stdout=subprocess.PIPE,check=True).stdout
        lines = re.findall(rb'^[01]{32}$', data, flags=re.MULTILINE)
        assert len(lines)==32*rows
        body = b''.join(lines).translate(bytes.maketrans(b'01',b'\0\1'))
        (out/f'optdigits-original-{split}.u8').write_bytes(body)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('out',type=Path)
    args=p.parse_args(); assert hashlib.sha256(args.archive.read_bytes()).hexdigest()==PIN
    args.out.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.archive) as z:prepare(z,args.out)
