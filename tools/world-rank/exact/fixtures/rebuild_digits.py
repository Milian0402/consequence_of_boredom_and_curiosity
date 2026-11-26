"""Check the two original UCI partitions and regenerate the four compact matrix byte streams."""
import argparse
import hashlib
from pathlib import Path
import re
import subprocess
import zipfile

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('archive',type=Path)
ap.add_argument('out',type=Path)
args=ap.parse_args()
def verify(data,digest):
    if hashlib.sha256(data).hexdigest()!=digest:raise ValueError('Source hash mismatch')
verify(args.archive.read_bytes(),'0d7b054fea010270e9b3f06411c654c5e59547732ad626381980baffe0a23fb0')
def features(raw,rows):
    vals=[]
    for line in raw.splitlines():
        x=list(map(int,line.split(b',')))
        if len(x)!=65 or not 0<=x[-1]<=9 or not all(0<=v<=16 for v in x[:64]):raise ValueError('Feature record')
        vals.extend(x[:64])
    if len(vals)!=rows*64:raise ValueError('Feature row count')
    return bytes(vals)
def images(raw,rows):
    lines=subprocess.run(['gzip','-dc'],input=raw,stdout=subprocess.PIPE,check=True).stdout.splitlines()
    pixels=b''.join(line for line in lines if re.fullmatch(rb'[01]{32}',line))
    if len(pixels)!=rows*1024:raise ValueError('Image row count')
    return bytes(v-48 for v in pixels)
def tr(raw,k): return b''.join(raw[p::k] for p in range(k))
with zipfile.ZipFile(args.archive) as z:
    out={'digits_A.u8':features(z.read('optdigits.tes'),1797),'digits_B.u8':tr(features(z.read('optdigits.tra'),3823),64),
         'image_A.u8':images(z.read('optdigits-orig.windep.Z'),1797),'image_B.u8':tr(images(z.read('optdigits-orig.tra.Z'),1934),1024)}
args.out.mkdir(exist_ok=True,parents=True)
manifest=Path(__file__).resolve().parent/'data/SHA256SUMS'
for line in manifest.read_text().splitlines():
    digest,name=line.split('  ',1);verify(out[name],digest);(args.out/name).write_bytes(out[name])
