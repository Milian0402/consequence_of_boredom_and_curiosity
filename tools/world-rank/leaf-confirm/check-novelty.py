"""Audit saved ordered and permuted triples and the committed public-main sources."""
import argparse
import csv
import hashlib
import io
import itertools
import json
from pathlib import Path
import re
import subprocess
import zipfile

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('workspace', type=Path)
p.add_argument('--public-repo', type=Path, required=True)
p.add_argument('--revision', required=True)
p.add_argument('--inventory', type=Path, required=True)
a = p.parse_args()
root = a.workspace.resolve()
panel = Path(__file__).resolve().with_name('panel.csv')
cases = list(csv.DictReader(panel.open()))
by_shape = {tuple(int(r[k]) for k in ('m','n','k')):r['family'] for r in cases}
by_permutation = {v:family for t,family in by_shape.items() for v in itertools.permutations(t)}
families = list(by_shape.values())
matches = {family:[set(),set()] for family in families}
inventory = []
seen = set()
deduplicated = 0
tables = records = texts = 0
coordinate = [str(n) for t in by_shape for n in t]
separator = rb'(?:[ \t]*(?:[,xX*|]|\xc3\x97)[ \t]*|[ \t]+)'
number = rb'(' + b'|'.join(x.encode() for x in sorted(set(coordinate))) + rb')'
pattern = re.compile(rb'(?<![0-9])(?=' + number + separator + number + separator + number + rb'(?![0-9]))')
def scan(name, data, kind):
    global deduplicated, records, tables, texts
    digest = hashlib.sha256(data).hexdigest()
    key = kind, digest
    if key in seen:
        deduplicated += 1
        return
    seen.add(key)
    direct, perm = set(), set()
    count = 0
    def consider(t):
        if t in by_shape: direct.add(by_shape[t])
        if t in by_permutation: perm.add(by_permutation[t])
    if kind == 'csv':
        try:
            reader = csv.reader(io.StringIO(data.decode('utf-8-sig')))
        except UnicodeDecodeError:
            return
        cols = [x.strip().lower() for x in next(reader, [])]
        if not all(c in cols for c in ('m','n','k')): return
        indexes = [cols.index(c) for c in ('m','n','k')]
        for row in reader:
            try: dims = tuple(int(row[i]) for i in indexes)
            except (ValueError, IndexError): continue
            consider(dims)
            count += 1
        tables += 1
        records += count
    else:
        if b'\0' in data: return
        for match in pattern.finditer(data):
            consider(tuple(map(int,match.groups())))
        texts += 1
    for f in direct: matches[f][0].add(name)
    for f in perm: matches[f][1].add(name)
    inventory.append([name, digest, kind, count, ' '.join(sorted(direct)), ' '.join(sorted(perm))])

def archive(name, data):
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for member in sorted(z.namelist()):
            if member.lower().endswith('.csv'):
                scan(name+'!'+member, z.read(member), 'csv')
            elif member.lower().endswith('.zip'):
                archive(name+'!'+member, z.read(member))

repo = a.public_repo.resolve()
def git(*args):
    return subprocess.check_output(['git','-C',str(repo),*args])
revision = git('rev-parse',a.revision+'^{commit}').decode().strip()
tracked = git('ls-tree','-r','-z',revision).split(b'\0')
with subprocess.Popen(['git','-C',str(repo),'cat-file','--batch'], stdin=subprocess.PIPE, stdout=subprocess.PIPE) as objects:
    for item in tracked:
        if not item: continue
        meta, path = item.split(b'\t',1)
        mode, typ, oid = meta.split()
        name = path.decode()
        if typ != b'blob' or mode == b'120000': continue
        suffix = Path(name).suffix.lower()
        if suffix not in ('.csv','.tsv','.md','.txt','.py','.c','.h','.inc','.cc','.cpp','.hpp','.sh','.json','.yml','.yaml','.s','.rst',''): continue
        objects.stdin.write(oid+b'\n'); objects.stdin.flush()
        head = objects.stdout.readline().split()
        payload = objects.stdout.read(int(head[-1])); assert objects.stdout.read(1)==b'\n'
        scan('public-main@'+revision+'/'+name,payload,'csv' if suffix=='.csv' else 'text')
    objects.stdin.close()
paths = subprocess.check_output(['rg','--files','--hidden','--no-ignore','-g','!**/.git/**','-g','!work/world-rank/leaf-confirm/**','-g','*.csv','work','outputs'],cwd=root,text=True).splitlines()
for name in sorted(paths):
    scan(name,(root/name).read_bytes(),'csv')
for z in sorted((root/'outputs').rglob('*.zip')):
    archive(z.relative_to(root).as_posix(),z.read_bytes())
with a.inventory.open('w',newline='') as h:
    out = csv.writer(h,lineterminator='\n')
    out.writerow(['source','sha256','kind','dimension_rows','ordered_matches','permuted_matches'])
    out.writerows(inventory)
results = {k:{'ordered':len(v[0]),'any_permutation':len(v[1]),'example':min(v[0]) if v[0] else None} for k,v in matches.items()}
print(json.dumps(dict(panel_sha256=hashlib.sha256(panel.read_bytes()).hexdigest(),public_main=revision,
    unique_dimension_tables=tables,geometry_rows=records,public_text_files=texts,duplicate_payloads_skipped=deduplicated,
    inventory_sha256=hashlib.sha256(a.inventory.read_bytes()).hexdigest(),matches=results,
    scope='Workspace CSV files outside work/world-rank/leaf-confirm, all nested CSV members of output ZIPs, plus tracked main CSV and source/document text at the stated commit. Text scan recognizes adjacent ordered triples in common comma, table, multiplication or whitespace notation.'),indent=2))
assert all(v['ordered']>0 if not k.startswith('fresh-') else v['any_permutation']==0 for k,v in results.items())
