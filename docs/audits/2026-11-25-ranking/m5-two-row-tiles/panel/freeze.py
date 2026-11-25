"""Freeze the second M5 compact holdout and check all previous data tables."""
import csv
import hashlib
import io
import json
import random
import subprocess
import zipfile
from pathlib import Path

here = Path(__file__).resolve().parent
root = next(p for p in here.parents if (p / 'outputs').is_dir() and (p / 'work').is_dir())
panel = here / 'panel.csv'
seeds = [804162307, 2065108391]
rng = random.Random(0x572DA1F3C94B86)
# Labels for the two repeats refer to positive direct-kernel results from the
# preceding compact panel. No performance information selects the fresh shapes.
cases = [
    ('repeat-positive-24', 24, 1280, 131, 0),
    ('repeat-positive-32', 32, 2496, 184, 0),
]
# Cover eight distinct heights in the eligible interval and scatter depths and
# widths so odd shapes and aligned shapes each include a wide/deep combination.
for i, (m, low_n, high_n, low_k, high_k) in enumerate([
    (24, 16, 26, 30, 38), (27, 27, 39, 12, 19),
    (29, 40, 52, 20, 29), (32, 53, 64, 39, 48)
], 1):
    cases.append((f'fresh-aligned-{i}', m, 64 * rng.randint(low_n, high_n),
                  8 * rng.randint(low_k, high_k), 0))
for i, (m, low_n, high_n, low_k, high_k) in enumerate([
    (25, 1537, 2303, 97, 159), (26, 3329, 4095, 305, 383),
    (30, 1025, 1535, 241, 303), (31, 2561, 3327, 161, 239)
], 1):
    cases.append((f'fresh-odd-{i}', m, rng.randrange(low_n, high_n + 1, 2),
                  rng.randrange(low_k, high_k + 1, 2), 0))
cases.extend([
    ('control-above-32', 33, 2880, 232, 0),
    ('control-padded', 28, 2673, 229, 5),
])
with panel.open('w', newline='') as f:
    w = csv.writer(f, lineterminator='\n')
    w.writerow(('family', 'm', 'n', 'k', 'pad'))
    w.writerows(cases)

targets = {(m, n, k): label for label, m, n, k, pad in cases}
matches = {label: set() for label in targets.values()}
digest = lambda b: hashlib.sha256(b).hexdigest()
unique = set()
inventory = {}
archives = set()
scanned = {'csv_tables':0, 'tsv_tables':0, 'json_documents':0, 'csv_tsv_rows':0,
           'json_dimension_objects':0, 'zip_archives':0}

def integral(x):
    if isinstance(x, bool):raise ValueError
    if isinstance(x, float) and not x.is_integer():raise ValueError
    return int(x)

def add_record(vals, label):
    try: t = tuple(map(integral, vals))
    except (ValueError, TypeError, OverflowError):return
    if t in targets: matches[targets[t]].add(label)

def visit(data, label, suffix):
    key = (suffix, digest(data))
    if key in unique:return
    unique.add(key)
    if suffix in ('.csv','.tsv'):
        try:
            reader = csv.reader(io.StringIO(data.decode('utf-8-sig')), delimiter=',' if suffix=='.csv' else '\t')
            cols = [s.strip().lower() for s in next(reader,())]
            if not {'m','n','k'}.issubset(cols):return
            idx = [cols.index(q) for q in ('m','n','k')]
            for row in reader:
                if len(row) <= max(idx):continue
                scanned['csv_tsv_rows'] += 1
                add_record([row[i] for i in idx],label)
        except (UnicodeError,csv.Error):return
        scanned[suffix[1:]+'_tables']+=1
    else:
        try: obj=json.loads(data)
        except (UnicodeError,ValueError):return
        count=0
        pending=[obj]
        while pending:
            curr=pending.pop()
            if isinstance(curr,dict):
                lookup={str(k).lower():v for k,v in curr.items()}
                if {'m','n','k'}.issubset(lookup):
                    add_record([lookup[q] for q in ('m','n','k')],label);count+=1
                pending.extend(x for x in curr.values() if isinstance(x,(dict,list)))
            elif isinstance(curr,list):pending.extend(x for x in curr if isinstance(x,(dict,list)))
        if not count:return
        scanned['json_documents']+=1;scanned['json_dimension_objects']+=count
    inventory[label]=key[1]

def scan_zip(data, label):
    dh=digest(data)
    if dh in archives:return
    archives.add(dh)
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for ent in sorted(z.infolist(),key=lambda x:x.filename):
                if ent.is_dir():continue
                suffix=Path(ent.filename).suffix.lower()
                if suffix in ('.csv','.tsv','.json'):visit(z.read(ent),label+'!'+ent.filename,suffix)
                elif suffix=='.zip':scan_zip(z.read(ent),label+'!'+ent.filename)
    except zipfile.BadZipFile:return
    scanned['zip_archives']+=1

files=subprocess.check_output(['rg','--files','--hidden','-g','!**/.git/**',
    '-g','*.csv','-g','*.tsv','-g','*.json','-g','*.zip','work','outputs'],cwd=root,text=True).splitlines()
for name in sorted(files):
    path=root/name
    # Retain all preceding compact-wave measurements.
    if path.is_relative_to(here.parent):continue
    s=path.suffix.lower()
    if s=='.zip':scan_zip(path.read_bytes(),name)
    else:visit(path.read_bytes(),name,s)
assert all(matches[k] for k in matches if k.startswith('repeat-'))
assert all(not matches[k] for k in matches if k.startswith(('fresh-','control-')))
doc={
  'panel_sha256':digest(panel.read_bytes()),'seeds':seeds,
  'scanner_scope':'Distinct saved CSV/TSV tables with m,n,k columns and JSON objects with m,n,k fields under work/ and outputs/, including ZIP members. Geometry matches ignore pad. Only this second Apple compact study directory is excluded; the preceding compact wave is included.',
  'generation':'Performance-blind fixed design. Four compact N/64 and K/8 integer layouts, four odd N/K compact layouts, two historical positive compact controls and two untreated controls.',
  'file_counts':scanned,
  'inventory_sha256':digest(('\n'.join(f'{inventory[f]}  {f}' for f in sorted(inventory))+'\n').encode()),
  'matching_prior_files_or_members':{k:len(v) for k,v in matches.items()},
  'repeat_example':{k:sorted(v)[0] for k,v in matches.items() if v},
}
(here/'novelty.json').write_text(json.dumps(doc,indent=2)+'\n')
print(json.dumps({k:doc[k] for k in ('panel_sha256','seeds','file_counts')}))
