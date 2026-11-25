"""Replay the two original infrastructure failures from their immutable hosted archives."""
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

ORIGINAL = '61f30a237adb45f72a403315e78fafc70dd7295b'
RUN = 45708206524
JOBS = {1: (139535589638,15066497283,'2e0646df2dfe06bf28b576db2e31d7ceb30d87f5adbf40e6b6f36def07c4bb7c'),
        2: (139535589668,15067655075,'0057bbd6bcfb5c2bed02b96d7676c81431de82c8ff31dd6047cdeb9ab4b576fe')}

def need(b, msg):
    if not b:
        raise ValueError(msg)

def initial(archive, receipt):
    receipt=json.loads(Path(receipt).read_text())
    job,artifact,digest=JOBS[receipt['replica']]
    need((receipt['run'],receipt['attempt'],receipt['benchmark_sha'],receipt['status'],receipt['job_id'],receipt['artifact_id'],receipt['artifact_sha256']) ==
         (RUN,1,ORIGINAL,'failure',job,artifact,digest),'initial hosted identity')
    need(hashlib.sha256(Path(archive).read_bytes()).hexdigest()==digest and Path(archive).stat().st_size==receipt['bytes'],'initial archives differ')
    with zipfile.ZipFile(archive) as z:
        need(z.testzip() is None and len(z.namelist())==40==receipt['members'],'initial hosted members or CRC')
        source=lambda f:z.read(f).decode()
        rows=lambda f:list(csv.DictReader(io.StringIO(source(f))))
        need('mkl-prefix.txt: No such file or directory' in source('handle-build.log') and 'source/build.sh: line 6' in source('handle-build.log'),'unexpected initial failure')
        need(not {'handle-observations.csv','handle-builds.csv','handle-routes.csv','handle-score.json','raw.csv','admissions.csv'} & set(z.namelist()),'initial timing or admission data now present')
        need(hashlib.sha256(z.read('panel.csv')).hexdigest()=='2b4d6871262a4d038533b6cab819e819c3cff1eaf3ef606678b2ad6e30520ed5' and
             hashlib.sha256(z.read('protocol.md')).hexdigest()=='b63ad49548f9c8fd0f429427de3c860309d991bc3a791a695c15d5f6623e673e','initial registration')
        data=rows('incumbent-routes.csv');panel=rows('panel.csv'); need(len(data)==len(panel)==11,'incomplete initial panel')
        for row,p in zip(data,panel,strict=True):
            need([row[k] for k in ('family','layout')]==[p[k] for k in ('family','layout')] and int(row['checked'])==int(p['m'])*int(p['n']) and all(row[k]=='1' for k in ('public_calls','kernel_calls','forwarded_args','workspace_aligned','passed')),'initial incumbent dispatch')
        need('100% tests passed, 0 tests failed out of 6' in source('stream-incumbent-tests.txt'),'initial baseline tests')
        require_native=re.fullmatch(r'stream-incumbent PASS native_width=(32|64) AVX2_width=32 checked=(\d+) max_abs_error=([^\s]+)\n?',source('grouped-guard-tests.txt'))
        need(require_native is not None and float(require_native[3])<=0.002,'initial baseline guard test')
        model=re.search(r'^Model name:\s*(.+)$',source('host.txt'),re.M)[1].strip()
        return {'model':model,'native_width':int(require_native[1]),'smoke_checked':int(require_native[2]),'smoke_max_error':float(require_native[3]),'shape_routes':len(data),'shape_outputs':sum(int(r['checked']) for r in data),'source_list_sha256':hashlib.sha256(z.read('handle-source.sha256')).hexdigest()}

if __name__=='__main__':
    import sys
    print(json.dumps(initial(*sys.argv[1:]),indent=2))
