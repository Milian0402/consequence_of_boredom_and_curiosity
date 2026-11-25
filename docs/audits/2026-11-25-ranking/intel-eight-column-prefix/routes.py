"""Independently reconstruct the three assisted SGEMM routes on each registered view."""
import csv
import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path

PANEL_SHA = '2b4d6871262a4d038533b6cab819e819c3cff1eaf3ef606678b2ad6e30520ed5'
SEEDS = (1047891527, 1936510481)
METHODS = ('host-forward', 'host-split8', 'host-prefix8')
GEOMETRY = 'm n k lda ldb ldc ar ac br bc cr cc'.split()
SPLIT_FIELDS = 'kind index m n k lda ldb ldc a_offset b_offset c_offset'.split()
MKL_SDK = {
    'libmkl_intel_lp64.so.3': 'eb5a6748ec4e85dc07bd4dbf860af21e86f26647dbba837a94338a8289bf1faa',
    'libmkl_sequential.so.3': '812f56b8de337f2bc591426462232a4b114a443b284dabfb985e9af0c39f32be',
    'libmkl_core.so.3': '2b7ed4833447281eec1edee574115f663169dec03d38c7cd2ed8fe2db8c0861e',
}
MKL_SOURCES = {
    'vendor_backend.c': 'dea6e72836e4cceb42148af9a379e61b509475569fa94742a33452371ed72536',
    'vendor_adapter.c': '56109f058436a5ede403cd70a7ea294c8cd0cbb866e9759404eb028c57593596',
    'build.sh': '814ad66db558df46037339b1fe01bebca2754e13fe7c4a8f5e8c392de76d6b8c',
}
PREFIX_SOURCES = {
    'source/README.md': '79b1e7a16e6da4eaf6cd705c5fa7280bc4552138b035fe156a1f4f2901bf9434',
    'source/local-test.sh': '257d8bb3fee15a01ca6ac42a3bb3a82daa51d38e20ca833f81fa604c5c27f63b',
    'source/mock-provider.c': 'df51ded8af70f2c031881b6be380332ec09038270d79e35382a4e4cc18eec6c7',
    'source/adapter.c': '0c2aed6992c7b199901fd82133def4b7725a5e8fae02b141144a044899e1901d',
    'source/check.c': '6a9df13fe5b4a04d5bf28cf37b45abb4dcffb6ab199bb764832eafdff0defb48',
    'source/observer.c': 'c1f7fce61c3f2e2a5a0852f6c4890973bca9614fe6ff6584aa62cc112f64c85d',
    'source/prefix.h': '8062e8d9125977335534e30e84372378ef0f4873a6c27b1b8e4f9346a9f41920',
    'source/build.sh': '5375587f0795cb15745fee3a393a750d4d84de6cebaf3f8e588be6176fc05539',
    'source/routes.py': '8c18bde90081073235a9672f656aa6999d4c74360b6226bc623e3bde005da016',
}


def need(condition, why):
    if not condition:
        raise ValueError(why)


def rows(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def index(records, keys):
    result = {}
    for row in records:
        key = tuple(row[k] for k in keys)
        need(key not in result, f'Duplicate receipt {key}')
        result[key] = row
    return result


def sha(value):
    need(bool(re.fullmatch('[0-9a-f]{64}', value)), 'Missing SHA-256 identity')
    return value


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hashes(path):
    result = {}
    for line in path.read_text().splitlines():
        value, name = line.split(maxsplit=1)
        need(name not in result, 'Duplicate member in ' + path.name)
        result[name] = sha(value)
    return result


def mkl_binaries(raw_dir, methods):
    linked = {}
    for name in ('mkl','mkl-jit'):
        proof = hashes(raw_dir / (name+'-identity.sha256'))
        expected = {**MKL_SDK, **MKL_SOURCES, name+'.so':sha(methods[name]['sha256'])}
        backend = name+'-backend.so'
        need(set(proof)==set(expected)|{backend} and all(proof[k]==v for k,v in expected.items()),
             'Native MKL libraries, public provider, or fixed build source differs: '+name)
        imports=(raw_dir/(name+'-backend-imports.txt')).read_text()
        symbols=set(re.findall(r'^\s*U\s+([\w]+)',imports,re.MULTILINE))
        need('cblas_sgemm' in symbols and 'dladdr' in symbols, 'No CBLAS binding and loader check: '+name)
        fast={'mkl_sgemm_jit','sgemm_direct'}
        need((fast <= symbols) if name=='mkl-jit' else not (symbols & fast),'Wrong ordinary/JIT ABI: '+name)
        linked[name]={'adapter_sha256':proof[name+'.so'],'backend_sha256':proof[backend]}
    need(linked['mkl']['backend_sha256'] != linked['mkl-jit']['backend_sha256'] and
         linked['mkl']['adapter_sha256'] != linked['mkl-jit']['adapter_sha256'], 'Ordinary and JIT-capable files collapsed')
    return linked


def host_data(raw_dir):
    info = {}
    for line in (raw_dir/'host.txt').read_text().splitlines():
        name, sep, value = line.partition(':')
        if sep:
            info[name.strip()] = value.strip()
    need(info['Architecture'] == 'x86_64', 'Native x86-64 required')
    vendor = info['Vendor ID']
    need(vendor in ('GenuineIntel','AuthenticAMD'), 'Unregistered CPU manufacturer')
    flags = set(info['Flags'].split())
    need(info['Model name'], 'Missing CPU model')
    return vendor, info['Model name'], vendor == 'GenuineIntel' and {'avx2','fma'} <= flags


def geometry(panel_row, page):
    m,n,k,lda,ldb,ldc,ar,ac,br,bc,cr,cc = (int(panel_row[f]) for f in GEOMETRY)
    need(min(m,n,k) > 0 and min(ar,ac,br,bc,cr,cc) >= 0, 'Invalid dimensions or origins')
    need(lda >= ac+k and ldb >= bc+n and ldc >= cc+n, 'Subview crosses its parent row')
    origins = (ar*lda+ac,br*ldb+bc,cr*ldc+cc)
    residues = tuple((4*v)%64 for v in origins)
    guard_cells = ((4*(m+cr+2)*ldc+page-1)//page)*(page//4)-m*n
    region = 24 <= m <= 160 and 1024 <= n <= 4096 and 96 <= k <= 384 and n >= 8*m and m*k <= 49152
    return (m,n,k,lda,ldb,ldc), origins, residues, guard_cells, region and residues[1] == 0


def expected_calls(method, eligible, dims):
    m,n,k,lda,ldb,ldc = dims
    def part(kind, ordinal, width, offset):
        return (kind,ordinal,m,width,k,lda,ldb,ldc,0,offset,offset)
    if eligible and method == 'host-split8':
        return 'provider-split', [part('provider',0,8,0),part('provider',1,n-8,8)]
    if eligible and method == 'host-prefix8':
        return 'vector-prefix', [part('vector-prefix',0,8,0),part('provider',0,n-8,8)]
    return 'forward', [part('provider',0,n,0)]


def provider_data(raw_dir, methods, panel):
    digest_mkl = sha(methods['mkl']['sha256'])
    builds = index(rows(raw_dir/'prefix-builds.csv'), ['method'])
    need(set(builds) == {(name,) for name in METHODS}, 'Wrapper build inventory')
    no_copy = index(rows(raw_dir/'prefix-zero-copy.csv'), ['method'])
    need(set(no_copy)==set(builds) and hashes(raw_dir/'prefix-source.sha256')==PREFIX_SOURCES,
         'Observed source or allocation-import fixtures differ from the reviewed program')
    actual = {}
    distinct_binaries = set()
    for name in METHODS:
        row = builds[name,]
        need(methods[name]['role'] == 'vendor-assisted' and methods[name]['policy'] == 'one-shot' and
             methods[name]['status'] == 'available', 'Unavailable/different ordinary wrapper '+name)
        need(sha(row['ordinary_sha256']) == methods[name]['sha256'] and sha(row['provider_sha256']) == digest_mkl,
             'Timed wrapper and traced provider did not use registered binaries '+name)
        actual[name] = {k:sha(row[k]) for k in ('ordinary_sha256','trace_sha256')}
        distinct_binaries.update(actual[name].values())
        zero = no_copy[name,]
        need(zero['ordinary_sha256']==row['ordinary_sha256'] and zero['alloc_imports']=='0' and
             zero['copy_imports']=='0' and zero['passed']=='1', 'Ordinary wrapper allocator/copy-import receipt '+name)
    need(len(distinct_binaries) == 6 and digest_mkl not in distinct_binaries, 'Distinct ordinary and diagnostic binaries required')
    mkl = methods['mkl']
    need(mkl['role']=='independent' and mkl['policy']=='one-shot' and mkl['status']=='available','Direct ordinary MKL prerequisite')
    build_info = (raw_dir/'mkl-build.txt').read_text()
    linked = set(re.findall(r'\(NEEDED\).*?\[([^]]+)\]', build_info))
    need({s for s in linked if s.startswith('libmkl')} == set(MKL_SDK) and not any('iomp' in s or 'gomp' in s for s in linked), 'Sequential MKL ELF dependencies')
    need('vendor=mkl\n' in build_info and 'version=2026.2.1\n' in build_info and
         'Padded FP32 SGEMM and same-shape changing operands passed' in build_info and
         'direct-call JIT:' not in build_info,'Ordinary MKL build or smoke identity')
    expected = {(name,s['family']) for name in METHODS for s in panel}
    logs = (raw_dir/'prefix-provider.txt').read_text().splitlines()
    seen = set()
    for line in logs:
        fields = line.split(maxsplit=2)
        need(len(fields)==3, 'Malformed sequential provider identity')
        name,family,message=fields
        need('oneMKL LP64 sequential: Intel(R) oneAPI Math Kernel Library Version 2026.2-Product Build 20260915' in message
             and 'libmkl_intel_lp64.so.3)' in message and 'direct-call JIT' not in message, 'Wrong actual provider identity')
        need((name,family) not in seen, 'Duplicate provider link fixture')
        seen.add((name,family))
    need(seen==expected, 'Provider version/ordinary soname missing for a method or layout')
    return digest_mkl, actual


def parity(raw_dir, panel_by_case, bypass):
    records = index(rows(raw_dir/'prefix-bypass-parity.csv'), ('family','layout','mode','seed','method'))
    modes = [('random',str(SEEDS[0])),('random',str(SEEDS[1])),('cancel',str(SEEDS[0])),('scaled',str(SEEDS[1]))]
    target = {(*case,mode,seed,method) for case in bypass for mode,seed in modes for method in ('mkl',*METHODS)}
    need(set(records)==target, 'Incomplete or extraneous ordinary byte-parity evidence')
    elems = 0
    for case in bypass:
        panel = panel_by_case[case]
        extent = int(panel['m'])*int(panel['n'])
        for mode,seed in modes:
            mkl = sha(records[(*case,mode,seed,'mkl')]['sha256'])
            for method in ('mkl',*METHODS):
                r=records[(*case,mode,seed,method)]
                need(r['sha256']==mkl and r['mkl_sha256']==mkl and int(r['checked'])==extent and r['passed']=='1',
                     f'Ordinary direct versus {method} output mismatch {case} {mode}')
                elems += extent
        need(records[(*case,'random',str(SEEDS[0]),'mkl')]['sha256'] != records[(*case,'random',str(SEEDS[1]),'mkl')]['sha256'],
             'The two independent random fixtures repeat an output hash')
    return {'bypass_layouts': len(bypass), 'fixtures_per_layout':4, 'digest_rows':len(records),
            'direct_reference_digests':len(bypass)*4, 'byte_equal_wrapper_digests':len(bypass)*4*len(METHODS),
            'logical_fp32_elements_hashed':elems}


def verify_routes(raw_dir, panel, cpu, method_records):
    """Check public native witness data; return a path-free, JSON-serializable summary."""
    raw_dir = Path(raw_dir)
    panel = list(panel.values() if isinstance(panel,dict) else panel)
    methods = {r['method']:r for r in (method_records.values() if isinstance(method_records,dict) else method_records)}
    need(digest(raw_dir/'panel.csv')==PANEL_SHA and panel==rows(raw_dir/'panel.csv'),'Route panel identity/order')
    panel_by_case = index(panel,('family','layout'))
    cpu = int(cpu)
    need(cpu >= 0, 'Invalid pinned CPU')
    page = int((raw_dir/'page-size.txt').read_text())
    need(page>=64 and page&(page-1)==0, 'Invalid recorded native page size')
    vendor, model, capable = host_data(raw_dir)
    mkl_hash, wrappers = provider_data(raw_dir, methods, panel)
    native = index(rows(raw_dir/'prefix-routes.csv'),('method','family','layout'))
    claim = {(method,*case) for method in METHODS for case in panel_by_case}
    need(set(native)==claim, 'Missing or unregistered route fixture')
    parts = defaultdict(list)
    for r in rows(raw_dir/'prefix-provider-calls.csv'):
        key=r['method'],r['family'],r['layout']
        need(key in claim,'Unknown provider call geometry')
        part=tuple(r[k] if k=='kind' else int(r[k]) for k in SPLIT_FIELDS)
        parts[key].append(part)
    bypass = []
    per_case=[]
    route_counts = Counter()
    checked=guards=0
    worst_error=0.0
    for case,p in panel_by_case.items():
        dims,origins,residues,guard_cells,in_scope=geometry(p,page)
        eligible=capable and in_scope
        if not eligible:
            bypass.append(case)
        m,n=dims[:2]
        summary={}
        for method in METHODS:
            key=(method,*case)
            r=native[key]
            need(all(int(p[k])==int(r[k]) for k in GEOMETRY), 'Route fixture arguments differ from frozen view')
            need(r['cpu_vendor']==vendor and int(r['intel_avx2'])==int(capable) and int(r['eligible'])==int(eligible), f'Eligibility/CPUID evidence on {key}')
            need(int(r['cpu'])==cpu and r['threads']=='1' and int(r['checked'])==m*n and int(r['guard_cells'])==guard_cells,
                 f'Incomplete/pinning diagnostic on {key}')
            need(tuple(int(r[f'{s}_mod64']) for s in 'abc')==residues and r['provider_sha256']==mkl_hash and r['passed']=='1',f'Wrong pointer/ordinary SGEMM digest on {key}')
            need(tuple(int(r[f'{s}_element_offset']) for s in 'abc')==origins,
                 f'Actual full parent origin differs from frozen view {key}')
            route,want=expected_calls(method,eligible,dims)
            tally=Counter(p[0] for p in want)
            need(r['route']==route and int(r['vector_calls'])==tally['vector-prefix'] and int(r['provider_calls'])==tally['provider'],f'Wrong route counts on {key}')
            need(parts[key]==want, f'Observed call dimensions, pointers or leading dimensions differ on {key}')
            errors=[float(r['ordinary_error']),float(r['trace_error'])]
            need(all(0<=e<=0.002 for e in errors), f'Failed ordinary/trace FP64 route check for {key}')
            worst_error=max(worst_error,*errors)
            checked+=2*m*n
            guards+=2*guard_cells
            route_counts[route]+=1
            summary[method]={'route':route,'vector_prefix_calls':tally['vector-prefix'],'mkl_calls':tally['provider']}
        per_case.append({'family':case[0],'layout':case[1],'eligible':eligible,'parent_offsets_floats':list(origins),'residues_bytes':list(residues),'routes':summary})
    need(len(panel)==11 and len(bypass)==(2 if capable else 11),'Registered host/panel coverage')
    return {'vendor':vendor,'model':model,'cpu':cpu,'intel_avx2_fma':capable,'eligible_layouts':len(panel)-len(bypass),
            'route_fixtures':len(native),'route_counts':dict(route_counts),'ordinary_plus_trace_fp32_outputs':checked,
            'ordinary_plus_trace_guard_cells':guards,'maximum_route_absolute_error':worst_error,'mkl_adapter_sha256':mkl_hash,
            'mkl_distribution':'2026.2.1','mkl_runtime_build':'20260915','wrappers':wrappers,'cases':per_case,
            'mkl_binaries':mkl_binaries(raw_dir,methods),'mkl_native_libraries_sha256':MKL_SDK,
            'operation':'FP32 row-major NN, alpha=1, beta=0, LP64, sequential',
            'ordinary_wrapper_explicit_allocator_or_copy_imports':0,
            'parity':parity(raw_dir,panel_by_case,bypass)}
