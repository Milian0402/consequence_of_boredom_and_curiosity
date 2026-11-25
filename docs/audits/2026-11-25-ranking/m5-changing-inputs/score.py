"""Independent offline scoring of the compact M5 single-thread memory-regime comparison."""
import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_csv(path):
    with path.open(newline='') as source:
        reader = csv.DictReader(source)
        require(reader.fieldnames and len(reader.fieldnames) == len(set(reader.fieldnames)), f'{path.name}: invalid header')
        rows = list(reader)
    require(rows and all(None not in row and None not in row.values() for row in rows), f'{path.name}: incomplete records')
    return rows


def index(rows, key_converter):
    indexed = {key_converter(row): row for row in rows}
    require(len(indexed) == len(rows), 'Duplicate primary key')
    return indexed


def number(row, key, positive=False):
    value = float(row[key]); require(math.isfinite(value) and (value > 0 if positive else value >= 0), f'Invalid {key}')
    return value


def ratio(left, right):
    require(left.keys() == right.keys() and left, 'Incomplete paired timings')
    return statistics.median(left[k] / right[k] for k in left)


def dump_csv(path, rows):
    with path.open('w',newline='') as dest:
        writer=csv.DictWriter(dest,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)

PANEL = {
    'fresh-aligned-1': (24,1088,280), 'fresh-aligned-2': (27,2240,96),
    'fresh-aligned-3': (29,2624,216), 'fresh-aligned-4': (32,3904,368),
    'fresh-odd-1': (25,2287,115), 'fresh-odd-2': (26,3487,307),
    'fresh-odd-3': (30,1043,275), 'fresh-odd-4': (31,3135,187),
    'boundary-challenge': (24,1031,383),
}
METHODS = ('A','B','D','Accelerate','OpenBLAS','ArmPL','MaxMulSK','MpGEMM','prepared-LIBXSMM','KleidiAI-fixed','KleidiAI-elastic')
PEERS = METHODS[3:]
SEEDS = {1:1088945771,2:1870216639}
MODES = ('hot','rotate')
THRESHOLD = 128 * 1024**2


def geometry(row):
    f = row['family']
    require(f in PANEL and tuple(int(row[k]) for k in ('m','n','k'))==PANEL[f] and int(row['pad'])==0, 'Unknown geometry '+f)
    return f


def trial(row):
    block=int(row['block'])
    require(block in SEEDS and int(row['seed'])==SEEDS[block], 'Unknown seed or block')
    return geometry(row),block


def process_order():
    families = list(PANEL)
    return [(f, block) for block in (1, 2) for f in (families if block == 1 else families[::-1])]


def verify_chronology(pool_rows, accuracy_rows, sample_rows):
    processes = process_order()
    require([trial(r) for r in pool_rows] == processes, "Pool epoch order differs from the registered traversal")
    actual = [(trial(r), r['method']) for r in accuracy_rows]
    require(actual == [(p, m) for p in processes for m in METHODS], "FP64 epoch order")
    expected = []
    for f, block in processes:
        ordinal = list(PANEL).index(f)
        first = (ordinal + block - 1) % 2
        for mode in (MODES[first], MODES[1-first]):
            for round_id in range(1,12):
                expected.extend((f, block, mode, round_id, pos) for pos in range(1,12))
    actual = [(geometry(r), int(r['block']), r['mode'], int(r['round']), int(r['position'])) for r in sample_rows]
    require(actual == expected, "Epoch hot/rotate order or batch chronology differs from frozen source")


def inputs(directory):
    pool_rows = read_csv(directory/'pool.csv'); accuracy_rows=read_csv(directory/'accuracy.csv'); sample_rows=read_csv(directory/'samples.csv')
    verify_chronology(pool_rows,accuracy_rows,sample_rows)
    pool = index(pool_rows,trial)
    require(set(pool)=={(f,b) for f in PANEL for b in SEEDS}, 'Missing pool receipt')
    summary = dict(slots={},logical_bytes={},resident_pre=[],resident_post=[],resident_low=[],max_abs={},fp64_cells=0,caller_to_wall=[])
    for (f,b),r in pool.items():
        slots=int(r['slots']); m,n,k=PANEL[f]
        logical_each=(m*k+k*n+m*n)*4
        capacity=int(r['logical_bytes']); arena=int(r['arena_bytes']); pages=int(r['pages']); page=int(r['page_bytes'])
        require(slots>1 and capacity==slots*logical_each and capacity>=THRESHOLD and arena>=capacity,'Logical working set mismatch')
        require(page==16384 and pages>0 and pages*page==arena,'Unexpected measured page geometry')
        for col in ('unique_a','unique_b','unique_c'):require(int(r[col])==slots, 'Incomplete distinct pool '+col)
        for col in ('readonly_ab','shared_method_output','same_alias','threads'):require(int(r[col])==1, 'Missing pool qualification '+col)
        summary['slots'][f]=slots;summary['logical_bytes'][f]=capacity
        for col,label in [('resident_pre','resident_pre'),('resident_post','resident_post'),('epoch_min_resident','resident_low')]:
            resident=int(r[col]);require(0<resident<=pages,'Invalid resident page count '+col)
            summary[label].append(resident/pages)
    summary['any_unresident_pages_observed']=any(min(summary[k])<1 for k in ('resident_pre','resident_post','resident_low'))
    summary['residency']=[dict(family=f,block=b,pages=int(r['pages']),pre=int(r['resident_pre']),post=int(r['resident_post']),minimum=int(r['epoch_min_resident'])) for (f,b),r in pool.items()]
    accuracy=index(accuracy_rows,lambda r:(*trial(r),r['method']))
    require(set(accuracy)=={(f,b,m) for f in PANEL for b in SEEDS for m in METHODS},'Missing method or accuracy cell')
    for (f,b,method),r in accuracy.items():
        m,n,_=PANEL[f];slots=int(pool[f,b]['slots']);checked=int(r['checked']);err=number(r,'max_abs')
        require(int(r['slots'])==slots and int(r['guards_ok'])==1 and checked==m*n*slots and err<=.002, 'FP64 or guard failure at '+str((f,b,method)))
        summary['fp64_cells']+=checked;summary['max_abs'][method]=max(summary['max_abs'].get(method,0),err)
    samples=index(sample_rows,lambda r:(*trial(r),r['mode'],int(r['round']),r['method']))
    return pool,samples,summary


def scores(times):
    shapes=[];comparisons=[];changes=[]
    for f in PANEL:
        for clock in ('wall','caller'):
            leaders={}
            for mode in MODES:
                ts=times[f,mode,clock]
                medians={m:statistics.median(ts[m].values()) for m in METHODS}
                lowest=min(medians[m] for m in PEERS)
                leaders[mode]=sorted(m for m in PEERS if medians[m]==lowest)
                fastest=leaders[mode][0]; incumbent={k:math.sqrt(ts['A'][k]*ts['B'][k]) for k in ts['A']}
                incumbent_ratio=ratio(incumbent,ts['D']); field=ratio(ts[fastest],ts['D'])
                shapes.append(dict(family=f,cohort='boundary' if f=='boundary-challenge' else 'original-eight',mode=mode,clock=clock,
                                   shipped_ns=medians['D'],fastest_external_ns=lowest,fastest_external=fastest,
                                   tied_external='|'.join(leaders[mode]),peer_over_shipped=field,incumbent_over_shipped=incumbent_ratio,
                                   alias_a_over_b=ratio(ts['A'],ts['B']),rounds_shipped_faster=sum(ts['D'][k]<ts[fastest][k] for k in ts['D']),
                                   block_1_peer_over_shipped=statistics.median(ts[fastest][k]/ts['D'][k] for k in ts['D'] if k[0]==1),
                                   block_2_peer_over_shipped=statistics.median(ts[fastest][k]/ts['D'][k] for k in ts['D'] if k[0]==2)))
                for m in METHODS:
                    comparisons.append(dict(family=f,mode=mode,clock=clock,method=m,median_ns=medians[m],paired_over_shipped=ratio(ts[m],ts['D'])))
            for m in METHODS:
                changes.append(dict(family=f,clock=clock,method=m,rotate_over_hot=ratio(times[f,'rotate',clock][m],times[f,'hot',clock][m]),
                                    hot_leader='|'.join(leaders['hot']),rotating_leader='|'.join(leaders['rotate'])))
    return shapes,comparisons,changes


def check_schedule(pool,samples,summary):
    total=len(PANEL)*len(SEEDS)*len(MODES)*len(METHODS)**2
    require(len(samples)==total,'Unexpected number of timing rows')
    times=defaultdict(lambda:defaultdict(dict));low=float('inf');high=0.
    for family in PANEL:
        m,n,k=PANEL[family]
        for mode in MODES:
            transitions=Counter();positions=Counter()
            for block,seed in SEEDS.items():
                slots=int(pool[family,block]['slots'])
                require(slots == math.ceil(THRESHOLD/(4*(m*k+k*n+m*n))),'Incorrect slot count')
                for round_id in range(1,12):
                    run=[]
                    for pos in range(1,12):
                        x=pos-1 if block==1 else 11-pos
                        delta=0 if x==0 else ((x+1)//2 if x%2 else 11-x//2)
                        method=METHODS[(delta+round_id-1)%11];key=(family,block,mode,round_id,method)
                        require(key in samples,'Missing scheduled cell '+str(key));r=samples[key]
                        require(int(r['position'])==pos,'Misordered method '+str(key));run.append(method);positions[block,pos,method]+=1
                        start=(seed%slots+17*(round_id-1))%slots
                        step=0
                        if mode=='rotate':
                            step=(seed%slots+2*(round_id-1)+1)%slots
                            while not step or math.gcd(step,slots)!=1:step=(step+1)%slots
                            require(len({(start+j*step)%slots for j in range(slots)})==slots,'Ring repeats an address slot')
                        require(int(r['start_slot'])==start and int(r['step'])==step and int(r['slots'])==slots and int(r['calls'])==slots,
                                'Ring selection differs from the schedule '+str(key))
                        require(int(r['guards_ok'])==1 and r['threads_before']==r['threads_after']=='1','Thread or guard failure '+str(key))
                        wall=number(r,'wall_ns',positive=True);caller=number(r,'caller_ns',positive=True)
                        low=min(low,caller/wall);high=max(high,caller/wall)
                        for clock,ns in [('wall',wall),('caller',caller)]:times[family,mode,clock][method][block,round_id]=ns/slots
                    transitions.update(zip(run,run[1:]))
            require(len(positions)==242 and set(positions.values())=={1},'Unbalanced method positions')
            require(len(transitions)==110 and set(transitions.values())=={2},'Unbalanced predecessors in the two-block design')
    summary['caller_to_wall']=[low,high];summary['timing_batches']=total;summary['timed_gemms']=sum(int(r['calls']) for r in samples.values())
    return times


def run(directory,destination):
    pool,samples,check=inputs(directory)
    times=check_schedule(pool,samples,check)
    shapes,peers,changes=scores(times)
    groups=[]
    for cohort in ('original-eight','boundary'):
        for mode in MODES:
            for clock in ('wall','caller'):
                v=[r for r in shapes if (r['cohort'],r['mode'],r['clock'])==(cohort,mode,clock)]
                groups.append(dict(cohort=cohort,mode=mode,clock=clock,shapes=len(v),wins=sum(r['peer_over_shipped']>1 for r in v),
                                   gm_peer_over_shipped=statistics.geometric_mean(r['peer_over_shipped'] for r in v),
                                   gm_incumbent_over_shipped=statistics.geometric_mean(r['incumbent_over_shipped'] for r in v),
                                   alias_min=min(r['alias_a_over_b'] for r in v),alias_max=max(r['alias_a_over_b'] for r in v)))
    seed_groups = []
    original = [f for f in PANEL if f != 'boundary-challenge']
    for mode in MODES:
        for clock in ('wall','caller'):
            for block in SEEDS:
                seed_cases = []
                for f in original:
                    series = times[f,mode,clock]
                    durations = {method:{r:v for (b,r),v in data.items() if b==block}
                                 for method,data in series.items()}
                    peer = min(PEERS, key=lambda m:(statistics.median(durations[m].values()),m))
                    pooled = next(row for row in shapes if (row['family'],row['mode'],row['clock'])==(f,mode,clock))
                    seed_cases.append(dict(family=f, peer=peer,
                        peer_over_shipped=ratio(durations[peer],durations['D']),
                        pooled_leader_over_shipped=ratio(durations[pooled['fastest_external']],durations['D']),
                        alias_a_over_b=ratio(durations['A'],durations['B'])))
                seed_groups.append(dict(mode=mode,clock=clock,block=block,seed=SEEDS[block],cases=seed_cases,
                    fastest_per_seed_gm=statistics.geometric_mean(r['peer_over_shipped'] for r in seed_cases),
                    pooled_leader_gm=statistics.geometric_mean(r['pooled_leader_over_shipped'] for r in seed_cases)))
    rotate_seeds = [g['fastest_per_seed_gm'] for g in seed_groups if (g['mode'],g['clock']) == ('rotate','wall')]
    rotate_null = {r['family']:r['alias_a_over_b'] for r in shapes if
                   (r['mode'],r['clock'],r['cohort']) == ('rotate','wall','original-eight')}
    null_pass = all(.97 <= ratio_ab <= 1.03 for ratio_ab in rotate_null.values())
    rule=dict(rotating_wall_seed_gms=rotate_seeds, rotating_wall_per_layout_null=rotate_null,
              both_seed_gms_above_one=all(x>1 for x in rotate_seeds), all_eight_nulls_within_three_percent=null_pass,
              passes=all(x>1 for x in rotate_seeds) and null_pass)
    limits=('These paired measurements cover eight prior compact FP32 shapes plus one boundary challenge on this M5 Max. '
            'Each batch uses the same starting address and number of calls for every implementation. The rotating batch visits every distinct slot once. '
            'Mach caller CPU time and endpoint process thread counts accompany wall time. The scheduler controls physical core placement. '
            'Resident virtual pages and a 128 MiB logical footprint establish the address-rotation regime. Hardware cache hit rates, system-level cache residency, '
            'DRAM traffic, and individual core placement were unmeasured. Internal scratch and library-retained state can differ across implementations.')
    hashes={p:hashlib.sha256((directory/p).read_bytes()).hexdigest() for p in ['samples.csv','accuracy.csv','pool.csv']}
    result={'inputs_sha256':hashes,'qualification':check,'groups':groups,'seed_groups':seed_groups,'registered_rotation_rule':rule,'interpretation':limits,
            'unit':'nanoseconds per complete call, from batch nanoseconds divided by calls',
            'score':'For each shape and clock, external leader has the lowest median per-call latency across 22 batches. Paired ratio is the median of 22 external/shipped per-call times. Ties are named; lexicographic first supplies the paired ratio. Eight-shape summaries give each shape equal weight.'}
    destination.mkdir(parents=True,exist_ok=True)
    dump_csv(destination/'independent-shapes.csv',shapes);dump_csv(destination/'independent-peers.csv',peers)
    dump_csv(destination/'independent-rotation.csv',changes);dump_csv(destination/'independent-groups.csv',groups)
    (destination/'independent-summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('raw_directory',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();root=args.raw_directory.resolve()
    out=run(root,args.output.resolve() if args.output else root)
    for row in out['groups']:
        print(row['cohort'],row['mode'],row['clock'],f"{row['wins']}/{row['shapes']} fastest-external wins, {row['gm_peer_over_shipped']:.4f}x")
