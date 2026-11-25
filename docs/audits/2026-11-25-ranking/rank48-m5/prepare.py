from pathlib import Path
import hashlib,io,os,subprocess,tarfile,json,sys
h=Path(__file__).resolve().parent
bench=Path(os.environ['BENCH_REPO']).resolve()
source='af33d7d4266ff27525839b6ed644929792720c76'
prod='b218beff1eaf8fd606d0934405cdc1e28ecdfe15'
def stage(repo,rev,dest,paths):
    dest.mkdir(exist_ok=True)
    data=subprocess.check_output(['git','-C',repo,'archive',rev,*paths])
    with tarfile.open(fileobj=io.BytesIO(data)) as tf:tf.extractall(dest,filter='data')
base='tools/world-rank/'
stage(bench,source,h/'recipes',[base+n for n in ('release-replay','release-public','medium/source.json')])
recipes=h/'recipes'/base
subprocess.run([sys.executable,recipes/'release-replay/prepare-sources.py',bench,h/'release'],check=True)
subprocess.run([sys.executable,recipes/'release-public/apply-test-backport.py',h/'release'],check=True)
proof=json.loads((recipes/'medium/source.json').read_text())
actual={str(p.relative_to(h/'release')):hashlib.sha256(p.read_bytes()).hexdigest() for p in (h/'release').rglob('*') if p.is_file()}
if proof['revision']!=prod or actual!=proof['release_files']:
    raise ValueError('the public recipe did not reconstruct the complete frozen release source')
stage(bench,source,h/'bench',[base+n for n in ('worker.c','vendor.h','rank48','study.py','aligned_admission.py')])
s=h/'bench'/base
def transform(text,old,new):
    if text.count(old)!=1:raise ValueError(old)
    return text.replace(old,new)
raw=(s/'worker.c').read_text()
raw=transform(raw,'static int single_task(void) { return 1; }\nstatic int pin_cpu(void) { fputs("affinity_cpu=unverified_non_linux_smoke\\n", stderr); return 1; }','''#include <unistd.h>
extern int apple_threads(int);
extern int rank_vendor_init(void);
static int single_task(void) { return apple_threads(getpid())==1; }
static int pin_cpu(void) { return !rank_vendor_init() && single_task(); }
static double cpu_seconds(clockid_t clock) {
    struct timespec t;
    if(clock_gettime(clock,&t)) exit(2);
    return t.tv_sec+t.tv_nsec*1e-9;
}''')
raw=transform(raw,'        double start=seconds();','        if (!single_task()) fatal("multiple process threads before timing");\n        double caller_start=cpu_seconds(CLOCK_THREAD_CPUTIME_ID), process_start=cpu_seconds(CLOCK_PROCESS_CPUTIME_ID);\n        double start=seconds();')
raw=transform(raw,'        double elapsed=seconds()-start;','        double elapsed=seconds()-start;\n        double caller_elapsed=cpu_seconds(CLOCK_THREAD_CPUTIME_ID)-caller_start, process_elapsed=cpu_seconds(CLOCK_PROCESS_CPUTIME_ID)-process_start;')
raw=transform(raw,'printf("TIME,%.12g,%d\\n",elapsed/calls,calls)','printf("TIME,%.12g,%d,%.12g,%.12g\\n",elapsed/calls,calls,caller_elapsed/calls,process_elapsed/calls)')
raw=transform(raw,'if (nonfinite || guards || maximum>0.002) fatal("FP64 or guard rejection");', '''if (nonfinite || guards || maximum>0.002) {
        /* Let the controller record READY and query the still-live process. */
        char reject[16];
        if (fgets(reject,sizeof reject,stdin) && strcmp(reject,"quit\\n")) fatal("ranked rejected method");
        free(original_a);free(original_b);free(c);return;
    }''')
(h/'worker-apple.c').write_text(raw)
study=(s/'study.py').read_text()
start=study.index('def observed_affinity(pid):');end=study.index('def reply(',start)
study=study[:start]+'''def observed_affinity(pid):
    import ctypes as ct, os
    lib=ct.CDLL(os.environ["APPLE_RANK_THREAD_PROBE"])
    count=lib.apple_threads(pid)
    if count!=1: raise ValueError(f"Expected one Mach process thread, observed {count}")
    return "unpinned",count

'''+study[end:]
study=transform(study,'full_peers = ["openblas",','full_peers = ["accelerate",')
study=transform(study,'[("openblas", str(args.openblas.resolve()))]', '[("accelerate", str(args.openblas.resolve()))]')
study=transform(study,'seconds, calls_text = reply(proc, "TIME", args.timeout)','seconds, calls_text, caller, process = reply(proc, "TIME", args.timeout)')
study=transform(study,'resident=int(resident)))','resident=int(resident), caller_cpu_seconds=caller, process_cpu_seconds=process))')
study=transform(study,'"seconds", "cpu", "threads", "resident"], files)', '"seconds", "cpu", "threads", "resident", "caller_cpu_seconds", "process_cpu_seconds"], files)')
(s/'study.py').write_text(study)
