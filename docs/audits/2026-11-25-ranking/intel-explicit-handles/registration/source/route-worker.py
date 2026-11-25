"""Isolated diagnostic process. Allocation, FP64 checks and library observations occur outside ranking."""
import ctypes as ct
import json
import os
from pathlib import Path
import sys

build=Path(sys.argv[1]).resolve()
method=sys.argv[2]
origin=list(map(int,sys.argv[3:]))
prepared=method.endswith("-prepared")
P=ct.c_void_p
I=ct.c_int
U=ct.c_size_t
D=ct.c_double
F=ct.c_float
class Trace(ct.Structure):
    _fields_=[(name,I) for name in "creates gets calls destroys create_status execution_status destroy_status layout transa transb m n k lda ldb ldc".split()]+[(name,F) for name in ("alpha","beta")]+[(name,U) for name in "a b c handle kernel".split()]

def bind(lib,name,result,args):
    f=getattr(lib,name);f.restype=result;f.argtypes=args;return f

def insistent(ok, why):
    if not ok: raise ValueError(f"{method} {origin}: {why}")

def single():
    insistent(len(os.sched_getaffinity(0))==1 and len(list(Path('/proc/self/task').iterdir()))==1,'one native task')

fixture=ct.CDLL(str(build/'handle-fixture.so'))
new=bind(fixture,'fixture_new',P,[I]*13)
reset=bind(fixture,'fixture_reset',None,[P])
free=bind(fixture,'fixture_free',None,[P])
addr=bind(fixture,'fixture_addresses',None,[P,ct.POINTER(U)])
check=bind(fixture,'fixture_check',I,[P,ct.POINTER(U),ct.POINTER(U),ct.POINTER(D)])
seen=[]
def make(g,seed):
    f=new(*g,seed);seen.append(f);v=(U*3)();addr(f,v)
    return f,list(v)

def api(trace):
    lib=ct.CDLL(str(build/(method+('.trace.so' if trace else '.so'))))
    insistent(bind(lib,'rank_vendor_init',I,[])()==0,'real serial provider initialization')
    return lib
normal=api(False);traced=api(True)
abi=[I,I,I,P,I,P,I,P,I]
def product(lib,g,p):
    return bind(lib,'rank_vendor_sgemm',None,abi)(*g[:3],p[0],g[3],p[1],g[4],p[2],g[5])
def prepare(lib,g):
    return bind(lib,'rank_handle_prepare',I,[I]*6)(*g[:6])
def release(lib):
    return bind(lib,'rank_handle_release',I,[])()
def status(lib):
    return bind(lib,'rank_handle_last_status',I,[])()
def verify(f):
    c,h,e=U(),U(),D();insistent(check(f,ct.byref(c),ct.byref(h),ct.byref(e)),'full FP64 or guards')
    return c.value,h.value,e.value
treset=bind(traced,'rank_handle_trace_reset',None,[])
read=bind(traced,'rank_handle_trace_read',None,[ct.POINTER(Trace)])
fault=bind(traced,'rank_handle_fault',None,[I])
single()
if not origin:
    # Per-host diagnostic for all six independently matched keys and both errors.
    g=[16,17,18,32,32,32,0,0,0,0,0,0];f,p=make(g,3)
    faults=[];keys=[]
    if prepared:
        for j,name in enumerate(['m','n','k','lda','ldb','ldc']):
            if j==0: insistent(prepare(traced,g) in (0,1),'initial key')
            changed=g.copy();changed[j]+=1
            for label,shape in [('change',changed),('restore',g)]:
                treset();st=prepare(traced,shape);t=Trace();read(ct.byref(t))
                insistent(st in (0,1) and (t.creates,t.gets,t.calls,t.destroys)==(1,1,0,1),'key change')
                keys.append(dict(key=name,step=label,status=st,creates=t.creates,gets=t.gets,calls=t.calls,destroys=t.destroys))
        insistent(release(traced)==0,'reset keys')
    for scenario in (1,2):
        treset();fault(scenario)
        product(traced,g,p)
        rc=status(traced);t=Trace();read(ct.byref(t))
        insistent(rc==(-2 if scenario==1 else -3) and t.calls==0 and t.destroys==(scenario==2),'injected failure cleanup')
        poison=ct.cast(p[2],ct.POINTER(F))
        insistent(all(poison[i*g[5]+j]!=poison[i*g[5]+j] for i in range(g[0]) for j in range(g[1])),'failed operation marked')
        fault(0);reset(f);product(traced,g,p);verify(f)
        insistent(status(traced) in (0,1),'recovery')
        if prepared: insistent(release(traced)==0,'cleanup')
        faults.append(dict(scenario='create-error' if scenario==1 else 'null-callable',status=rc,creates=t.creates,gets=t.gets,calls=t.calls,destroys=t.destroys,recovered=1))
    print(json.dumps(dict(keys=keys,faults=faults)));sys.exit(0)

changed=origin.copy();changed[5]+=1
one,p1=make(origin,7);two,p2=make(origin,29);three,p3=make(changed,53)
insistent(all(a!=b for a,b in zip(p1,p2)),'independent operand pointers')
g_fields="m n k lda ldb ldc".split()
records=[];last_handle=0;last_kernel=0
insistent(status(normal)==-1 and status(traced)==-1,'status before first product')
def excluded(stage,ns,ts,t):
    reason={-2:'create-error-or-null-handle',-3:'null-callable',-4:'destroy-error',-5:'invalid-geometry'}.get(min(ns,ts),'unexpected-status')
    print(json.dumps(dict(excluded=True,stage=stage,reason=reason,ordinary_status=ns,trace_status=ts,create_status=t.create_status,
                          execution_status=t.execution_status,destroy_status=t.destroy_status,creates=t.creates,gets=t.gets,calls=t.calls,destroys=t.destroys)))
    sys.exit(0)

def stage(name,g,f,p,count,kind='multiply'):
    global last_handle,last_kernel
    treset()
    checked=guards=0;error=ordinary_error=0.0
    if kind=='prepare':
        ns=prepare(normal,g);ts=prepare(traced,g)
        if min(ns,ts)<0:
            t=Trace();read(ct.byref(t));excluded(name,ns,ts,t)
    elif kind=='release':
        insistent(release(normal)==0 and release(traced)==0,'release')
        ns,ts=status(normal),status(traced)
    else:
        reset(f);product(normal,g,p);ns=status(normal)
        if ns>=0:checked,guards,ordinary_error=verify(f)
        reset(f);product(traced,g,p);ts=status(traced)
        if min(ns,ts)<0:
            t=Trace();read(ct.byref(t));excluded(name,ns,ts,t)
        c2,h2,error=verify(f)
        insistent((c2,h2)==(checked,guards),'ordinary/trace coverage')
    single()
    insistent(ns==ts and ts in (0,1),'matching ordinary/trace status')
    t=Trace();read(ct.byref(t))
    insistent((t.creates,t.gets,t.calls,t.destroys)==count,'actual lifecycle counts')
    if t.creates:
        insistent((t.layout,t.transa,t.transb,t.alpha,t.beta)==(101,111,111,1,0) and [getattr(t,k) for k in g_fields]==g[:6] and t.create_status==ts,'creation ABI')
    if t.calls:
        insistent([t.a,t.b,t.c]==p and t.execution_status==ts,'actual callable/pointers')
    if t.destroys:insistent(t.destroy_status==0,'teardown status')
    owner=bind(traced,'rank_handle_callable_owner',ct.c_char_p,[])().decode() or 'none'
    perms=bind(traced,'rank_handle_callable_perms',ct.c_char_p,[])().decode() or 'none'
    if t.kernel:insistent('x' in perms,'executable pointer')
    match_h=int(bool(t.handle) and t.handle==last_handle)
    match_k=int(bool(t.kernel) and t.kernel==last_kernel)
    if prepared and name in ('first','new-pointers'):
        insistent(match_h and match_k,'retained identical handle and entry')
    if prepared and name in ('changed-ldc','restore'):
        insistent(not match_h,'old key replaced')
    if t.handle: last_handle,last_kernel=t.handle,t.kernel
    fields=dict(zip(g_fields,g[:6]));fields.update(dict(stage=name,creates=t.creates,gets=t.gets,calls=t.calls,destroys=t.destroys,
        create_status=t.create_status,execution_status=t.execution_status,destroy_status=t.destroy_status,ordinary_status=ns,trace_status=ts,
        status_name='MKL_JIT_SUCCESS' if ts==0 else 'MKL_NO_JIT',order=t.layout,transa=t.transa,transb=t.transb,alpha=t.alpha,beta=t.beta,
        checked=checked,guards=guards,ordinary_error=ordinary_error,trace_error=error,callable=int(bool(t.kernel)),
        callable_owner=owner,callable_permissions=perms,same_handle=match_h,same_kernel=match_k,operands_checked=t.calls,
        a_mod64=p[0]%64,b_mod64=p[1]%64,c_mod64=p[2]%64,threads=1,passed=1))
    records.append(fields)
if prepared:stage('prepare',origin,one,p1,(1,1,0,0),'prepare')
for name,g,f,p in [('first',origin,one,p1),('new-pointers',origin,two,p2),('changed-ldc',changed,three,p3),('restore',origin,one,p1)]:
    count=(1,1,1,1) if not prepared or name in ('changed-ldc','restore') else (0,0,1,0)
    stage(name,g,f,p,count)
if prepared:stage('release',origin,one,p1,(0,0,0,1),'release')
for f in seen:free(f)
print(json.dumps(records))
