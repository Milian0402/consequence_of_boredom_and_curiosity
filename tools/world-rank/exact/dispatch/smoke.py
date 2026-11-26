"""Local structural regressions for the integer classifier and dynamically selected endpoints (no timing)."""
import ctypes as ct
import json
import math
import os
from pathlib import Path
import random
import struct
import subprocess
import sys
import tempfile


HERE=Path(__file__).resolve().parent
FLOAT_PTR=ct.POINTER(ct.c_float)
ARGS=[ct.c_int]*3+[FLOAT_PTR,ct.c_int,FLOAT_PTR,ct.c_int,FLOAT_PTR,ct.c_int]
TEST_PROVIDER=r'''
#include <stddef.h>
#include <stdlib.h>
int rank_vendor_init(void){return 0;}
void rank_vendor_sgemm(int m,int n,int k,const float *a,int la,const float *b,int lb,float *c,int lc){
    for(int i=0;i<m;++i)for(int j=0;j<n;++j){
        double s=0;
        for(int l=0;l<k;++l)s+=(double)a[(size_t)i*la+l]*b[(size_t)l*lb+j];
        c[(size_t)i*lc+j]=(float)s;
    }
}
#ifdef TEST_FAST
int precision_last_route(void){return 1;}
int precision_last_products(void){return 1;}
int precision_last_plan_count(void){return 1;}
int precision_last_plan_native(int i){return i==0 && !getenv("SIMULATE_NONNATIVE");}
const char *precision_last_plan_impl(int i){return i==0 ? "test-double-only" : "";}
#endif
'''


def classify(dll):
    scalar=dll.check_word
    scalar.argtypes=[ct.c_uint32];scalar.restype=ct.c_int
    enc=lambda x:int.from_bytes(struct.pack('<f',x),'little')
    words={0x80000000,0,1,0x007fffff,0x7f800000,0x7fc00001,0x7f800001}
    for i in range(-257,258):
        b=enc(i)
        words.update((b,(b-1)&0xffffffff,(b+1)&0xffffffff))
    words.update(i<<16 for i in range(65536))
    rnd=random.Random(1979963)
    words.update(rnd.getrandbits(32) for _ in range(50000))
    for word in words:
        x=struct.unpack('<f',word.to_bytes(4,'little'))[0]
        expect=int(abs(x)) if math.isfinite(x) and abs(x)<=256 and x.is_integer() else -1
        assert scalar(word)==expect,(hex(word),x,expect,scalar(word))
    return len(words)


def child(folder,mode):
    dll=ct.CDLL(str(folder/'dispatch.so'))
    assert dll.rank_vendor_init()==0
    dll.rank_vendor_sgemm.argtypes=ARGS
    dll.exact_last_inspected.restype=ct.c_ulonglong
    dll.exact_last_reason.restype=ct.c_char_p
    dll.exact_last_impl.restype=ct.c_char_p
    ref=ct.CDLL(str(folder/'ordinary.so'));ref.rank_vendor_sgemm.argtypes=ARGS
    cases=0
    def perform(k,aval,bval,reason,amax,bmax,mutate=None,scanned=None,n=3):
        nonlocal cases
        m=2;la=k+2;lb=n+3;lc=n+4
        a=(ct.c_float*(m*la+5))(*([17.5]*(m*la+5)))
        b=(ct.c_float*(k*lb+8))(*([float('nan')]*(k*lb+8)))
        c=(ct.c_float*(m*lc+8))(*([9.]*(m*lc+8)));oracle=type(c)(*c)
        for i in range(m):
            for l in range(k):a[3+i*la+l]=aval
        for l in range(k):
            for j in range(n):b[5+l*lb+j]=bval
        if mutate:mutate(a,b,la,lb)
        ai,bi=bytes(a),bytes(b)
        ptr=lambda buf,k:ct.cast(ct.byref(buf,k*4),FLOAT_PTR)
        def run(lib,out):lib.rank_vendor_sgemm(m,n,k,ptr(a,3),la,ptr(b,5),lb,ptr(out,2),lc)
        run(ref,oracle);run(dll,c)
        assert (bytes(c),bytes(a),bytes(b))==(bytes(oracle),ai,bi)
        observed=dll.exact_last_reason().decode()
        if mode in ('disabled','absent','self','unverified-provider'):
            reason={'disabled':'native-disabled','absent':'fast-plugin-unavailable','self':'fast-plugin-unavailable','unverified-provider':'fast-native-unverified'}[mode]
            amax=bmax=-1
        nonnative=(mode=='missed-route' and reason=='native-exact')
        if nonnative:reason='fast-native-unverified'
        assert observed==reason,(mode,k,observed,reason)
        native=reason=='native-exact'
        examined=k*(m+n) if scanned is None else scanned
        if k==0 or k>4096 or mode not in ('enabled','missed-route'): examined=0
        route=1 if native else 3 if reason in ('native-disabled','fast-plugin-unavailable','fast-native-unverified') else 2
        assert (dll.exact_last_amax(),dll.exact_last_bmax(),dll.exact_last_inspected(),dll.exact_last_route(),
                dll.exact_last_fast_calls(),dll.exact_last_fallback_calls(),dll.exact_last_products(),dll.exact_last_native()) == (
                amax,bmax,examined,route,int(native or nonnative),int(not native),int(native or nonnative),int(native))
        if native:assert dll.exact_last_impl()==b'test-double-only'
        cases+=1
    perform(256,-256,256,'native-exact',256,256)
    perform(257,256,256,'accumulator-bound',256,256)
    perform(4096,63,-63,'native-exact',63,63)
    perform(4097,1,1,'depth-outside',-1,-1)
    perform(0,0,0,'empty-or-invalid-view',-1,-1)
    perform(29,-0.,-256,'native-exact',0,256)
    perform(39,256,1,'integer-domain',-1,-1,lambda a,b,la,lb:a.__setitem__(3,256.5),scanned=39)
    perform(39,1,256,'integer-domain',1,-1,lambda a,b,la,lb:b.__setitem__(5+38*lb+2,float('nan')))
    perform(21,1,1,'integer-domain',-1,-1,lambda a,b,la,lb:(a.__setitem__(3,float('inf')),b.__setitem__(5,1.5)),scanned=21)
    perform(199,1,1,'integer-domain',-1,-1,lambda a,b,la,lb:a.__setitem__(3,1.5),scanned=64)
    perform(199,1,1,'integer-domain',-1,-1,lambda a,b,la,lb:a.__setitem__(3+64,1.5),scanned=128)
    perform(199,1,1,'integer-domain',-1,-1,lambda a,b,la,lb:a.__setitem__(3+la+198,1.5),scanned=398)
    perform(3,1,1,'integer-domain',1,-1,lambda a,b,la,lb:b.__setitem__(5+lb+64,1.5),scanned=6+70+70,n=70)
    print(json.dumps(dict(mode=mode,whole_calls=cases,scalar_words=classify(dll) if mode=='enabled' else 0)))


def main():
    cc=os.environ.get('CC','cc')
    with tempfile.TemporaryDirectory() as name:
        folder=Path(name)
        stub=folder/'provider.c';stub.write_text(TEST_PROVIDER)
        source=folder/'guard-test.c'
        source.write_text('#include "dispatch.c"\nint check_word(uint32_t u){float x; memcpy(&x,&u,sizeof x);unsigned long long n=0;return maximum_integer(&x,1,1,1,&n);}\n')
        common=[cc,'-std=c11','-O3','-fPIC','-shared','-Wall','-Wextra','-Werror']
        for fast, dest in ((False,'ordinary.so'),(True,'one.so')):
            subprocess.run([*common, *(['-DTEST_FAST'] if fast else []),str(stub),'-o',str(folder/dest)],check=True)
        extra=['-ldl'] if sys.platform.startswith('linux') else []
        subprocess.run([*common,'-I',str(HERE),str(source),'-o',str(folder/'dispatch.so'),'-pthread',*extra],check=True)
        for mode in ('enabled','disabled','absent','self','unverified-provider','missed-route'):
            env=dict(os.environ,EXACT_FALLBACK_PLUGIN=str(folder/'ordinary.so'),EXACT_FAST_PLUGIN=str(folder/'one.so'),
                     EXACT_NATIVE_ENABLED='0' if mode=='disabled' else '1')
            if mode=='absent': env.pop('EXACT_FAST_PLUGIN')
            if mode=='self': env['EXACT_FAST_PLUGIN']=str(folder/'dispatch.so')
            if mode=='unverified-provider': env['EXACT_FAST_PLUGIN']=env['EXACT_FALLBACK_PLUGIN']
            if mode=='missed-route': env['SIMULATE_NONNATIVE']='1'
            subprocess.run([sys.executable,__file__,'--child',str(folder),mode],env=env,check=True)


if __name__=='__main__':
    if len(sys.argv)>1: child(Path(sys.argv[2]),sys.argv[3])
    else: main()
