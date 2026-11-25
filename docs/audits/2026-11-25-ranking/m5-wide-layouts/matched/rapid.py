from pathlib import Path
import csv,subprocess as sp,os
h=Path(__file__).resolve().parent;p=h.parent;r=p.parents[1];b=h/'reproduction';xs=r/'libxsmm/upstream'
s=(h/'null.c').read_text()
s=s.replace('e->out=matrix(m,ldc);','e->out=count?methods[0].out:matrix(m,ldc);').replace('2000000000u/','14000000u/').replace('if(calls<16)calls=16;if(calls>320)calls=320;','if(calls<1)calls=1;if(calls>8)calls=8;')
mark='    for(int round=0;round<6;++round) for(int pos=0;pos<count;++pos) {'
assert s.count(mark)==1
s=s.replace(mark,'''    if(methods[0].out!=methods[1].out || methods[1].out!=methods[2].out)fail("shared output required");
    unsigned before=threads();
    if(before!=1)fail("single caller required");
    for(int round=0;round<60;++round) for(int pos=0;pos<count;++pos) {''').replace('orders[block==1?round:5-round][pos]','orders[block==1?(round+round/6)%6:5-(round+round/6)%6][pos]')
s=s.replace('prepare_output(e->out,m,n,ldc);\n        unsigned nt0=threads();','\n        unsigned nt0=before;')
start=s.index('        unsigned nt1=threads();');end=s.index('    }\n    return 0;',start)
s=s[:start]+'''        unsigned nt1=before;
        if(round>=54) {
            e->error=fmax(e->error,verify(m,n,ldc,e->out,ref));
            if(memcmp(acopy,a,sa)||memcmp(bcopy,b,sb))fail("input modified");
            check_guard(a,(size_t)m*lda);check_guard(b,(size_t)k*ldb);
        } else {
            int i=(round*7+pos*11)%m,j=(round*151+pos*97)%n;
            double d=fabs(e->out[(size_t)i*ldc+j]-ref[(size_t)i*n+j]);
            if(!isfinite(d)||d>tolerance)fail("spot check");
        }
        printf("%d,%d,%d,%d,%u,%d,%d,%d,%s,%d,%.12g,%.12g,%.12g,%u,%u,%.8f,%zu\\n",m,n,k,pad,seed,block,round+1,pos+1,e->name,calls,wall*1e-9/calls,thr*1e-9/calls,e->error,nt0,nt1,(double)proc/thr,(size_t)(round>=54?m*n:1));
    }
    if(threads()!=1)fail("thread count changed");
    return 0;
}
'''
(h/'rapid.c').write_text(s);sp.run(['cc','-std=c11','-O3','-I'+str(p),'-I'+str(xs/'include'),h/'rapid.c',xs/'lib/libxsmm.a','-o',h/'rapid'],check=True)
t=[v for z in [('A',b/'incumbent.dylib'),('B',b/'incumbent.dylib'),('D',b/'candidate.dylib')] for v in z]
cases=list(csv.DictReader((p/'panel/panel.csv').open()))
with (h/'rapid.csv').open('w') as f,(h/'rapid.log').open('w') as log:
 csv.writer(f).writerow('m n k pad seed block round position method calls warm_seconds caller_seconds max_abs_error threads_before threads_after process_to_caller_cpu checked_cells'.split());f.flush()
 for block,seed in enumerate((427297853,1670820329),1):
  for c in (cases if block==1 else reversed(cases)):
   sp.run([h/'rapid',*[c[k] for k in ('m','n','k','pad')],str(seed),str(block),*t],stdout=f,stderr=log,check=True)
