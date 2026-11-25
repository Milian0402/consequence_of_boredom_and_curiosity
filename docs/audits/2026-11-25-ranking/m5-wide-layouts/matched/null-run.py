from pathlib import Path
import os,csv,subprocess as sp,sys,hashlib,json
h=Path(__file__).resolve().parent;p=h.parent;r=p.parents[1]
build=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else h/'reproduction'
xs=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else r/'libxsmm/upstream'
s=(p/'bench.c').read_text()
mark='    int calls=(int)'
assert s.count(mark)==1
s=s.replace(mark,'''    if(count!=3||strcmp(methods[0].name,"A")||strcmp(methods[1].name,"B")||strcmp(methods[2].name,"D"))fail("3 aliases A/B/D expected");
    if(!methods[0].call || methods[0].call!=methods[1].call || methods[0].call==methods[2].call)fail("function identity check failed");
    fprintf(stderr,"A=B; D distinct, %d %d %d %d %u\\n",m,n,k,pad,seed);
    int calls=(int)''')
start=s.index('    const int rounds = getenv("LAYOUT_TRAIN")');end=s.index('        entry *e=methods+slot;',start)
s=s[:start]+'''    static const int orders[6][3]={{0,1,2},{1,2,0},{2,0,1},{2,1,0},{1,0,2},{0,2,1}};
    for(int round=0;round<6;++round) for(int pos=0;pos<count;++pos) {
        int slot=orders[block==1?round:5-round][pos];
'''+s[end:]
(h/'null.c').write_text(s)
sp.run(['cc','-std=c11','-O3','-I'+str(p),'-I'+str(xs/'include'),h/'null.c',xs/'lib/libxsmm.a','-o',h/'null'],check=True)
keys={'A':build/'incumbent.dylib','B':build/'incumbent.dylib','D':build/'candidate.dylib'}
assert keys['A'].samefile(keys['B'])
cases=list(csv.DictReader((p/'panel/panel.csv').open()))
with (h/'null.csv').open('w') as f,(h/'null.log').open('w') as log:
 csv.writer(f).writerow(next(csv.reader((p/'holdout.csv').open())));f.flush()
 for block,seed in enumerate((427297853,1670820329),1):
  for c in (cases if block==1 else reversed(cases)):
   args=[h/'null',*[c[k] for k in ('m','n','k','pad')],str(seed),str(block),*[v for kv in keys.items() for v in kv]]
   sp.run(args,check=True,stdout=f,stderr=log)
