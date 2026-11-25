from pathlib import Path
import hashlib,subprocess as sp,re,json
h=Path(__file__).resolve().parent;b=h/'reproduction';p=h.parent
sha=lambda b:hashlib.sha256(b).hexdigest()
def textcode(path):
 s=sp.check_output(['otool','-s','__TEXT','__text',path],text=True)
 return bytes.fromhex(''.join(''.join(line.split()[1:]) for line in s.splitlines() if re.match(r'^[0-9a-fA-F]{8,16}\s',line)))
out={}
for method in ('incumbent','candidate'):
 d=b/(method+'-build')
 objs={'gemm':d/'CMakeFiles/cob_matmul.dir/src/gemm.c.o','sme':d/'CMakeFiles/cob_strassen_sme_leaf.dir/third_party/maxmulsk/strassen_leaf.c.o'}
 exe=b/(method+'.dylib');archive=b/(method+'-install')/'lib/libcob_matmul.a'
 out[method]={'dylib_sha':sha(exe.read_bytes()),'archive_sha':sha(archive.read_bytes()),'dylib_text_sha':sha(textcode(exe)),'dylib_text_length':len(textcode(exe)),'objects':{n:{'sha':sha(path.read_bytes()),'text_sha':sha(textcode(path)),'text_bytes':len(textcode(path))} for n,path in objs.items()},'driver_flags':(d/'CMakeFiles/cob_matmul.dir/flags.make').read_text().split('C_FLAGS = ')[-1].strip(),'sme_flags':(d/'CMakeFiles/cob_strassen_sme_leaf.dir/flags.make').read_text().split('C_FLAGS = ')[-1].strip()}
assert out['incumbent']['objects']['gemm']['text_bytes']>10000
assert out['incumbent']['objects']['gemm']==out['candidate']['objects']['gemm']
assert all(out['incumbent'][q]==out['candidate'][q] for q in ('driver_flags','sme_flags'))
out['compiler']=sp.check_output(['cc','--version'],text=True).splitlines()[0];out['sdk']=sp.check_output(['xcrun','--sdk','macosx','--show-sdk-version'],text=True).strip()
(h/'toolchain.json').write_text(json.dumps(out,indent=2)+'\n');print('same driver machine code',out['incumbent']['objects']['gemm'])
