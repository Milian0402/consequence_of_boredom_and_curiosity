from pathlib import Path
import subprocess as sp,sys,difflib,tarfile,io,shutil
h=Path(__file__).resolve().parent;r=h.parents[1];repo=Path(sys.argv[1]) if len(sys.argv)>1 else r/'signed-wave/stream-release'; base='a30d71f0aaba4bcceb9ec5c530cffd63e6967cd6'
for mode in ('release','candidate'):
 s=h/(mode+'-source');s.mkdir(exist_ok=True)
 raw=sp.check_output(['git','-C',repo,'archive',base,'src','include','third_party','LICENSE','CMakeLists.txt','Makefile','tests','bench'])
 with tarfile.open(fileobj=io.BytesIO(raw)) as x:x.extractall(s,filter='data')
 if mode=='candidate':
  g=s/'src/gemm.c';v=g.read_text();marker='extern int cob_sgemm_sme_tiny(int m, int n, int k'
  assert v.count(marker)==1
  v=v.replace(marker,'extern int cob_sgemm_sme_medium_wide(int m, int n, int k, const float *a, int lda,\n    const float *b, int ldb, float *c, int ldc);\n'+marker)
  marker='    if (m >= 16 && m <= 96 && n >= 16 && n <= 96 && k >= 16 && k <= 96 &&'
  assert v.count(marker)==1
  v=v.replace(marker,'''    if (m >= 24 && m <= 160 && n >= 1024 && n <= 4096 && k >= 96 && k <= 384 &&
        a && b && c && lda >= k && ldb >= n && ldc >= n && cob_apple_sme2p1_available() &&
        !cob_f32_ranges_overlap(c, (size_t)(m - 1) * ldc + n, a, (size_t)(m - 1) * lda + k) &&
        !cob_f32_ranges_overlap(c, (size_t)(m - 1) * ldc + n, b, (size_t)(k - 1) * ldb + n) &&
        cob_sgemm_sme_medium_wide(m, n, k, a, lda, b, ldb, c, ldc)) return;
'''+marker)
  g.write_text(v);f=s/'third_party/maxmulsk/strassen_leaf.c';f.write_text(f.read_text()+'\n'+(h/'kernel.c').read_text())
  patch=''
  for f in (g,f):
   path=str(f.relative_to(s))
   patch+=''.join(difflib.unified_diff((h/'release-source'/path).read_text().splitlines(True),f.read_text().splitlines(True),'a/'+path,'b/'+path))
  (h/'candidate.patch').write_text(patch)
 b=h/(mode+'-build')
 for cmd in (['cmake','-S',s,'-B',b,'-DCMAKE_BUILD_TYPE=Release','-DCOB_STRASSEN_SME=ON'],['cmake','--build',b,'--target','cob_matmul','-j','2'],['cc','-O3','-dynamiclib','-I'+str(s/'include'),h/'adapter.c','-Wl,-force_load,'+str(b/'libcob_matmul.a'),'-o',h/(mode+'.dylib')]):
  sp.run(cmd,check=True)
