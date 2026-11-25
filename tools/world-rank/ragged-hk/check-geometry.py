"""Verify the registered core, disjoint fringes, parent extents and testing budget."""
import csv
import sys
from pathlib import Path
panel = Path(__file__).resolve().with_name("panel.csv")
out = csv.writer(sys.stdout, lineterminator="\n")
out.writerow("family core_m core_n core_k child_m child_n child_k rm rn rk child_harmonic hk_child_fmas_before_child_recursion depth_fmas right_fmas bottom_fmas fringe_percent hk_outer_scratch_bytes input_output_parent_bytes fp64_bytes fp64_dgemm_flops a_mod64 b_mod64 c_mod64".split())
for row in csv.DictReader(panel.open()):
    m,n,k,la,lb,lc,ar,ac,br,bc,cr,cc = (int(row[f]) for f in "m n k lda ldb ldc ar ac br bc cr cc".split())
    assert la >= ac+k and lb >= bc+n and lc >= cc+n
    rm,rn,rk = m%2,n%3,k%3
    mc,nc,kc = m-rm,n-rn,k-rk
    mh,nh,kh = mc//2,nc//3,kc//3
    assert 768 <= min(mh,nh,kh) and max(mh,nh,kh) <= 1280
    assert 4*max(mh,nh,kh) <= 5*min(mh,nh,kh)
    depth,right,bottom = mc*nc*rk, mc*rn*k, rm*n*k
    assert depth+right+bottom+mc*nc*kc == m*n*k
    assert 18*mh*nh*kh == mc*nc*kc
    area = mh*nh+mh*kh+nh*kh
    harmonic = 3*mh*nh*kh/area
    scratch = 4*area
    assert kh*nh >= k*rn
    view_bytes = sum(((rows+origin+2)*ld*4+4095)//4096*4096 +8192
                     for rows,origin,ld in ((m,ar,la),(k,br,lb),(m,cr,lc)))
    residues = tuple(4*(r*ld+c)%64 for r,ld,c in ((ar,la,ac),(br,lb,bc),(cr,lc,cc)))
    if row["layout"]=="cropped":assert 0 not in residues and len(set(residues))==3
    out.writerow([row["family"],mc,nc,kc,mh,nh,kh,rm,rn,rk,round(harmonic,4),
                  15*mh*nh*kh,depth,right,bottom,round(100*(depth+right+bottom)/(m*n*k),6),
                  scratch,view_bytes,8*m*n,2*m*n*k,*residues])
