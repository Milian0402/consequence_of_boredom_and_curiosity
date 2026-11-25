#!/usr/bin/env bash
set -euo pipefail
src=$(cd "${1:?patched source required}" && pwd)
work=$(mkdir -p "${2:?work directory required}" && cd "$2" && pwd)
out=$(mkdir -p "${3:?receipt directory required}" && cd "$3" && pwd)
cpu=${4:?CPU required}
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$work/source" "$work/mock"
cp -R "$src/src" "$src/include" "$src/third_party" "$work/source/"
python3 - "$work/source/src/cob_hk15.inc" <<'PY'
from pathlib import Path
import sys
p=Path(sys.argv[1]);s=p.read_text();i=s.index('static int cob_hk15_shape');j=s.index('static int cob_hk15_try')
assert s[i:j].count('768')==3
p.write_text(s[:i]+s[i:j].replace('768','8').replace('1280','32')+s[j:])
PY
cat > "$work/mock/cblas.h" <<'H'
enum CBLAS_ORDER {CblasRowMajor=101,CblasColMajor=102};
enum CBLAS_TRANSPOSE {CblasNoTrans=111,CblasTrans=112};
int openblas_get_num_threads(void);
void cblas_sgemm(enum CBLAS_ORDER,enum CBLAS_TRANSPOSE,enum CBLAS_TRANSPOSE,int,int,int,
    float,const float*,int,const float*,int,float,float*,int);
H
case $(uname -m) in x86_64) options=(-march=native);; aarch64) options=(-mcpu=native -DCOB_USE_NEON=1);; *) exit 2;; esac
: > "$out/hk-leaf-fallback-tests.txt"
for variant in 0 1; do
    "${CC:-gcc}" -O3 -std=c11 "${options[@]}" -I"$work/mock" -I"$work/source/include" \
        -DCOB_ENABLE_HK15=1 -DCOB_ENABLE_HK15_RAGGED=1 -DCOB_ENABLE_HK15_CBLAS_LEAF=$variant \
        -DCOB_ENABLE_PORTABLE_STRASSEN=1 -DCOB_ENABLE_RECURSIVE_STRASSEN=1 -DCOB_ENABLE_OPENBLAS_LEAF=1 \
        "$work/source/src/gemm.c" "$work/source/src/cob_hk15_ragged.c" "$here/hk-leaf-selftest.c" -lm -o "$work/selftest-$variant"
    taskset -c "$cpu" "$work/selftest-$variant" >> "$out/hk-leaf-fallback-tests.txt"
done
