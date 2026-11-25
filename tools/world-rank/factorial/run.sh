#!/usr/bin/env bash
# Expects public a30d71f and sequential oneMKL from the existing vendors/build.sh.
set -euo pipefail
release=$(cd "$1" && pwd);vendor=$(cd "$2" && pwd);out=$(mkdir -p "$3" && cd "$3" && pwd)
dir=$(cd "$(dirname "$0")" && pwd);common=$(dirname "$dir");repo=$(git -C "$common" rev-parse --show-toplevel)
build=$(mktemp -d "${RUNNER_TEMP:-/tmp}/gemm-factorial.XXXXXXXX");sdk=$build/sdk
rev=a30d71f0aaba4bcceb9ec5c530cffd63e6967cd6
ref=bce9e5367a2b1b414207460f635ac498663b213f
[[ $(uname -sm) == 'Linux x86_64' ]]
[[ $(git -C "$release" rev-parse HEAD) == "$rev" && $(git -C "$release" rev-parse HEAD^{tree}) == 18f22622de1a4b1b7aebbdb66985741aa38c6e9e ]]
git -C "$release" diff --quiet HEAD --
[[ -s "$vendor/mkl.so" ]]
export OPENBLAS_NUM_THREADS=1 GOTO_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 MKL_NUM_THREADS=1 MKL_DYNAMIC=FALSE BLIS_NUM_THREADS=1
cpu=$(python3 -c 'import os;print(min(os.sched_getaffinity(0)))')
getconf PAGE_SIZE > "$out/page-size.txt"
cp "$dir"/{panel.csv,protocol.md,factors.json,estimands.md,historical-endpoints.csv,score.py} "$out/"
git -C "$repo" show "$ref:tools/world-rank/views/panel.csv" > "$out/previous-panel.csv"
{ lscpu; gcc --version; } > "$out/host.txt"
cmake -S "$release" -B "$build/cmake" -DCMAKE_C_COMPILER=gcc -DCMAKE_C_FLAGS='-march=x86-64 -masm=att' -DCMAKE_BUILD_TYPE=Release \
    -DCOB_BUILD_TESTS=ON -DCOB_BUILD_BENCH=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_INSTALL_LIBDIR=lib -DCOB_ENABLE_X86_STREAM_B=ON > "$out/configure.txt"
cmake --build "$build/cmake" --parallel 2 > "$out/build.txt" 2>&1 || { tail -40 "$out/build.txt";exit 1; }
taskset -c "$cpu" ctest --test-dir "$build/cmake" --output-on-failure > "$out/tests.txt" || { cat "$out/tests.txt";exit 1; }
cmake --install "$build/cmake" --prefix "$sdk" > "$out/install.txt"
archive=$sdk/lib/libcob_matmul.a
cmp "$archive" "$build/cmake/libcob_matmul.a";cmp "$sdk/include/cob_gemm.h" "$release/include/cob_gemm.h"
python3 -B "$common/stream-release/flags.py" "$build/cmake/compile_commands.json" cmake-att "$out/compile-flags.json"
gcc -std=c11 -O2 "$common/stream/isa.c" -o "$build/isa"
taskset -c "$cpu" "$build/isa" > "$out/isa.txt"
flags=(-std=c11 -O3 -march=x86-64 -ffunction-sections -fdata-sections -Wl,--gc-sections -I"$sdk/include"
       -Dhybrid_sgemm_rowmajor=cob_sgemm_rowmajor -Drank_hybrid_next=cob_sgemm_rowmajor)
gcc "${flags[@]}" "$dir/worker.c" "$archive" -ldl -lm -o "$build/stream.worker"
gcc "${flags[@]}" "$dir/worker.c" "$common/frontier/witness.c" -Wl,--wrap=cob_sgemm_rowmajor,--wrap=posix_memalign,--wrap=cob_stream_b_kernel_avx2,--wrap=cob_stream_b_kernel_avx512 "$archive" -ldl -lm -o "$build/stream.trace.worker"
nm --defined-only "$build/stream.worker" > "$build/symbols"
grep -E ' T cob_sgemm_rowmajor$' "$build/symbols" > "$out/public-symbol.txt"
if grep -Eq ' T __wrap_' "$build/symbols"; then exit 1; fi
ldd "$build/stream.worker" > "$out/stream-libraries.txt"
if grep -Ei 'lib(mkl|openblas|blas|gomp|iomp)|not found' "$out/stream-libraries.txt";then exit 1;fi
cp "$vendor/mkl-build.txt" "$out/"
hash(){ sha256sum "$1" | cut -d' ' -f1; }
printf 'method,worker_sha256,archive_sha256,adapter_sha256\nstream,%s,%s,none\nmkl,%s,none,%s\n' \
    "$(hash "$build/stream.worker")" "$(hash "$archive")" "$(hash "$build/stream.worker")" "$(hash "$vendor/mkl.so")" > "$out/binaries.csv"
python3 -B - "$out" "$dir" "$rev" "$ref" "$(git -C "$repo" rev-parse HEAD)" "$cpu" "$(hash "$build/stream.trace.worker")" <<'PY'
from pathlib import Path
import sys,os,hashlib,json
out,source=map(Path,sys.argv[1:3])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
data=dict(release=sys.argv[3],tree="18f22622de1a4b1b7aebbdb66985741aa38c6e9e",historical_benchmark=sys.argv[4],benchmark=sys.argv[5],cpu=int(sys.argv[6]),
          trace_worker_sha256=sys.argv[7],reference_panel_sha256=sha(out/"previous-panel.csv"),panel_sha256=sha(out/"panel.csv"),
          run=os.environ["GITHUB_RUN_ID"],attempt=os.environ["GITHUB_RUN_ATTEMPT"],source_sha256={p.name:sha(p) for p in source.iterdir() if p.is_file()})
if data["panel_sha256"]!="5742b14e8e2fd71993ce98070fabf85b648f1ad69efb7f16ff07f2a8eb246631":raise ValueError("Wrong panel")
(out/"sources.json").write_text(json.dumps(data,indent=2)+"\n")
PY
taskset -c "$cpu" python3 -B "$dir/study.py" "$build" "$vendor" "$out" "$cpu"
python3 -B "$dir/score.py" "$out"
