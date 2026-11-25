#!/usr/bin/env bash
set -euo pipefail
vendor=$(cd "$1" && pwd)
out=$(mkdir -p "$2" && cd "$2" && pwd)
repo=$(pwd); common=$repo/tools/world-rank; here=$common/rank48-confirm
build=$(mktemp -d "${RUNNER_TEMP:-/tmp}/rank48-arm.XXXXXX"); trap 'rm -rf "$build"' EXIT
[[ $(uname -sm) == 'Linux aarch64' ]]
cc=${CC:-gcc}
cpu=$(python3 -c 'import os; print(min(os.sched_getaffinity(0)))')
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 BLIS_NUM_THREADS=1 RANK_VERIFY_INPUTS=1 RANK_FINITE_LEAF="$vendor/openblas.so"
unset RANK_MEDIUM_BATCH
cp "$here"/{panel.csv,protocol.md} "$out/"
cp "$vendor"/{openblas-build,armpl-build,fastmmw-768-build,fastmmw-1536-build}.txt "$out/"
lscpu > "$out/cpu.txt"; git rev-parse HEAD > "$out/commit.txt"; "$cc" --version > "$out/compiler.txt"
(cd "$repo" && git diff --exit-code af33d7d4266ff27525839b6ed644929792720c76 -- tools/world-rank/rank48/adapter.c tools/world-rank/rank48/generate.py tools/world-rank/rank48/schedules tools/world-rank/rank48/check.c)
(cd "$common/rank48" && sha256sum schedules/*.slp adapter.c generate.py check.c) > "$out/finite-source.sha256"
python3 -B "$common/release-replay/prepare-sources.py" "$repo" "$build/source" > "$out/release-source.txt"
python3 -B "$common/release-public/apply-test-backport.py" "$build/source" >> "$out/release-source.txt"
python3 - "$common/medium/source.json" "$build/source" >> "$out/release-source.txt" <<'PY'
from pathlib import Path
import hashlib,json,sys
m=json.loads(Path(sys.argv[1]).read_text()); r=Path(sys.argv[2]); files={p.relative_to(r).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in r.rglob('*') if p.is_file()}; assert files==m['release_files']; print('Verified ordinary b218bef release inventory:',len(files),'original files')
PY
read -r prefix < "$vendor/openblas-prefix.txt"; test -s "$prefix/lib/libopenblas.a"
mkdir "$build/pkgconfig"; export PKG_CONFIG_PATH='' PKG_CONFIG_LIBDIR=$build/pkgconfig
cat > "$build/pkgconfig/openblas.pc" <<PC
prefix=$prefix
includedir=\${prefix}/include
libdir=\${prefix}/lib
Name: OpenBLAS
Description: Sequential 0.3.34
Version: 0.3.34
Libs: -L\${libdir} -lopenblas -lpthread -lm
Cflags: -I\${includedir}
PC
cmake -S "$build/source" -B "$build/cmake" -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_C_FLAGS=-mcpu=native \
  -DCOB_BUILD_BENCH=OFF -DCOB_BUILD_TESTS=ON -DCOB_ENABLE_PORTABLE_STRASSEN=ON -DCOB_ENABLE_RECURSIVE_STRASSEN=ON \
  -DCOB_ENABLE_ALIGNED_WINOGRAD=ON -DCOB_BLAS_LEAF=OPENBLAS "-DCOB_BLAS_ROOT=$prefix" > "$out/configure.log"
cmake --build "$build/cmake" --parallel 3 > "$out/build.log"
taskset -c "$cpu" ctest --test-dir "$build/cmake" --output-on-failure | tee "$out/ctest.txt"
archive=$build/cmake/libcob_matmul.a
flags=(-O3 -mcpu=native -std=c11 -DCOB_ENABLE_OPENBLAS_LEAF=1 -I"$build/source/include" -I"$common" -I"$prefix/include"
    -Dhybrid_sgemm_rowmajor=cob_sgemm_rowmajor -Drank_hybrid_next=cob_sgemm_rowmajor)
libs=("$archive" "$prefix/lib/libopenblas.a" -ldl -lpthread -lm)
"$cc" "${flags[@]}" "$common/worker.c" "${libs[@]}" -o "$build/release.worker"
"$cc" "${flags[@]}" -DSIGNED_MODE=0 "$common/worker.c" "$common/signed/trace.c" -Wl,--wrap=cblas_sgemm,--wrap=cob_sgemm_rowmajor "${libs[@]}" -o "$build/release.trace.worker"
ldd "$build/release.worker" > "$out/release-ldd.txt"
if grep -Eq 'lib(openblas|gomp|iomp)|not found' "$out/release-ldd.txt"; then exit 1; fi
nm --defined-only "$build/release.worker" > "$build/nm.txt"
grep -E '[[:space:]]T cob_sgemm_rowmajor$' "$build/nm.txt" > "$out/ordinary-symbol.txt"
if grep -E '[[:space:]]T __wrap_(cblas_sgemm|cob_sgemm_rowmajor)$' "$build/nm.txt"; then exit 1; fi
cat "$build/cmake/CMakeFiles/cob_matmul.dir/flags.make" > "$out/flags.txt"
grep -F "$prefix/lib/libopenblas.a" "$build/cmake/CMakeFiles/cob_gemm_test.dir/link.txt" >> "$out/flags.txt"
sha256sum "$archive" "$prefix/lib/libopenblas.a" "$build"/{release.worker,release.trace.worker} "$vendor"/{openblas,armpl,fastmmw-768,fastmmw-1536}.so > "$out/build-sha256.txt"
taskset -c "$cpu" python3 -B "$here/release-routes.py" "$here/panel.csv" "$build" "$out/release-routes.csv"
taskset -c "$cpu" python3 -B "$common/signed/external-routes.py" "$here/panel.csv" "$vendor" "$out/fastmmw-routes.csv"
"$cc" -std=c11 -O2 "$common/rank48/check.c" -ldl -o "$build/check"
printf 'rank,m,n,k,pad,core_calls,fringe_calls,checked,errors,inputs_unchanged,route_verified\n' > "$out/routes.csv"
for rank in 48 49; do
    mkdir "$build/$rank"
    python3 -B "$common/rank48/generate.py" "$rank" "$build/$rank/schedule.h" >> "$out/certificates.txt"
    "$cc" -O3 -mcpu=native -std=c11 -fPIC -shared -I"$build/$rank" "$common/rank48/adapter.c" -ldl -o "$build/form$rank.so"
    sha256sum "$build/form$rank.so" >> "$out/build-sha256.txt"
    while IFS=, read -r name m n k pad; do
        [[ $name == family ]] && continue
        taskset -c "$cpu" "$build/check" "$build/form$rank.so" "$rank" "$m" "$n" "$k" "$pad" >> "$out/routes.csv"
    done < "$here/panel.csv"
done
status=0
taskset -c "$cpu" python3 -B "$here/study.py" "$build/release.worker" "$here/panel.csv" "$out" "$vendor/openblas.so" \
    --phase rank48-arm-confirm --rounds 3 --stress-all --vendor "armpl=$vendor/armpl.so" \
    --candidate "form48=direct:$build/form48.so" --candidate "form49=direct:$build/form49.so" \
    --candidate "release-openblas=linked-public:$build/release.worker" \
    --candidate "fastmmw-768=direct:$vendor/fastmmw-768.so" --candidate "fastmmw-1536=direct:$vendor/fastmmw-1536.so" || status=$?
printf '%d\n' "$status" > "$out/study-status.txt"
python3 -B "$here/score.py" "$out"
