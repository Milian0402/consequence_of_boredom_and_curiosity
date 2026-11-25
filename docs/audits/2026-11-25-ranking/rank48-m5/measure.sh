#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
home=$PWD
python3 -B prepare.py
src=$home/bench/tools/world-rank
r=$src/rank48
result=${RANK_APPLE_RESULT:-results}
if [[ -e $result/accuracy.csv ]]; then echo "Use a fresh result directory." >&2; exit 2; fi
mkdir -p "$result" gen48 gen49
sysctl -n machdep.cpu.brand_string > "$result"/cpu.txt
sw_vers >> "$result"/cpu.txt
clang --version >> "$result"/cpu.txt
export VECLIB_MAXIMUM_THREADS=1 RANK_VERIFY_INPUTS=1 APPLE_RANK_THREAD_PROBE=$home/threads.dylib
clang -O2 -dynamiclib apple_thread_count.c -o threads.dylib
clang -O3 -mcpu=native -dynamiclib accelerate.c -framework Accelerate -o accelerate.dylib
export RANK_FINITE_LEAF=$home/accelerate.dylib
cmake -S release -B build -DCMAKE_BUILD_TYPE=Release -DCOB_BUILD_TESTS=OFF -DCOB_BUILD_BENCH=OFF > build.log
cmake --build build --parallel 2 --target cob_matmul >> build.log
clang -O3 -mcpu=native -std=c11 -Irelease/include -I"$src" -Dhybrid_sgemm_rowmajor=cob_sgemm_rowmajor -Drank_hybrid_next=cob_sgemm_rowmajor \
    worker-apple.c apple_thread_count.c accelerate.c build/libcob_matmul.a -framework Accelerate -lm -o worker
clang -O2 -std=c11 "$r/check.c" -o check
cp "$r/panel.csv" "$result"/panel.csv
printf 'rank,m,n,k,pad,core_calls,fringe_calls,checked,errors,inputs_unchanged,route_verified\n' > "$result"/routes.csv
for rank in 48 49; do
    python3 -B "$r/generate.py" "$rank" "gen$rank/schedule.h" >> "$result"/certificates.txt
    clang -O3 -mcpu=native -std=c11 -dynamiclib -I"gen$rank" "$r/adapter.c" -o "form$rank.dylib"
    while IFS=, read -r family m n k pad; do
        [[ $family != family ]] || continue
        ./check "$home/form$rank.dylib" "$rank" "$m" "$n" "$k" "$pad" >> "$result"/routes.csv
    done < "$r/panel.csv"
done
shasum -a 256 "$r/"{generate.py,adapter.c,panel.csv} "$r"/schedules/*.slp worker-apple.c "$src/study.py" build/libcob_matmul.a worker *.dylib > "$result"/build-hashes.txt
set +e
python3 -B "$r/study_runner.py" "$home/worker" "$home/$result/panel.csv" "$home/$result" "$home/accelerate.dylib" \
    --phase finite-ranks --rounds 3 --stress-all --candidate "cob-release=linked-public:$home/worker" \
    --candidate "form48=direct:$home/form48.dylib" --candidate "form49=direct:$home/form49.dylib"
study_status=$?
set -e
[[ $study_status == 0 || $study_status == 1 ]] || exit "$study_status"
python3 -B profile.py
RANK_APPLE_RESULT="$result" python3 -B profile-run.py
python3 -B audit.py "$result" > "$result/analysis.json"
