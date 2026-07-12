# Benchmarking Guide

This is the detailed reference for measuring COB. For a quick introduction,
start with the [README](../README.md).

## Basic benchmark

```sh
make bench
./build/cob_gemm_bench 128 256 512
./build/cob_gemm_bench 832x960x896
```

Arguments can be square sizes or rectangular `MxNxK` shapes. On macOS, the
benchmark also builds an Apple Accelerate comparison when the framework is
available. It pins common BLAS thread environment variables to `1` and raises
its own thread QoS to reduce scheduler noise.

Common controls:

| Variable | Purpose |
| --- | --- |
| `COB_BENCH_REPEATS=N` | Change the number of timed samples. |
| `COB_BENCH_ITERS=N` | Run multiple GEMMs inside each timed sample. |
| `COB_BENCH_COOLDOWN_US=N` | Cool down after each repeat. |
| `COB_BENCH_CSV=1` | Print machine-readable rows. |
| `COB_BENCH_ROUTE=1` | Include the selected COB route. |
| `COB_BENCH_PACK_SETUP=1` | Include one-time packed-B setup cost. |
| `COB_BENCH_ONLY=...` | Isolate `one-shot`, `packed`, `packed-AB`, `pack-setup`, or `accelerate`. |
| `COB_BENCH_HOT_SECONDS=N` | Keep one isolated row hot for profiler attachment. |

The paired harness compares every output element and defaults to the same
`0.002` maximum absolute difference used by the correctness suite.
`COB_AB_MAX_ABS_DIFF` can override it when deliberately testing another
numeric contract. Sampled checksums are diagnostic, not the correctness gate.

## Grid sweeps

```sh
sh tools/bench_grid.sh 832 896 960
COB_GRID_M="64 96 128" COB_GRID_N="1024 2048" COB_GRID_K="512 1024" \
  sh tools/bench_grid.sh
COB_BENCH_ROUTE=1 sh tools/bench_grid.sh 832 896 960 \
  | python3 tools/bench_gap_report.py
COB_BENCH_ROUTE=1 sh tools/bench_grid.sh 832 896 960 \
  > /tmp/cob-grid.csv
python3 tools/bench_heatmap.py /tmp/cob-grid.csv \
  --output /tmp/cob-grid.png
```

Use `python3 tools/bench_gap_report.py --max-drop-percent 15` when a long sweep
shows large best-to-median drops. `tools/claim_audit.sh` uses the same sanity
filter before ranking gaps.

## Candidate versus baseline

Use the paired A/B harness for source changes. It defaults to the one-shot API.
Set `COB_AB_MODE=packed` for packed B or `COB_AB_MODE=packed-AB` for both
inputs prepacked.

```sh
COB_AB_REPEATS=61 COB_AB_MAX_REPEATS=101 \
  sh tools/paired_ab_bench.sh baseline/src/gemm.c src/gemm.c 512
COB_AB_MODE=packed-AB COB_AB_REPEATS=61 \
  sh tools/paired_ab_bench.sh baseline/src/gemm.c src/gemm.c 512
```

For thermally noisy shapes, set `COB_AB_COOLDOWN_US=N`. This makes confirmation
runs slower but helps keep false dispatch wins out of the timeline.

To compare COB's one-shot path against Apple Accelerate with paired statistics:

```sh
COB_AB_REPEATS=101 COB_AB_ITERS=8 COB_AB_COOLDOWN_US=20000 \
  sh tools/paired_accelerate_bench.sh src/gemm.c 512x1216x3072
```

Prefer this harness over comparing medians from separate COB and Accelerate
processes when a small candidate delta matters.

## External BLAS libraries

For an open-source CBLAS implementation, build the external target. For
example, with a local BLIS build:

```sh
make bench-cblas \
  CBLAS_NAME=blis_apple \
  CBLAS_HEADER=blis.h \
  CBLAS_CFLAGS=-isystem/path/to/blis/include/aaplmx \
  CBLAS_LDFLAGS=/path/to/blis/lib/aaplmx/libblis.a
```

For a library exporting the standard Fortran `sgemm_` symbol:

```sh
make bench-fortran-blas \
  FORTRAN_BLAS_NAME=blasfeo \
  FORTRAN_BLAS_LDFLAGS=/path/to/blasfeo/lib/libblasfeo.a
```

For MpGEMM, keep its checkout outside this repository:

```sh
git clone https://github.com/MpGEMM/mpgemm /private/tmp/mpgemm_latest
sh tools/mpgemm_calibration.sh
```

Set `COB_MPGEMM_RUN_MPGEMM=0` to skip rebuilding and running MpGEMM and only
refresh COB's matching route-aware CSV. The helper emits gap-only and all-row
FP32 comparisons. MpGEMM's FP16 and int8 results are different contracts.

## Hardware counters

Build `mperf-stat` from <https://github.com/tmcgilchrist/mperf>. On M5 systems,
force the local `as5` PMC database:

```sh
sudo env MPERF_KPEP_DB=as5 COB_COUNTER_ONLY=one-shot \
  sh tools/counter_probe.sh 64x4096x7168
sudo env MPERF_KPEP_DB=as5 COB_COUNTER_ONLY=packed \
  sh tools/counter_probe.sh 512x1280x1536
```

The helper isolates the requested benchmark row so Accelerate and other COB
paths do not pollute the counters.

## Xcode Instruments from the command line

If root-backed counters are unavailable, `xctrace` CPU Counters can expose
top-level pipeline ratios. These are CPU pipeline ratios, not direct AMX or SME
engine utilization counters.

```sh
env COB_BENCH_ONLY=one-shot COB_BENCH_REPEATS=1 COB_BENCH_ITERS=128 \
  COB_BENCH_ROUTE=1 \
  xcrun xctrace record --template 'CPU Counters' \
  --output /private/tmp/cob-512.trace \
  --launch -- "$(pwd)/build/cob_gemm_bench" 512x512x4096
xcrun xctrace export --input /private/tmp/cob-512.trace \
  --xpath '/trace-toc/run[@number="1"]/data/table[@schema="MetricTable"]' \
  --output /private/tmp/cob-512-metric.xml
python3 tools/xctrace_metric_summary.py /private/tmp/cob-512-metric.xml \
  --process cob_gemm_bench --thread "Main Thread" --min-duration-ns 750000
```

For Time Profiler, export the already-symbolized `time-profile` table:

```sh
env COB_BENCH_ONLY=one-shot COB_BENCH_REPEATS=1 COB_BENCH_ITERS=128 \
  COB_BENCH_ROUTE=1 \
  xcrun xctrace record --template 'Time Profiler' \
  --output /private/tmp/cob-512-time.trace \
  --launch -- "$(pwd)/build/cob_gemm_bench" 512x512x4096
xcrun xctrace export --input /private/tmp/cob-512-time.trace \
  --xpath '/trace-toc/run[@number="1"]/data/table[@schema="time-profile"]' \
  --output /private/tmp/cob-512-time-profile.xml
python3 tools/xctrace_time_sample_summary.py \
  /private/tmp/cob-512-time-profile.xml --group symbol --limit 20
```

As a simpler fallback, keep one process alive long enough for macOS `sample`:

```sh
COB_BENCH_ONLY=one-shot COB_BENCH_ROUTE=1 COB_BENCH_HOT_SECONDS=20 \
  build/cob_gemm_bench 512x1024x1536
```

## Claim audits

The full claim recipe, current baseline set, and publication rules live in
[claims.md](claims.md). A performance change is not accepted from a peak number
alone. Confirm the paired median, bootstrap interval, sign test, split-half
holdout, correctness, and clean build before changing a route.
