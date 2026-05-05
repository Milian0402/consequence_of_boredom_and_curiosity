# Optimization Attempt Timeline

Last updated: 2026-05-05

This tracks the matrix-multiplication speed work so far. The narrow comparison
scope is single-threaded FP32 row-major GEMM for `C = A * B` with `alpha = 1`
and `beta = 0`.

## Timeline

### 2026-05-04: AMX packed-B path

The first major speed path used Apple Silicon AMX with a public packed-`B` API,
native AMX `B` layout, faster aligned packing/stores, and reuse of packed `B`
panels across row blocks.

Result: this became the strongest exact-scope path in COB. On large square
cases, the packed-`B` path measured around 1.9-2.0 TF/s in later validation.

### 2026-05-04: one-shot direct-B AMX optimizations

The one-shot path was optimized for smaller square cases by avoiding some
scratch allocations and extending the direct-`B` AMX range. The latest pushed
commit in this series was `f335c66 Avoid direct-B AMX conflict at 512 stride`.

Result: validation after that commit passed with `make test`, `make bench`, a
CMake build, and `ctest`. COB improved substantially, but Apple Accelerate
could still win in some tested small cases, including around `n = 192`.

### 2026-05-04 to 2026-05-05: external library comparisons

The default square-size benchmark comparisons showed COB beating `blis_apple`
and OpenBLAS. BLASFEO, Rust `matrixmultiply`, Eigen, and `coral-aarch64` were
far below COB in the tested exact scope.

LIBXSMM's exact `beta = 0` path was also far below COB. Its `beta = 1` plus
zeroed-output workaround was close, but that is not the same exact API.

`tract-linalg` AMX one-shot and packed-`B` paths were below COB. Its
compute-only packed-both path was strong, but that comparison does not use the
same API shape.

### 2026-05-05: MpGEMM comparison

MpGEMM was found at <https://github.com/MpGEMM/mpgemm>, with paper
<https://arxiv.org/abs/2512.21473>. It is source-available, but no license file
was observed.

Result: in same-process benchmarks, MpGEMM beat COB in several one-shot FP32
single-thread square cases, including `n = 256`, `384`, `512`, `768`, and
`1024`.

### 2026-05-05: clean-room SME prototype

A clean-room SME prototype was correct and looked promising in compute-only
runs, reaching around 2.0 TF/s in some measurements. The one-shot version was
slower overall because packing dominated the runtime.

Result: the current follow-up is to test a packed-`B` SME path, where packing
can be amortized instead of paid on every call.

### 2026-05-05: guarded SME packed-B path

Commit `0b3f398 Add SME packed-B GEMM path` added a guarded clean-room Apple
SME2.1 `16x64` path for the public packed-`B` API. It reuses the existing
32-column packed-`B` layout and AMX `A` packing.

The path is runtime gated on SME2.1 and 16 FP32 lanes. The dispatch threshold
was tightened to `m/n/k >= 512` after `n = 384` proved noisy.

Result: `make test` passed with 24 shapes, including a new 512
packed-vs-direct check. `make bench` showed packed-`B` around 1837 GF/s at
`n = 512`, 1982 GF/s at `n = 768`, and 1992 GF/s at `n = 1024` in one run. A
CMake build and `ctest --test-dir build-cmake --output-on-failure` also passed.
This improves COB packed-`B`, but still does not close the one-shot gap to
MpGEMM.

### 2026-05-05: narrow SME route for one-shot 512

Commit `bc00b5c Route 512 one-shot GEMM through SME` added a narrow one-shot
optimization for the 512-wide direct-`B` conflict case. After `B` has already
been packed, that case reuses the guarded SME packed-`B` kernel.

The route is limited to `n == 512`, aligned AMX rows, and `k >= 512`. Broader
SME direct-`B` experiments were backed out because they were noisy and
shape-sensitive.

Result: `make test` passed with 24 shapes. In repeated same-session `n = 512`
benchmarks, one-shot improved from roughly 1.58-1.61 TF/s with SME disabled to
about 1.65-1.70 TF/s with the route enabled. A CMake build and
`ctest --test-dir build-cmake --output-on-failure` also passed. This improves
one-shot 512, but still does not complete the universal fastest claim.

### 2026-05-05: post-0182f31 ACL and SME direct-B attempts

ACL CMake SVE/SME builds crashed on Apple M5 Max with `SIGILL`, traced to a
non-streaming SVE instruction in `CpuGemmAssemblyDispatch::configure`. An ACL
NEON-only build ran, but measured only about 120-126 GF/s single-thread, far
below COB.

An SME packed-`B` loop-order experiment was tested and not kept because results
were mixed and noisy.

A conservative local SME direct-`B` one-shot route was added for medium
contiguous square sizes `832-1216`, excluding `1024`. Paired runs improved the
one-shot benchmark results. `1280` was excluded as unstable.

Final pushed route commit `faf4401` uses blocked `A` packing for the medium
direct-`B` SME path. Validation passed with `make test`, a CMake build, and
`ctest --test-dir build-cmake --output-on-failure`. A lower-size gate experiment
extending the route to `768` was rejected after integrated benchmarks regressed
at `768`, so the final gate remains `832-1216`, excluding `1024`.

### 2026-05-05: rectangular SME direct-B route

Commit `3f4ea03` broadened the direct-`B` SME route from square-only to selected
medium contiguous rectangular cases. It also added an Apple-arm64 rectangular
direct-vs-packed test.

Result: `make test` passed with 26 shapes, plus a CMake build and
`ctest --test-dir build-cmake --output-on-failure`. A temporary rectangular
benchmark showed direct around 1.94-2.01 TF/s on the tested rectangular cases.

A temporary two-step SME inner-loop unroll in
`/private/tmp/cob_sme_unroll2_exp` passed tests, but was rejected because it
slowed important routed sizes and hurt packed-`B` stability.

### 2026-05-05: post-841ee51 rejected experiments

Widening the post-pack SME route beyond `n == 512` was tested in
`/private/tmp/cob_postpack_sme_exp` and rejected because it regressed
`n = 768` and `n = 1536`, and did not beat the existing direct-`B` route for
gated sizes.

Changing `COB_SGEMM_AMX_MC` from `384` to `512` was tested in
`/private/tmp/cob_mc512_exp` and rejected because it hurt packed-`B` at larger
sizes such as `1280` and `1408` without a sufficient one-shot gain.

A narrow `n = 384` SME direct-`B` gate was tested in
`/private/tmp/cob_sme384_exp`. It passed tests, but did not beat the current
integrated AMX one-shot path at `n = 384`, so it was rejected.

A strided-`B` AMX row-blocking experiment was tested in
`/private/tmp/cob_strided_b_block_exp`. It passed tests, but regressed small
one-shot sizes such as `64`, `128`, and `256`, so the existing per-32-row `A`
packing stayed.

### 2026-05-05: median benchmark output and admin follow-up

Commit `7ea85c5` added median-aware throughput output to the benchmark harness,
so benchmark rows now show median GF/s alongside the existing timing summary.

Removing the `n == 1024` exclusion from the SME direct-`B` medium gate was
tested and rejected. The change passed tests, but did not improve `1024` median
throughput and caused instability around the gated range, so it was reverted.

Commit `a27a03d` documented an admin/tooling fix after an edit-confirmation UI
appeared while patch-editing a `/private/tmp` worktree. `AGENTS.md` and
`.codex-reminder.md` now tell agents to treat that as client/tooling friction,
avoid that edit path, and continue in the repo checkout without asking.

### 2026-05-05: packed-B SME dispatch capped

Commit `c699b56` capped the public packed-`B` SME dispatch to `n <= 1216`.
Benchmarks showed the AMX packed-`B` path was faster and more stable for large
packed-`B` cases, especially `n = 1536`.

Result: `make test` passed. A selected benchmark after the change showed
packed-`B` median around 1.94-1.96 TF/s at `n = 1536`, versus the earlier broad
SME route around roughly 1.45-1.53 TF/s in the same session.

### 2026-05-05: one-shot SME direct-B 1280 experiment

Commit `d68430c` extended the medium direct-`B` SME route from `1216` to `1280`
and added an Apple-arm64 direct-vs-packed correctness case for `1280`.

Result: with the temporary `1280` gate, repeated `n = 1280` one-shot medians
were mostly about 1.84-1.89 TF/s. With the original `1216` cap, repeated
`n = 1280` one-shot medians were mostly about 1.65-1.74 TF/s with one severe
outlier. Validation passed with `make test` across 27 shapes, and
`git diff --check` passed.

### 2026-05-05: benchmark repeat controls and n=1024 SME rejection

A one-shot `n = 1024` experiment routed the case through the post-pack SME path
after `B` packing by extending the existing `n == 512` conflict check to also
cover `n == 1024`. It passed `make test` across 27 shapes, but did not
materially improve one-shot `n = 1024` versus the AMX path, so it was reverted.

Commit `598444e` made benchmark repeats configurable with
`COB_BENCH_REPEATS`, changed large-shape default repeats from 4 to 5 for a real
median, and documented the environment variable in `README.md`. Validation
included `make test`, `make all`, `git diff --check`, and
`env COB_BENCH_REPEATS=9 ./build/cob_gemm_bench 1024 1280`.

### 2026-05-05: packed-B medium-width dispatch narrowed

Commit `8143b9f` excluded `n = 832` and `n = 960` from the public packed-`B`
SME route, so those widths use AMX packed-`B` instead.

Result: a focused `COB_BENCH_REPEATS=9` benchmark after the change showed
packed-`B` medians about 2003-2014 GF/s at `n = 832` and about 2018 GF/s at
`n = 960`, better than the prior SME-dispatch baseline observed around
1969 GF/s at `832` and 1977-1988 GF/s at `960`. Validation passed with
`make test` across 28 shapes, and `git diff --check` passed.

### 2026-05-05: SME direct-B row block split

Two strided-`B` SME loop experiments were tested and rejected. A B-block-first
loop order passed tests, but regressed key one-shot sizes, especially `n = 896`
and `n = 1216`, so it was reverted. A pointer-increment rewrite inside the same
loop also passed tests, but regressed `n = 896`, so it was reverted.

Commit `73cb130` added a separate 256-row SME direct-`B` row block while
keeping public packed-`B` SME on the 384-row AMX block.

Result: a focused `COB_BENCH_REPEATS=9` benchmark improved several one-shot
routed sizes, including medians around 2013 GF/s at `n = 960`, 1962 GF/s at
`n = 1088`, 1922 GF/s at `n = 1152`, and 1926 GF/s at `n = 1216`. Validation
passed with `make test` across 28 shapes, and `git diff --check` passed.

### 2026-05-05: packed-B n=1088 dispatch narrowed

Commit `04f2a91` excluded `n = 1088` from the public packed-`B` SME route, so
that width uses AMX packed-`B` instead. It also added an Apple-arm64
direct-vs-packed correctness case for `1088`.

Result: a focused `COB_BENCH_REPEATS=9` benchmark after the change showed
`n = 1088` packed-`B` medians about 2027-2044 GF/s, better than the prior SME
route around roughly 1990-2010 GF/s in nearby runs. Validation passed with
`make test` across 29 shapes, and `git diff --check` passed.

### 2026-05-05: one-shot n=832 AMX direct-B extension

Commit `50bcead` extended the AMX direct-`B` max `N` from `768` to `832`, so
one-shot `n = 832` uses AMX direct-`B` instead of the SME medium route.

Result: a focused `COB_BENCH_REPEATS=15` benchmark showed `n = 832` one-shot
medians around 2014-2021 GF/s, versus the old cutoff control around roughly
1983-2003 GF/s. Validation passed with `make test` across 29 shapes, and
`git diff --check` passed.

### 2026-05-05: one-shot n=960 AMX direct-B extension

Commit `c99f5d5` extended the AMX direct-`B` max `N` from `832` to `960`, so
one-shot `n = 960` uses AMX direct-`B`.

Result: a focused `COB_BENCH_REPEATS=15` benchmark showed `n = 960` one-shot
medians about 2029-2036 GF/s, improving over the previous 832-cutoff control
range around roughly 1995-2006 GF/s. Accelerate still remained around
2058-2067 GF/s. Later split-gate work kept `n = 896` on the SME medium route
while preserving the `n = 960` AMX win. Validation passed with `make test`
across 29 shapes, and `git diff --check` passed.

### 2026-05-05: one-shot n=896/960 AMX direct-B split gate

Commit `7fec4b5` split the AMX direct-`B` medium gate so AMX handles
`n <= 832` plus an explicit extra `n == 960`, letting `n = 896` fall back to
the SME medium route.

Result: focused `COB_BENCH_REPEATS=15` validation showed one-shot medians about
1976 GF/s at `n = 896` and about 2029 GF/s at `n = 960`. Validation passed
with `make test` across 29 shapes, and `git diff --check` passed.

### 2026-05-05: latest post-1088 rejected experiments

After commits `04f2a91` and `822123c`, several follow-up experiments were
tested and rejected. Extending the SME direct-`B` one-shot gate to `n = 1408`
after the 256-row direct-`B` block change passed tests, but did not improve
one-shot `1408`, so it was reverted. Lowering the AMX large-block threshold
from `n >= 1152` to `n >= 1024` in both one-shot and public packed-`B` AMX
paths passed tests, but did not help one-shot `1024` and created noisy
regressions around `1152` and `1216`, so it was reverted.

Forcing public packed-`B` AMX at `n = 1152` and `n = 1216` by excluding those
widths from SME packed-`B` also passed tests, but was mixed and not clearly
better, so it was reverted. A stack-backed `A` scratch experiment for small AMX
direct-`B` one-shot, including an overrideable macro cutoff and cutoff `224`,
passed tests, but a controlled heap-only comparison was faster for `96-224`, so
it was reverted.

Excluding `n = 1088` from the SME direct-`B` one-shot route was also tested and
rejected. It made one-shot `1088` fall back to the slower packed/large path and
benchmarked clearly worse, with medians around 1841-1857 GF/s versus the
committed SME route around 1966-1970 GF/s, so the experiment was reverted.

Workflow note: patch-editing detached `/private/tmp` worktrees can still trigger
client edit-confirmation friction despite standing repo permission. Future
experiments should prefer the main checkout, or use non-editing toggles when a
detached worktree is only needed for measurement.

Validation after the reverts included `make test`; the tree returned clean.

### 2026-05-05: rectangular benchmark arguments

Commit `4074513` added rectangular `MxNxK` benchmark arguments, such as
`832x960x896`, while preserving the previous square `N` arguments.

Result: validation passed with `make test` and a `COB_BENCH_REPEATS=1`
benchmark smoke run.

### 2026-05-05: one-shot n=1088 AMX direct-B gate

Commit `2306377` routes one-shot `n = 1088` through AMX direct-`B` by adding a
second extra direct-`B` gate. It leaves `n = 1024` on the previous packed path,
because testing AMX direct-`B` at `n = 1024` did not improve it.

Result: the same-session old SME one-shot `1088` baseline was about
1954-1963 GF/s median. The new AMX direct-`B` gate measured about
2025-2033 GF/s median in focused 15-repeat runs, with a later 7-repeat sanity
run around 2009-2011 GF/s median.

### 2026-05-05: one-shot n=1152/1216 AMX direct-B gate

Commit `f45bb5e` routes one-shot `n = 1152` and `n = 1216` through AMX
direct-`B` as large extra direct-`B` sizes. It leaves `n = 1280` on the
previous path because testing there was mixed.

Result: focused 15-repeat square validation showed the old baseline around
1911-1916 GF/s median at `n = 1152` and 1925-1927 GF/s at `n = 1216`. The new
direct-`B` route measured about 1985-1991 GF/s at `n = 1152` and
2027-2032 GF/s at `n = 1216` in final validation.

Validation passed with `make test` across 31 shapes after adding `1152` and
`1216` direct-vs-packed coverage. Rectangular smoke with the `MxNxK` benchmark
showed acceptable behavior for `832x1152x896`, `896x1152x1152`,
`1280x1152x960`, `1152x1216x896`, and `1216x1216x1024`.

### 2026-05-05: packed-B setup benchmark row

Commit `0a7de51` added an opt-in `COB_BENCH_PACK_SETUP=1` benchmark row for
the public packed-`B` setup cost and documented the environment variable in
`README.md`.

Result: validation passed with `make test`, a benchmark build, and a smoke run
using `COB_BENCH_REPEATS=3 COB_BENCH_PACK_SETUP=1 ./build/cob_gemm_bench 64
832x960x896`.

### 2026-05-05: row-wise public B-pack order rejected

A row-wise public `B`-pack order experiment changed source traversal during
setup. It passed tests, but setup throughput collapsed compared with the
existing panel-wise packing order.

Result: the baseline panel-wise setup measured roughly 90-114 GB/s across
`512..1536` in a 15-repeat pack-setup run. The row-wise traversal measured only
about 16-67 GB/s depending on size, so it was reverted. The existing panel-wise
packing order stays.

### 2026-05-05: small-k scalar A-pack rejected

Forcing scalar `A`-panel packing for `k <= 256` passed `make test`, but it was
a hard regression in the focused small-size sweep.

Result: a 15-repeat sweep dropped COB one-shot and packed-`B` medians for
`96..256` to roughly 557, 587, 813, 811, 975, and 934 GF/s. That was far below
the prior AMX-pack small-size range of about 1300-1750 GF/s, so the experiment
was reverted.

### 2026-05-05: public packed-B SME n=1280 rejected

Re-enabling public packed-`B` SME dispatch through `n = 1280` passed tests, but
the benchmark results were unstable and mixed.

Result: a focused 15-repeat run had one good `n = 1280` packed-`B` median
around 1978 GF/s, while a duplicate same-run `n = 1280` packed-`B` median
collapsed around 1536 GF/s. `n = 1408` remained on AMX packed-`B` around
1950 GF/s. The change was reverted, keeping the public packed-`B` SME cap at
`n <= 1216`.

### 2026-05-05: 32x64 AMX tile mapping rejected

After commit `7523ab7`, a 32x64 AMX tile experiment tried to reuse one
packed-`A` load across two adjacent 32-column `B` panels. The mapping used
`z` rows `0..7` with paired `X` loads.

Result: the experiment did not pass correctness. `make test` failed on multiple
shapes, including `32x64x31`, `64x64x64`, `96x64x128`, `64x96x128`,
`128x128x129`, and the large direct-vs-packed checks for `1152`, `1216`, and
`1280`, with large max diffs. The attempted 32x64 mapping over AMX `Z` storage
is invalid. The patch was reverted, and `make test` then passed 31 shapes.

### 2026-05-05: two-panel public B-pack setup rejected

After commit `ad43943`, a public `B`-pack setup experiment packed two adjacent
32-column `B` panels in one pass over each source row. The intent was to improve
source locality without creating the full write-stream explosion from the
earlier row-wise setup order.

Result: the experiment passed `make test`, but pack setup throughput regressed
versus the existing panel-wise packing. In a 15-repeat
`COB_BENCH_PACK_SETUP=1` run over `512..1536`, setup medians were roughly 106,
94, 96, 92, 90, 90, 83, and 68 GB/s. That was below the earlier panel-wise
baseline of roughly 90-114 GB/s, and especially worse at larger sizes. The
experiment was reverted, so the existing panel-wise packing remains.

### 2026-05-05: skinny one-shot GEMM dispatch

Commit `32e0221` improved one-shot dispatch for LLM-style skinny shapes after
comparing COB against MpGEMM-style workloads. The public MpGEMM repository
currently has no license file, but its README calls the project open-source and
targets skinny DeepSeek/LLaMA workloads.

COB's public packed-`B` path was already strong on examples such as
`64x2112x7168`, `64x24576x1536`, and `64x32768x512`. The weak point was
one-shot dispatch, where full `B` packing dominated runtime. The commit added
an AMX direct-`B` skinny fallback for `m <= 128` and `n >= 1024`, plus a
256-column chunked packed-`B` one-shot path for `m = 96..128`. It also added
direct-vs-packed correctness coverage for `64x2112x512`, `96x4096x1024`, and
`128x2048x2048`.

Full chunking was rejected for `m = 64` because it hurt that row count, so
`m = 64` uses direct-`B`. Skinny SME direct-`B` was worse for larger `K`; only
a narrow `k = 512` fallback remains after chunk allocation. The new path
improves COB one-shot skinny performance, but still does not universally beat
Accelerate on skinny shapes.

Result: validation passed with `make test` across 34 shapes, and
`git diff --check` passed. Correction after the latest isolated validation:
the earlier `m = 64` direct-`B` medians were noisy and overstated. Isolated
15-repeat runs showed `m = 64` skinny direct-`B` is only a modest improvement
over the old full-pack path, with examples around 365-373 GF/s median for
`64x2112x7168`, about 384 GF/s for `64x4096x7168`, about 452 GF/s for
`64x7168x2048`, and about 295 GF/s for `64x32768x512`. It remains below
Accelerate on those one-shot cases, but avoids some full-pack overhead. The
`m = 96/128` chunked path was later improved by `40454ae`; see the follow-up
note below.

### 2026-05-05: skinny chunk packer follow-up

Commit `40454ae` tuned the skinny one-shot chunk packer added after the earlier
dispatch change. Packing panels within each chunk was faster than the initial
row-wise chunk packing order, so the chunked skinny path now keeps the
panel-wise setup shape. The chunk width is also conditional: `m = 128` with
`k >= 1024` uses 512-column chunks, while the other chunked skinny shapes stay
at 256 columns.

Result: validation passed with `make test` across 34 shapes, and
`git diff --check` passed. Focused skinny samples showed about 733 and
839 GF/s medians for `96x4096x1024` and `96x8192x512`, plus about 944, 1007,
and 1008 GF/s medians for `128x4096x1024`, `128x2048x2048`, and
`128x8192x512`. These are still below the packed-`B` path, but closer to
Accelerate than the first chunked one-shot version.

### 2026-05-05: rejected skinny AMX chunk experiments

Two skinny AMX chunk-pack experiments were tried after the `40454ae` follow-up
and reverted. A temporary compile-time chunk-width sweep tested 384, 512, 768,
and 1024 columns against the default 96=>256, 128/k>=1024=>512, otherwise 256
policy. Broad variants were mostly worse; the isolated `M128_K1024=768` case
was mixed and noisy rather than a reliable win, and `M96=1024` was unstable.

The compute loop in `cob_sgemm_rowmajor_amx_skinny_pack_b_chunks` was also
temporarily changed from chunk->panel->ap to chunk->ap->panel. It passed
`make test`, but did not improve enough and regressed large skinny cases:
`128x32768x512` measured around 475 GF/s median versus about 530 GF/s in the
prior clean run. No code from either experiment was kept.

### 2026-05-05: rejected skinny SME packed-B one-shot route

A temporary `COB_USE_APPLE_SME` path packed `A` once, packed `B` in 256-column
chunks, and called the existing `16x64` SME packed-`B` kernel for skinny
one-shot shapes. The experiment passed `make test` while present, but was
reverted because it was not a reliable win.

The wide `m = 64..128` gate was rejected because `m = 64` became much slower
than the AMX direct-`B` fallback. Same-session medians included the SME route at
about 361 GF/s versus AMX/no-SME about 707 GF/s for `64x2112x7168`, about
272 versus 477 GF/s for `64x24576x1536`, and about 248 versus 380 GF/s for
`64x32768x512`. Narrowing to `m = 128, k >= 1024` was still inconclusive or
regressive: duplicate `128x4096x1024` medians with the SME route were about
801 and 855 GF/s versus AMX/no-SME about 889 and 892 GF/s, while
`128x2048x2048` was mixed.

A temporary `build-nosme` directory was used for A/B runs against an SME-disabled
build. `.gitignore` now ignores `build-*` directories so those local comparison
builds stay out of status.

### 2026-05-05: m=64 k=512 large-N SME direct-B route

The existing skinny SME direct-`B` fallback was narrowed into a useful large-`N`
case: it now allows `m = 64` only when `n >= 32768` and `k = 512`. This catches
`64x32768x512` after the AMX chunked path rejects `m = 64`.

Wider gates were rejected. Allowing the route from `n >= 1024` hurt smaller
`m = 64, k = 512` cases such as `64x2112x512` and `64x4096x512`. A threshold
at `n >= 8192` was still mixed and noisy, with weak results around `24576`.
The final narrow route improved duplicate 15-repeat `64x32768x512` one-shot
medians from the AMX/no-SME control around 252-256 GF/s to about 316-317 GF/s.
Accelerate was still faster on that case at roughly 410-415 GF/s.

Validation passed with `make test` across 35 shapes after adding
`64x32768x512` direct-vs-packed correctness coverage.

### 2026-05-05: rejected small-square SME direct-B route

A temporary dispatch experiment lowered the SME direct-`B` medium route to also
try square `192` and `320`, and moved that route ahead of AMX direct-`B` so the
small cases could reach it.

Result: the experiment passed `make test`, but was rejected. `192` regressed
badly in one duplicate 15-repeat run, with medians around 1387-1619 GF/s versus
the AMX direct-`B` baseline around 1696 GF/s. `320` remained below Accelerate,
and moving the medium SME route earlier also regressed routed medium sizes such
as `1088`, `1152`, and `1216`. The code change was reverted.

### 2026-05-05: rejected AMX loop-control experiments

Two more low-risk control-flow experiments were tested and rejected, then
re-evaluated with the paired A/B harness. The AMX fixed-width strided-`B`
panel-loop unroll in `/private/tmp/cob_amx_unroll_exp` passed `make test`, but
paired B/A results for `960`, `1088`, `1152`, and `1216` showed no convincing
win: medians were `0.9954x`, `0.9992x`, `0.9993x`, and `0.9928x`, with
negative trends at `960` and `1216`.

The skinny AMX single-allocation scratch experiment in
`/private/tmp/cob_skinny_single_alloc_exp` also passed `make test`, but paired
skinny-case results were mixed and noisy: medians ranged from `0.9828x` to
`1.0070x`, with one negative trend at `128x8192x512`. No code from either
experiment was kept.

### 2026-05-05: paired benchmark diagnostics tightened

Commit `d36a41c` tightened the paired A/B benchmark harness. It added paired
speedup CV output, lets auto-repeat stop once the paired ratio is stable, prints
a two-tailed sign-test p-value, and exits nonzero on checksum divergence. The
default no-argument shape set was expanded, side-specific compiler flags
`COB_AB_A_FLAGS` and `COB_AB_B_FLAGS` were added, and the design rules were
updated.

The commit also added neutral `COB_SGEMM_SME_DIRECT_MAX_N` compile-time routing
bound for route A/B tests without changing default dispatch.

Result: validation before commit passed with `make all`, `make test` across
37 shapes, `sh -n tools/paired_ab_bench.sh`, `git diff --check`, a paired
self-smoke at `128`, and a side-flag smoke forcing
`COB_SGEMM_AMX_STRIDED_B_MAX_N=128` at `192`.

### 2026-05-05: rejected m=64 skinny KC probe

A compile-flag probe tested `COB_SGEMM_SKINNY_SME_KC=768` against the current
default on `64x2112x7168`, `64x4096x7168`, `64x8192x1024`, and
`64x7168x2048`.

Result: the probe was mixed and noisy. Paired results were `0.9969x` median,
mean-log `1.0046x`, CI `[0.9962,1.0131]` for `64x2112x7168`; `0.9913x`
median, mean-log `0.9890x`, CI `[0.9795,0.9986]` for `64x4096x7168`;
`1.0058x` median, CI `[0.9960,1.0083]` for `64x8192x1024`; and `1.0017x`
median, CI `[0.9952,1.0085]` for `64x7168x2048`. The probe was rejected, with
no source behavior change.

### 2026-05-05: rejected skinny direct-B bypass

The AMX skinny chunked path was temporarily disabled for `m = 96/128` below
`n = 32768`, forcing those shapes onto the AMX direct-`B` path instead. This
tested whether avoiding one-shot chunk packing would help when `B` packing
dominates.

Result: the bypass helped one huge `96x24576x1536` sample, improving the
one-shot median from roughly 380 GF/s to about 472 GF/s, but it badly regressed
common skinny shapes such as `96x4096x1024`, `96x8192x512`, `128x4096x1024`,
`128x2048x2048`, and `128x8192x512`. Large `k = 512` cases still lost to
Accelerate. The experiment was reverted.

### 2026-05-05: K-blocked m=64 SME skinny route

MpGEMM's live FP32 row-SGEMM benchmark was rebuilt from the current checkout
and remains the stronger open-source one-shot skinny blocker. The latest local
baseline measured about 1285 GF/s on `64x2112x7168`, 1071 GF/s on
`64x4096x7168`, 833 GF/s on `64x32768x512`, 1045 GF/s on
`64x24576x1536`, 1099 GF/s on `64x7168x2048`, and 1056 GF/s on
`64x7168x16384`.

COB added a clean-room K-blocked SME direct-`B` route for selected `m = 64`
skinny cases. The route uses `KC = 512`, packs only the current `A` K block,
streams `B`, and accumulates later K blocks into `C`. The dispatch is narrow:
`n = 1024..4096, k >= 7168`, plus the existing long-`N` `k = 512` case.
Wider routing was rejected because it hurt shapes such as `64x24576x1536` and
`64x7168x2048`.

Result: validation passed with `make test` across 37 shapes after adding
`64x2112x7168` and `64x4096x7168` direct-vs-packed coverage. After removing
the failed 64x16 experiment below, the latest clean working-tree `make test`
also passed all 37 shapes.

Latest focused 15-repeat skinny medians:

| Shape | COB one-shot | Accelerate | COB packed-B |
| --- | ---: | ---: | ---: |
| `64x2112x7168` | 1118.16 GF/s | 1131.87 GF/s | 1686.48 GF/s |
| `64x4096x7168` | 785.56 GF/s | 660.13 GF/s | 1702.81 GF/s |
| `64x32768x512` | 364.54 GF/s | 433.14 GF/s | 1366.95 GF/s |
| `64x24576x1536` | 462.60 GF/s | 654.19 GF/s | 1602.07 GF/s |
| `64x7168x2048` | 480.82 GF/s | 720.77 GF/s | 1712.90 GF/s |
| `64x7168x16384` | 508.43 GF/s | 696.10 GF/s | 1688.46 GF/s |
| `64x2112x512` | 1977.31 GF/s | 1949.46 GF/s | n/a |
| `64x4096x512` | 1322.34 GF/s | 1225.73 GF/s | n/a |

### 2026-05-05: rejected 64x16 B-reuse SME experiment

A 64x16 B-reuse SME experiment was tried after the K-blocked `m = 64` route
and removed. It regressed large-`K`, moderate-`N` skinny cases, with one run
around 833 GF/s on `64x2112x7168` and 672 GF/s on `64x4096x7168`. The apparent
long-`N`, `k = 512` gain did not repeat, later measuring only about
314-316 GF/s, so no code from the experiment was kept.

### 2026-05-05: Sapphire package checked

An online search found the `sapphire-compute` PyPI package, which advertises
1.56 TF/s SGEMM on Apple Silicon. The wheel was installed into a temporary
environment and inspected locally. Its native `libsapphire.dylib` links
Accelerate and imports `cblas_sgemm`, and the Python SGEMM wrapper documents
that it uses Apple's Accelerate framework.

Result: it is not a separate open-source single-thread SGEMM blocker for COB's
scope. With `VECLIB_MAXIMUM_THREADS=1`, local `native.sgemm` medians were about
604 GF/s on `64x2112x7168`, 395 GF/s on `64x4096x7168`, 230 GF/s on
`64x32768x512`, 643 GF/s on `64x24576x1536`, 401 GF/s on `64x7168x2048`, and
684 GF/s on `64x7168x16384`.

### 2026-05-05: m=64 SME B-pack reuse route

COB added a clean-room SME fused B-pack/reuse route for
`m = 64, n >= 32768, k = 512`, then generalized it into one `m = 64` SME
B-reuse helper with two narrow gates: `n >= 32768 && k == 512`, and
`n == 4096 && k >= 7168`. It packs all of `A` once, streams `B` and packs a
64-column panel while computing the first 16 rows, then reuses that packed `B`
for the other three 16-row panels. This was inspired by a high-level read of
MpGEMM's row-kernel behavior, but no assembly was copied.

The large-`K` gate is intentionally limited to `n == 4096`; `64x2112x7168`
stays on the existing K-blocked direct SME route because B-pack reuse regressed
that shape.

Result: `make test` passed all 37 shapes, and `git diff --check` passed. After
rebuilding `build/cob_gemm_bench`, a final focused repeat-25 benchmark measured
`64x2112x7168` at 1124.00 GF/s COB one-shot, 1705.78 GF/s packed-`B`, and
1139.19 GF/s Accelerate; `64x4096x7168` at 941.88, 1714.46, and 694.27 GF/s;
`64x4096x8192` at 943.74, 1629.35, and 695.88 GF/s; and `64x32768x512` at
918.91, 1421.23, and 435.15 GF/s.

Follow-up: the B-reuse helper was extended with a third narrow-ish wide gate,
`m = 64, n >= 7168, k >= 1024`, while leaving `64x2112x7168` on the existing
direct K-blocked SME route because B-reuse still regressed that shape. After
that gate, `make test` passed all 37 shapes and `git diff --check` passed. A
focused repeat-25 rebuild measured COB one-shot / packed-`B` / Accelerate at
`64x24576x1536` 924.93 / 1605.80 / 660.99 GF/s, `64x7168x2048` 980.71 /
1739.86 / 721.60 GF/s, `64x7168x16384` 945.91 / 1716.61 / 694.46 GF/s, and
`64x8192x1024` 985.08 / 1701.65 / 618.87 GF/s; the same run had
`64x2112x7168` at 1134.52 GF/s and `64x4096x7168` at 957.96 GF/s. This beats
Accelerate across those wide one-shot cases and narrows the MpGEMM gap, but
still does not universally overtake MpGEMM.

Tuning follow-up, not a new route: after the wide gate, the reuse K chunk was
tuned separately so `m = 64, n >= 7168, k >= 1536` now uses `KC = 768`; the
`n = 4096` large-`K` gate and `k = 1024` wide cases stay at `KC = 512`.
Validation again passed with all 37 `make test` shapes and `git diff --check`.
After rebuild, repeat-25 measured COB one-shot / packed-`B` / Accelerate at
`64x24576x1536` 932.25 / 1592.04 / 668.03 GF/s, `64x7168x2048` 971.08 /
1632.54 / 695.69 GF/s, `64x7168x16384` 951.84 / 1699.92 / 695.85 GF/s,
`64x8192x1024` 1011.06 / 1698.96 / 697.23 GF/s, `64x4096x7168` 959.43 /
1708.23 / 688.80 GF/s, and `64x32768x512` 906.11 / 1205.10 / 403.28 GF/s.

This still does not close the MpGEMM gap on `64x2112x7168` or
`64x4096x7168`, but it narrows `64x4096x7168` and beats the earlier local
MpGEMM `64x32768x512` baseline of about 833 GF/s.

Rejected AMX follow-up: from main commit `5744026`, a temporary experiment in
`/private/tmp/cob_amx_m64_chunk_exp` changed
`cob_sgemm_rowmajor_amx_skinny_pack_b_chunks` to allow
`m == 64, n >= 7168, k >= 1024` with `skinny_nc = 256`. The experiment passed
`/private/tmp/cob_amx_m64_chunk_exp/build/cob_gemm_test` across all 37 shapes,
but repeat-25 one-shot benchmarks were much worse than the existing SME
B-reuse route: `64x24576x1536` measured 271.24 GF/s,
`64x7168x2048` 506.07 GF/s, `64x7168x16384` 238.27 GF/s, and
`64x8192x1024` 474.27 GF/s, versus roughly 930-1010 GF/s for SME B-reuse on
those cases. The better-looking `64x4096x7168` at 946.39 GF/s and
`64x32768x512` at 924.84 GF/s were not useful enough to justify broad AMX
routing, so the AMX chunk route was rejected.

### 2026-05-05: exact 768 one-shot SME direct route

Commit `931daf0` (`Route square 768 through SME direct`) added one narrow
one-shot dispatch exception: exact `768x768x768` now bypasses the AMX
strided-`B` route and uses the existing SME medium direct-`B` path.

Result: full `make test` passed. Focused repeat-25 single-shape runs measured
one-shot medians around 1956.7 and 1961.0 GF/s, versus about 1915.4 GF/s in an
unchanged baseline worktree. Packed-`B` for the same shape stayed around
1969-1974 GF/s.

Rejected follow-ups after `a80d7d4`: Arm's BSD-licensed April 2026 SME2
learning-path sample was built and timed with a throwaway driver, but it was
educational rather than competitive: about 1552 GF/s median at `512`, 1713 at
`1024`, and about 500 on `m = 64` wide shapes. An AMX-only build with
`COB_DISABLE_APPLE_SME=1` confirmed the current SME dispatch is needed for
`m = 64` one-shot and is not a better global square route. Revisited
one-shot `1024` SME-from-packed routing passed tests but was equal or worse
than the current path. A low-threshold SME-before-AMX experiment hurt `192` and
`512` and was mixed at `256` and `384`, so only exact `768` was isolated and
kept. `m = 64` reuse `NC = 256` and `NC = 1024` probes were much slower, and a
`K = 1024` chunk-size probe stayed noisy; none were committed.

Commit `1f5bb7e` (`Route square 384 through SME direct`) added the matching
narrow exception for exact `384x384x384`: it now bypasses AMX strided-`B` and
uses the existing SME medium direct path. `make test` passed. Focused repeat-31
A/B measured main/baseline one-shot around 1696-1705 GF/s median, while the
direct-SME route measured around 1762-1766 GF/s median; the final focused rerun
after the patch measured about 1765 GF/s median at `384` and kept `768` around
1957 GF/s median.

Rejected follow-ups after `1f5bb7e`: a one-shot `1024` direct-SME exception
passed tests but measured around 1750 GF/s median, worse than current
one-shot/Accelerate in that run. A B64 one-shot SME layout for exact `512` and
`1024` passed tests but was not useful: `512` one-shot dropped to about
1583 GF/s median and `1024` was about 1760 GF/s median. Packed-`B` SME A-block
retests with 512-row and 256-row blocks both passed tests but reduced packed-`B`
medians at `768` and `1024` versus the current 384-row block. None were
committed.

Additional post-`1c1e827` rejected probes: exact `256` direct-SME routing was
A/B tested and rejected after the repeat-31 broad low-threshold worktree route
measured about 1491 GF/s median versus current AMX around 1633 GF/s. Medium
exact `1088` direct-SME also failed to hold up after the `384`/`768` wins:
repeat-31 current one-shot was around 1808-1868 GF/s median versus direct-SME
around 1687-1784 GF/s. For `m = 64`, `COB_SGEMM_SKINNY_SME_KC=256` was below
current, with repeat-25 medians around 914 GF/s at `64x8192x1024`, 870 at
`64x4096x7168`, and 775 at `64x32768x512`. Row-major full-panel `B` packing
traversal preserved the packed layout and passed tests, but pack setup fell to
about 19-20 GB/s and one-shot `512`/`1024` collapsed. A fixed NEON 128-byte
copy for `B` panel packing passed tests but did not beat current `memcpy`, whose
main-branch pack setup bandwidth was higher under the same harness. Finally,
the B-panel-first loop order for the SME packed-`B` kernel passed tests but
reduced packed-`B` medians at `768`, `1024`, and `1280`. None were committed.

### 2026-05-05: KleidiAI FP32 SME2 scan

ARM-software/kleidiai was cloned to `/private/tmp/kleidiai_latest` at commit
`7a7da26`. The checkout is Apache-2.0 and reports version `v1.24.0`. It
contains relevant FP32 SME/SME2 matmul microkernels, not just quantized
kernels.

The benchmark was built with `KLEIDIAI_BUILD_BENCHMARK=ON` and
`KLEIDIAI_INTERNAL_EXTRA_ARCH=+sme`. The stock benchmark binary exited before
printing results on Apple, so a temporary direct driver was written at
`/private/tmp/kleidiai_fp32_driver.c`.

Direct elastic FP32 SME2 path median results:

| Shape | Compute-only | One-shot |
| --- | ---: | ---: |
| `384^3` | 1986.77 GF/s | 1715.85 GF/s |
| `768^3` | 2063.71 GF/s | 1911.33 GF/s |
| `1024^3` | 2078.88 GF/s | 1791.06 GF/s |
| `64x4096x7168` | 1341.22 GF/s | 685.78 GF/s |
| `64x8192x1024` | 1317.47 GF/s | 630.13 GF/s |
| `64x32768x512` | 1005.38 GF/s | 535.80 GF/s |

Conclusion: KleidiAI is not a faster one-shot replacement for this repo. Its
compute-only path is an interesting upper-bound/reference, but it requires both
LHS and RHS prepacking, so it is not directly comparable to COB's public
one-shot SGEMM path.

### 2026-05-05: skinny AP-first loop order rejected

An AP-first skinny AMX chunk-loop experiment in
`/private/tmp/cob_skinny_apfirst_exp` swapped the packed-`B` chunk loop from
panel-first to `A`-panel-first. It passed `make all` and `make test`, but the
paired A/B signal was mixed and noisy.

Result: repeat-31, 8-iteration validation showed `96x4096x1024` median
`0.9992x` CI `[0.9909,1.0209]`, `96x8192x512` `1.0012x`
`[0.9934,1.0092]`, `128x4096x1024` `1.0038x` with mean-log `0.9928x`
`[0.9691,1.0101]`, `128x2048x2048` `1.0067x` `[1.0017,1.0126]`, and
`128x8192x512` `1.0077x` `[0.9997,1.0145]`. Smaller-N validation was also
noisy: `96x1024x512` `0.9981x` `[0.9926,1.0087]`, `96x2048x512` `0.9981x`
`[0.9753,1.0367]`, `128x1024x512` `1.0096x` `[0.9993,1.0574]`, and
`128x2048x512` `1.0174x` `[0.9730,1.1025]`. There was no broad
no-regression signal, so the kernel change was not committed.

### 2026-05-05: paired A/B repeat extension

Commit `edfc327` added optional CV-based automatic repeat extension to the
paired A/B benchmark harness. The new knobs are `COB_AB_MAX_REPEATS`,
`COB_AB_REPEAT_BATCH`, and `COB_AB_CV_TARGET`.

Result: default behavior remains fixed-repeat. The harness only extends beyond
`COB_AB_REPEATS` when `COB_AB_MAX_REPEATS` is set above it.

### 2026-05-05: benchmark-boundary infrastructure

Commit `f024172 Add CSV benchmark output` added `COB_BENCH_CSV=1` to
`bench/bench_gemm.c`, emitting
`kind,implementation,m,n,k,best_throughput,median_throughput,unit,best_seconds,median_seconds,checksum`.
README documents the env var. Validation passed `make all`, `make test` across
37 shapes, `git diff --check`, a default text smoke with
`COB_BENCH_REPEATS=1 ./build/cob_gemm_bench 64`, and a CSV plus pack-setup
smoke with `COB_BENCH_REPEATS=1 COB_BENCH_CSV=1 COB_BENCH_PACK_SETUP=1
./build/cob_gemm_bench 64`.

Commit `732453f Add benchmark grid helper` added `tools/bench_grid.sh`, a CSV
wrapper for explicit shape arguments or env Cartesian grids via `COB_GRID_M`,
`COB_GRID_N`, and `COB_GRID_K`, plus `COB_GRID_SIZES`, `COB_GRID_BATCH`, and
`COB_GRID_BENCH`. README documents examples. Validation passed
`sh -n tools/bench_grid.sh`, an explicit-shape smoke with `64 128`, and a grid
smoke with `COB_GRID_M="64 96" COB_GRID_N="1024" COB_GRID_K="512"`.

Result: forcing `832` off current AMX direct-B by compiling with
`COB_SGEMM_AMX_STRIDED_B_MAX_N=768` was rejected. Paired A/B for
`832x832x832` repeat-61 showed candidate median `0.9931x`, mean-log `0.9983x`,
CI `[0.9911,1.0070]`, B-faster `15/61`, sign-p `8.84e-05`; the candidate was
consistently slower despite noisy CV. `896x896x896` stayed neutral with median
`1.0000x`, mean-log `0.9998x`, CI `[0.9900,1.0080]`, sign-p `1`. Keep the
current `832` AMX route; no source behavior change. `/usr/bin/xctrace` was
unusable because the active developer directory is CommandLineTools, so
hardware-counter profiling was deferred.

### 2026-05-05: m=64 k=512 SME reuse threshold lowered

Commit `e5f758d` lowered the default
`COB_SGEMM_M64_SME_LONG_N_K512_MIN_N` from the old implicit `32768` to `4096`,
so the `m = 64, k = 512` SME `B`-reuse route applies from `n >= 4096`. It also
added direct-vs-packed correctness coverage for `64x4096x512` and
`64x8192x512`, raising coverage to 39 shapes.

Before the source change, the grid showed weak old one-shot medians around
561.58 GF/s at `64x6144x512`, 484.54 GF/s at `64x8192x512`, and 259.11 GF/s
at `64x16384x512`, while `64x32768x512` jumped to 809.45 GF/s under the
existing `B`-reuse route. A paired old-threshold versus new-threshold
comparison, with the old threshold forced by
`COB_AB_A_FLAGS=-DCOB_SGEMM_M64_SME_LONG_N_K512_MIN_N=32768`, was neutral at
`64x3072x512` and `64x32768x512` but clearly positive from `n = 4096` through
`16384`: median ratios were `1.0164x` CI `[0.9991,1.0330]` at `3072`,
`1.2932x` CI `[1.2646,1.3234]` at `4096` with B faster `61/61` and sign-p
`8.67e-19`, `1.5324x` CI `[1.4818,1.5686]` at `6144`, `2.2419x` CI
`[2.2258,2.3454]` at `8192`, `2.5033x` CI `[2.4697,2.6239]` at `16384`, and
`0.9942x` CI `[0.9860,1.0059]` at `32768`.

Result: accepted. Final validation passed with `make all`, `make test` across
39 shapes, and `git diff --check`. A focused 15-repeat run after the source
commit measured COB one-shot medians of 1390.86 GF/s versus Accelerate 1109.24
at `64x4096x512`, 1290.55 versus 1021.96 at `64x6144x512`, 1182.53 versus
769.16 at `64x8192x512`, 1017.76 versus 599.86 at `64x16384x512`, and 942.71
versus 621.92 at `64x32768x512`. This closes a clear `m = 64, k = 512`
one-shot gap against Accelerate for `n >= 4096`; the universal fastest claim
is still not fully proven because other MpGEMM and square gaps may remain.

### 2026-05-05: rejected m=96/128 skinny direct-B fallback probe

A temporary compile-time `COB_SGEMM_AMX_SKINNY_CHUNK_MAX_N` guard was tested
with `=4096`, so `m = 96/128, n > 4096, k = 512` fell back to AMX direct-`B`
instead of the chunked skinny route.

Result: rejected, and the temporary guard was removed. Paired results were
neutral at `4096`, but clearly worse for larger `n`: `96x4096x512` median
`1.0000x` CI `[0.9980,1.0164]`, sign-p `0.576`; `96x8192x512` `0.8108x` CI
`[0.7956,0.8247]`, sign-p `8.88e-16`; `96x16384x512` `0.7677x` CI
`[0.7563,0.7781]`, sign-p `8.88e-16`; `128x4096x512` `1.0000x` CI
`[0.9934,1.0067]`, sign-p `0.78`; `128x8192x512` `0.7026x` CI
`[0.6959,0.7086]`, sign-p `8.88e-16`; and `128x16384x512` `0.6396x` CI
`[0.5851,0.6274]`, sign-p `8.88e-16`. Keep the chunked skinny route for these
shapes; no source behavior change came from the probe.

### 2026-05-05: wider skinny k512 SME B-reuse route

Commit `85e7afb` generalized the internal SME B-pack/reuse helper from a
hard-coded `m = 64` / four-16-row-panel shape into a loop over 16-row panels.
The existing `m = 64` routes stayed in place, and the same reuse path is now
enabled for `m = 96/128, k = 512, n >= 4096`. It also added direct-vs-packed
tests for `96x4096x512`, `96x8192x512`, `128x4096x512`, and `128x8192x512`,
raising `make test` coverage to 43 shapes.

Before the change, the grid still showed long-`N` one-shot gaps for
`m = 96/128, k = 512`: examples included `96x8192x512` around 483.96 GF/s COB
median versus 665.54 Accelerate, `96x16384x512` 269.33 versus 657.39,
`128x8192x512` 809.15 versus 924.84, and `128x16384x512` 639.32 versus
765.32. A compile-flag probe with
`COB_SGEMM_M96_128_SME_REUSE_K512_MIN_N=8192` was strongly positive at
`8192` and `16384`, and neutral at `4096`.

Result: accepted with threshold `4096`. Paired threshold validation was neutral
at `2048`, then strongly positive: `96x4096x512` median `1.6720x` CI
`[1.6374,1.6768]` with B faster `61/61`, `96x8192x512` `1.8655x` CI
`[1.8486,1.8806]`, `128x4096x512` `1.5446x` CI `[1.5343,1.5572]`, and
`128x8192x512` `1.6819x` CI `[1.6780,1.6983]`. HEAD-vs-candidate paired runs
checked existing `m = 64` route risk: `64x4096x512`, `64x8192x512`,
`64x4096x7168`, and `64x8192x1024` were neutral/noisy, while the new shapes
won strongly: `96x4096x512` `1.7104x`, `96x8192x512` `1.6743x`,
`128x4096x512` `1.5913x`, and `128x8192x512` `1.5505x`, all B faster
`61/61`.

Validation before commit passed with `make all`, `make test` across 43 shapes,
and `git diff --check`. A focused post-change benchmark measured COB one-shot
median versus Accelerate at `96x4096x512` 1443.20 versus 1011.69 GF/s,
`96x8192x512` 1167.11 versus 837.99, `96x16384x512` 1089.72 versus 672.21,
`128x4096x512` 1542.73 versus 1195.70, `128x8192x512` 1439.33 versus 975.24,
and `128x16384x512` 1138.04 versus 388.83. `96x2048x512` is outside the new
route; one outlier row collapsed, but an immediate duplicate rerun returned to
about 1032-1038 GF/s, matching the paired neutral result. This closes another
skinny `k = 512` one-shot gap for `n >= 4096`; the universal fastest claim is
still not fully proven.

### 2026-05-05: wider skinny k1024 SME B-reuse route

Commit `ff0332b` extended the generalized SME B-reuse helper from `85e7afb`
to `m = 96/128, k >= 1024, n >= 4096`, using
`COB_SGEMM_M96_128_SME_REUSE_K1024_MIN_N=4096`. It added direct-vs-packed
tests for `96x8192x1024`, `128x4096x1024`, and `128x8192x1024`, raising
`make test` coverage to 46 shapes.

After the k512 route, the grid still showed k1024 gaps: `96x4096x1024`
measured about 679.58 GF/s COB median versus 891.81 Accelerate,
`96x8192x1024` 580.19 versus 801.70, `96x16384x1024` 456.85 versus 694.98,
`128x4096x1024` 756.69 versus 992.37, `128x8192x1024` 680.88 versus 911.88,
and `128x16384x1024` 549.09 versus 854.21. A compile-flag probe was neutral at
`2048` and strongly positive at `4096+`: `96x4096x1024` median `1.6868x` CI
`[1.6560,1.6986]`, `96x8192x1024` `1.9255x` CI `[1.8973,1.9606]`,
`96x16384x1024` `2.2737x` CI `[2.1768,2.2854]`, `128x4096x1024` `1.6486x` CI
`[1.6269,1.6692]`, `128x8192x1024` `1.8273x` CI `[1.7853,1.8423]`, and
`128x16384x1024` `2.1552x` CI `[2.1216,2.1888]`, with `51/51` B-faster on the
winners.

HEAD-vs-candidate paired validation against the pre-`ff0332b` baseline was
neutral at `96x2048x1024` and `128x2048x1024`, while the routed shapes won:
`96x4096x1024` median `1.7352x` CI `[1.6622,1.7840]`, `96x8192x1024`
`2.0858x` CI `[1.9582,2.0931]`, `128x4096x1024` `1.6755x` CI
`[1.6364,1.7290]`, and `128x8192x1024` `1.9331x` CI `[1.8644,1.9405]`.
Existing k512 shapes also remained strong.

Result: accepted. Validation before commit passed with `make all`,
`make test` across 46 shapes, and `git diff --check`. Focused post-change
COB one-shot medians versus Accelerate were mostly strong wins:
`96x8192x1024` 1141.47 versus 826.80 GF/s, `96x16384x1024` 1147.98 versus
767.51, `128x4096x1024` 1350.62 versus 975.24, `128x8192x1024` 1336.33
versus 981.93, and `128x16384x1024` 1256.21 versus 803.85. The first
`96x4096x1024` median was noisy, but duplicate reruns measured COB one-shot
1066.63 and 1214.64 GF/s versus Accelerate 731.43 and 759.01. This closes
another skinny one-shot k1024 gap for `n >= 4096`; the universal fastest claim
is still not fully proven.

Follow-up commit `aa1811e` covered higher-`k` samples for the same route. A
grid check over `m = 96/128`, `n = 4096/8192`, and `k = 1024/2048/4096` kept
the skinny SME reuse path faster than Accelerate in those samples. It added
direct-vs-packed correctness tests for `96x4096x2048` and `128x8192x4096`,
raising `make test` coverage to 48 shapes. Validation passed with `make all`,
all 48 test shapes, `git diff --check`, and a pushed commit.

## Current Conclusion

COB is very competitive in its exact current scope. The packed-`B` AMX path is
around 1.9-2.0 TF/s on large square cases and beat `blis_apple`, OpenBLAS,
BLASFEO, Rust `matrixmultiply`, Eigen, `coral-aarch64`, LIBXSMM exact
`beta = 0`, and the comparable `tract-linalg` paths in the tested cases.

Post-`5e6da0a` rejected/probed follow-ups:

- Successful tuple-B load/store follow-up: ACLE tuple `svld1_x4` / `svst1` was
  tested in `/private/tmp/cob_tuple_b_exp` and passed all 37 tests. The full
  tuple experiment improved several blockers, with repeat-25 medians of
  `64x24576x1536` 975.14 GF/s, `64x7168x2048` 1029.62 GF/s,
  `64x7168x16384` 980.65 GF/s, `64x4096x7168` 982.00 GF/s, and
  `64x2112x7168` 1214.90 GF/s, but regressed `64x32768x512` to
  767.23 GF/s. The main selective patch keeps tuple direct strided and tuple
  `m = 64` B-reuse except long-`n`, `k = 512`, which stays on the old helper.
  Selective validation passed `make test` across 37 shapes and
  `git diff --check`. Focused repeat-25 selective medians were
  `64x24576x1536` 971.42 GF/s, `64x7168x2048` 1017.35 GF/s,
  `64x7168x16384` 988.00 GF/s, `64x8192x1024` 999.76 GF/s,
  `64x4096x7168` 989.23 GF/s, and `64x2112x7168` 1210.35 GF/s;
  `64x32768x512` was noisy, but A/B repeat-31 measured default 896.28 GF/s
  versus selective 916.16 GF/s.
- Direct-SME-wide worktree `/private/tmp/cob_sme_direct_wide_exp` disabled wide
  B-reuse and let direct strided SME cover `n >= 7168, k >= 1024`. It passed
  all 37 GEMM tests, but repeat-25 one-shot results were much worse on wide
  shapes: `64x24576x1536` median 437.31 GF/s, `64x7168x2048` 637.61 GF/s,
  `64x7168x16384` 610.06 GF/s, and `64x8192x1024` 564.24 GF/s. Reject.
- Compile-time `NC` probes for `m = 64` B-reuse did not produce a commit-worthy
  default. `NC = 256` improved some medians in one run but hurt or was noisy
  for `64x32768x512`; `NC = 512` looked modestly better in repeat-25, including
  `64x24576x1536` median 950.59 GF/s, `64x7168x2048` 997.37 GF/s, and
  `64x8192x1024` 1024.56 GF/s, but repeat-31 was inconclusive/noisy.
  `NC = 768` was not better. No source commit yet.
- `703de33` (`Tighten m64 SME reuse chunks`) revisited the `m = 64` B-reuse
  chunk size after tuple B loads/stores landed in `10849f3` and were documented
  in `3cb1ee2`. Retesting `COB_SGEMM_M64_SME_REUSE_NC=512` showed it had become
  a consistent win over the previous `1024` chunk size for the tuple-era
  B-reuse path. Before the source commit, `make test` passed all 37 shapes,
  `make all` rebuilt, and `git diff --check` passed. Focused repeat-25 after
  the source change measured `64x4096x7168` 1029.05 GF/s, `64x7168x16384`
  1029.47 GF/s, `64x24576x1536` 936.58 GF/s, `64x7168x2048` 1013.51 GF/s,
  `64x32768x512` 935.72 GF/s, `64x2112x7168` 1235.82 GF/s, and
  `64x8192x1024` 1053.72 GF/s. This improves the tuple-era B-reuse route, but
  MpGEMM still remains ahead on some shapes, especially `64x4096x7168`,
  `64x7168x16384`, and `64x8192x1024`.
- `47e5e7a` (`Tune long wide m64 SME chunks`) followed up after `703de33` made
  `NC = 512` the default. A narrow long-wide gate was tested and committed:
  use `COB_SGEMM_M64_SME_LONG_WIDE_NC=256` only when `m = 64`, wide B-reuse
  applies, `n >= 24576`, and `k == 1536`; otherwise keep `NC = 512`.
  Validation before the source commit passed `make test` across 37 shapes,
  `make all`, and `git diff --check`. Focused repeat-25 after the source change
  measured `64x24576x1536` median 1020.02 GF/s, `64x32768x512` median
  937.77 GF/s, `64x7168x16384` median 1025.75 GF/s, and `64x4096x7168`
  median 1029.62 GF/s. This targets the `64x24576x1536` MpGEMM comparison:
  MpGEMM's `O0` timing driver measured best 1042.92 GF/s and median
  853.23 GF/s in a noisy but usable separate-process run, while COB now
  measured best 1031.78 GF/s and median 1020.02 GF/s in the focused bench.
- `COB_SGEMM_M64_SME_REUSE_NC=384` was rejected after a broad paired run looked
  only weakly positive and focused validation did not hold up. Broad paired
  ratios were `64x4096x7168` median 1.0046x, mean-log 1.0080x, CI
  [0.9987,1.0223], sign-p 0.349; `64x7168x2048` median 1.0016x, CI
  [0.9960,1.0127], sign-p 0.755; `64x7168x16384` median 1.0034x, CI
  [0.9958,1.0089], sign-p 0.0596; and `64x8192x1024` median 1.0049x, CI
  [1.0006,1.0121], sign-p 0.0784. Focused validation rejected it:
  `64x7168x16384` repeat-81 median 1.0002x, mean-log 1.0002x, CI
  [0.9911,1.0077], sign-p 1; `64x8192x1024` repeat-51 median 0.9990x,
  mean-log 1.0011x, CI [0.9958,1.0068], sign-p 1. No source behavior change.
- `KC` probes also stayed uncommitted. `NC512 + KC1024` and targeted
  `k == 2048` / `k == 1024` variants were not clean enough to commit; the
  `k == 1024`-only repeat-31 gave `64x8192x1024` median 970.83 GF/s and
  `64x7168x1024` 952.86 GF/s, so reject.
- The 64x16 SME strided kernel temp worktree
  `/private/tmp/cob_sme_64x16_exp` passed all 37 tests but was much slower:
  `64x24576x1536` median 564.27 GF/s, `64x7168x2048` 708.54 GF/s,
  `64x4096x7168` 667.99 GF/s, and `64x2112x7168` 887.26 GF/s. Reject.
- Post-`030448f` direct-SME-wide route in
  `/private/tmp/cob_sme_direct_wide_exp` disabled wide B-reuse and let direct
  strided SME cover `n >= 7168, k >= 1024`. It was correct, but much slower on
  wide `m = 64` shapes, so reject.
- Post-`030448f` tuple chunk sweeps were rejected. Global `NC = 256` helped
  `64x24576x1536` but hurt broader targets, and `KC = 640`, `896`, and `1024`
  variants were worse than the current `KC = 768`.
- The `n = 2112` B-reuse gate was correct but regressed to about 995 GF/s
  median. Direct tuple SME remains better, so reject.
- The 64x16 SME strided kernel was correct but much slower. Reject.
- Panel-immediate reuse and fused-four helper variants were correct but slower.
  Reject.
- Tuple pack-only and tuple reuse-only split experiments were correct but worse
  or noisy. Reject.
- One-shot SME packed `n = 1024` plus the medium gate, low gate, and
  packed-384 experiments were correct but did not close the `768` / `1024` /
  `1280` gaps. Reject.
- Packed-B micro-experiments were rejected. `x2` loads and K-loop unroll were
  correct but slower; the first `vg4` column-store failed correctness, and the
  corrected `vg4` row-group store passed tests but lost packed-B median at
  `768`, `1024`, and `1280`.
- MpGEMM checkout `/private/tmp/mpgemm_latest` was refreshed and was already up
  to date at `8d83011`.

The universal "fastest" claim is not achieved if MpGEMM counts, because MpGEMM
won several same-process one-shot FP32 single-thread square benchmarks.
Accelerate also still wins some small cases. The SME packed-`B` path improves
COB's packed-`B` result, and the narrow SME route improves one-shot `n = 512`,
but not enough to change that conclusion. Later ACL attempts were either
crashing or far slower, and the local direct-`B` SME route remains promising but
shape-limited.
