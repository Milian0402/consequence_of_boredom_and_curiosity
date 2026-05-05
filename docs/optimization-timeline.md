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

## Current Conclusion

COB is very competitive in its exact current scope. The packed-`B` AMX path is
around 1.9-2.0 TF/s on large square cases and beat `blis_apple`, OpenBLAS,
BLASFEO, Rust `matrixmultiply`, Eigen, `coral-aarch64`, LIBXSMM exact
`beta = 0`, and the comparable `tract-linalg` paths in the tested cases.

The universal "fastest" claim is not achieved if MpGEMM counts, because MpGEMM
won several same-process one-shot FP32 single-thread square benchmarks.
Accelerate also still wins some small cases. The SME packed-`B` path improves
COB's packed-`B` result, and the narrow SME route improves one-shot `n = 512`,
but not enough to change that conclusion. Later ACL attempts were either
crashing or far slower, and the local direct-`B` SME route remains promising but
shape-limited.
