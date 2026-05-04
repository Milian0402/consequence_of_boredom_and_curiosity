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
