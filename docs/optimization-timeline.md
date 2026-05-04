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

### 2026-05-05: post-841ee51 rejected experiments

Widening the post-pack SME route beyond `n == 512` was tested in
`/private/tmp/cob_postpack_sme_exp` and rejected because it regressed
`n = 768` and `n = 1536`, and did not beat the existing direct-`B` route for
gated sizes.

Changing `COB_SGEMM_AMX_MC` from `384` to `512` was tested in
`/private/tmp/cob_mc512_exp` and rejected because it hurt packed-`B` at larger
sizes such as `1280` and `1408` without a sufficient one-shot gain.

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
