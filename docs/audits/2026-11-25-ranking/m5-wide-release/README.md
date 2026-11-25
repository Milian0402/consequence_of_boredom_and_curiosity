# M5 Max thin medium-wide FP32 multiplication

**The installed default-off kernel averaged 4.67× released COB on nine fresh layouts and beat it on all nine with both seeds.** Its smallest per-layout speedup was 1.94×. It averaged 0.972× the fastest external method per layout and won by at least 1% on 2 of nine. The frozen opt-in rule passed this installed trial and the preceding confirmation. A subsequent clean rebuild fell below the external floor of the opt-in rule. All three runs missed the separate external-field victory rule.

The public function handles 24–55 output rows, 1024–4096 columns and inner dimensions 96–384 on supported Apple SME2.1 hardware. A is packed once. The first 16-row output tile consumes B while packing its 64-column panel for subsequent rows. Every public call pays its own workspace and packing cost. The implementation is in commit `602a9dddce7e940a282b0716a7b2ca40bd2ad35e` on `a30d71f0aaba4bcceb9ec5c530cffd63e6967cd6`. Enable `COB_ENABLE_APPLE_MEDIUM_WIDE=ON` with Make or use `-DCOB_ENABLE_APPLE_MEDIUM_WIDE=ON` with CMake.

## Installed release comparison

Six warm one-thread trials per layout across two seeds, with counterbalanced order. The provider listed won on median total latency among independent rivals. The ratio uses matching within-round observations. The independent field includes explicitly serial Accelerate, ArmPL, OpenBLAS and directly invoked prepared LIBXSMM. MaxMulSK and MpGEMM also participated on compact layouts. Every LIBXSMM entry was a verified generated matrix-multiply function. Each warm measurement excludes LIBXSMM's JIT setup and includes all ordinary COB work.

| M × N × K | Pad | Cohort | COB µs | vs release | Fastest independent | vs fastest |
|---|---:|---|---:|---:|---|---:|
| 25 × 1087 × 97 | 3 | repeat-hardest | 4.44 | 14.084× | Accelerate | 0.881× |
| 41 × 2377 × 139 | 11 | repeat-mid-thin | 18.98 | 12.155× | prepared-LIBXSMM | 0.965× |
| 55 × 3072 × 184 | 1 | repeat-upper-thin | 49.05 | 8.573× | Accelerate | 1.104× |
| 24 × 1280 × 131 | 0 | fresh-24-compact | 6.62 | 2.584× | prepared-LIBXSMM | 0.957× |
| 28 × 1057 × 107 | 9 | fresh-28-narrow | 4.87 | 14.291× | prepared-LIBXSMM | 0.899× |
| 31 × 4033 × 199 | 7 | fresh-31-wide | 34.22 | 9.208× | prepared-LIBXSMM | 0.947× |
| 34 × 1799 × 241 | 3 | fresh-34-odd | 23.77 | 7.266× | prepared-LIBXSMM | 0.946× |
| 39 × 2304 × 160 | 0 | fresh-39-aligned | 19.98 | 2.642× | Accelerate | 1.040× |
| 46 × 2983 × 353 | 13 | fresh-46-deep | 62.24 | 6.499× | prepared-LIBXSMM | 0.977× |
| 49 × 2059 × 127 | 5 | fresh-49-odd | 18.92 | 2.005× | Accelerate | 0.991× |
| 52 × 3403 × 283 | 7 | fresh-52-wide | 75.79 | 6.424× | Accelerate | 1.027× |
| 55 × 1536 × 384 | 0 | fresh-55-aligned | 46.32 | 1.940× | prepared-LIBXSMM | 0.970× |
| 23 × 1379 × 199 | 7 | control-below | 90.29 | 1.020× | Accelerate | 0.105× |
| 56 × 1597 × 239 | 3 | control-above | 57.16 | 1.000× | Accelerate | 0.448× |

The installed candidate averaged 1.005× serial Accelerate, 1.003× prepared LIBXSMM and 4.81× the installed OFF build on the nine fresh layouts. The original 25 × 1087 × 97 loss was intentionally included among the three repeat anchors. The 23- and 56-row controls use the existing release fallback. All 624 warm observations used one caller thread, with one task thread observed before and after each timed batch. The M5 Max used user-initiated QoS; macOS supplied no CPU affinity guarantee.

For a separate first-call comparison, each of 208 method/layout/seed combinations ran in its own new process. Inputs and an independent FP64 reference were prepared before the call. Dynamic loading was excluded for all methods. Explicit initialization, JIT generation and the first multiply are included for freshly prepared LIBXSMM. OS caches were left in their natural state. Geometric means over the nine fresh cases were 3.91× versus release and 1.21× versus the fastest external process-first time per shape. Two observations per method and layout give less precision than the warm study.

## Clean rebuild and timing variability

A separate native replay used the included reproduction script. It checked out the original release, applied the exported patch, built and installed both configurations from source, and repeated all 624 warm trials and 208 process-first calls. On the nine fresh layouts, the source-rebuilt candidate still beat released COB on every layout and both seeds, with a **3.49×** geometric mean. It averaged **0.950×** the strongest external peer. At 39 × 2304 × 160, its speedup relative to MpGEMM was **0.782×**, below the declared floor of 0.85×. The preceding confirmation and installed trial cleared this floor. The option's improvement over the earlier COB release repeated. Its per-shape floor relative to other libraries did not.

The 39-row candidate ranged from 19.4 to 22.8 µs in the installed trial and from 23.0 to 71.7 µs after the clean rebuild. Cause was not established. Both observations retained their original matched trials. The full stress data were byte-identical across the two builds.

## Numerical behavior

All 73,317,128 checked output cells in the installed warm run and 18,329,282 separate first-call cells passed the 0.002 absolute threshold on normalized fixtures. The largest observed absolute error in the warm study was 0.00002748.

The installed ON and OFF builds produced **bit-identical output on all 8,789,904 stress-test cells**. The larger correlated cases nevertheless exceed a 0.002 absolute tolerance. The final table column counts outputs for which even a correctly rounded FP32 result would exceed that tolerance.

| Fixture | Cells | Largest absolute error, ON and OFF | Above 0.002 in each build | Above 0.002 with nearest FP32 |
|---|---:|---:|---:|---:|
| cancellation | 1,464,984 | 0.00000261 | 0 | 0 |
| correlated16 | 1,464,984 | 0.02248727 | 24,736 | 0 |
| correlated32 | 1,464,984 | 0.08994909 | 250,514 | 10,667 |
| correlated4 | 1,464,984 | 0.00140545 | 0 | 0 |
| normalized | 1,464,984 | 0.00002710 | 0 | 0 |
| reciprocal | 1,464,984 | 0.00002710 | 0 | 0 |

Correlated inputs set B[p,j] to A[j modulo M,p] and scale both inputs by 4, 16 or 32. Reciprocal inputs redistribute a power-of-two factor across A and B at each inner index, leaving the exact dot product unchanged. Each fixture uses three seeds at six geometries. All input snapshots, output padding and sentinels survived. Sixty independent protected-row tests on each of the CMake ON, OFF, route-traced, Make ON and Make OFF builds checked 6,638,493 cells per build. All passed, including controls beyond the route bounds. Read-only page protection covered both inputs, and a separate probe checked shared input pointers and bypass for overlapping output ranges.

## Selection and reproduction

The earlier broad treatment beat released COB on all eight original fresh M5 shapes with a geometric mean of 2.77×. Its strongest-peer ratio was 0.978× and it failed the broad acceptance rule. Its 16×64 arithmetic qualified for the narrower default-off option on this separate follow-up. Before the new panel ran, the study declared distinct deployment and field-victory rules. Against that wider predecessor, an alternate 32×32 direct-B refinement averaged 0.976× and was discarded. The source-identical 16×64 arithmetic, before final build integration, averaged 4.84× old release and 0.994× the strongest per-layout rival. Both the installed and pre-integration measurements are preserved.

The independent panel selector found none of the nine fresh shapes or two controls in 752 previously saved geometry tables. The field and treated architecture remain limited to the methods and machine tested here. All installed CMake tests passed (4/4). Make passed its tests with the same build directory switched from ON to OFF, and the disabled archive lacked the specialized symbol.

Run `python3 verify-results.py .` inside the evidence folder to recompute the published ratios and numerical summary. Native benchmarking on Apple silicon uses `python3 reproduce.py RELEASE_REPO LIBXSMM_CHECKOUT PEERS.csv NEW_OUTPUT_DIR`. The peer file supplies absolute shared-library paths using columns `method,path` and the names Accelerate, OpenBLAS, ArmPL, MaxMulSK and MpGEMM. The program compiles the ordinary public ON/OFF archives and baseline from the recorded Git revision and patch. Run `python3 verify-results.py clean-rebuild .` to recompute the separate clean rebuild using the original protected-page receipts. The package includes the normalized, first-call, and scaled input fixtures, the guards, the per-method hashes and observed JIT code sizes. The public patch and test source carry the repository licenses.
