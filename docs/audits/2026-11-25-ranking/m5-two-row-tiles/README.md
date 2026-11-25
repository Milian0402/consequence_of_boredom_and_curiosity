# Direct B on two Apple SME2 output row tiles

On one **Apple M5 Max**, COB's new compact path beat the preceding opt-in on **all eight previously unmeasured layouts, on both seeds, in both builds**. The released-source candidate averaged **1.057×** its incumbent and **1.019× the fastest external library selected independently for each shape**. The research build measured 1.070× and 1.032× respectively. Five fresh layouts beat every measured external peer by at least 1% in both builds. Prepared LIBXSMM led on two layouts in both builds, and the integrated candidate tied MpGEMM to four significant figures on one more.

The mechanism applies to FP32 row-major A, B and C with **24–32 output rows, 1024–4096 columns and reduction depth 96–384**, all stored compactly. With two 16-row output tiles, the implementation reads B twice directly and avoids allocating and writing a packed B panel. A is still packed once per call. The surrounding runtime checks require a supported 16-lane Apple SME2.1 implementation, adequate strides and disjoint output. The pre-existing opt-in remains `COB_ENABLE_APPLE_MEDIUM_WIDE=ON` for Make or CMake and defaults to OFF. The source is commit `69b239d6ab9b9a86425ef6874634e3a1952a7f73` on `602a9dddce7e940a282b0716a7b2ca40bd2ad35e`.

## Warm, single-thread results

Ratios above one favor COB. The previous opt-in and every external library received the same A, B **and output C address** within a process. Each fresh shape ran in two seeded processes per build. Two labels resolved to one incumbent function pointer. The schedule balanced every method's position and every adjacent pair of methods using repeated forward and reversed Latin orders. All ordinary public timings included allocation, data preparation inside the library and computation. The prepared LIBXSMM competitor called its verified, already-generated kernel directly. Other competitors were explicitly serial Apple Accelerate, OpenBLAS, ArmPL, original MaxMulSK and MpGEMM on compact data.

| M × N × K | Cohort | vs incumbent, initial / integrated | Fastest external, integrated | vs fastest, initial / integrated |
|---|---|---:|---|---:|
| 24 × 1280 × 131 | repeat-positive-24 | 1.246× / 1.132× | prepared-LIBXSMM | 1.025× / 1.015× |
| 32 × 2496 × 184 | repeat-positive-32 | 1.062× / 1.048× | MpGEMM | 1.030× / 1.024× |
| 24 × 1088 × 280 | fresh-aligned-1 | 1.039× / 1.019× | prepared-LIBXSMM | 0.944× / 0.948× |
| 27 × 2240 × 96 | fresh-aligned-2 | 1.142× / 1.115× | MpGEMM | 1.140× / 1.124× |
| 29 × 2624 × 216 | fresh-aligned-3 | 1.046× / 1.033× | MpGEMM | 1.025× / 1.015× |
| 32 × 3904 × 368 | fresh-aligned-4 | 1.033× / 1.020× | MpGEMM | 1.015× / 1.000× |
| 25 × 2287 × 115 | fresh-odd-1 | 1.152× / 1.150× | Accelerate | 1.104× / 1.045× |
| 26 × 3487 × 307 | fresh-odd-2 | 1.048× / 1.013× | MpGEMM | 1.048× / 1.037× |
| 30 × 1043 × 275 | fresh-odd-3 | 1.077× / 1.076× | prepared-LIBXSMM | 0.940× / 0.941× |
| 31 × 3135 × 187 | fresh-odd-4 | 1.033× / 1.037× | MpGEMM | 1.052× / 1.053× |
| 33 × 2880 × 232 | control-above-32 | 1.001× / 1.001× | MpGEMM | 1.012× / 1.004× |
| 28 × 2673 × 229 | control-padded | 1.015× / 1.011× | prepared-LIBXSMM | 0.969× / 0.978× |

The integrated build averaged 1.064× serial Accelerate, 1.060× prepared LIBXSMM, 1.054× MpGEMM and 1.513× MaxMulSK across the eight fresh shapes. Its per-shape fastest-peer ratio ranged from 0.941× to 1.124×. Same-function median ratios across every layout were 0.993–1.026× initially and 0.998–1.004× in the integrated build. A separate first-call benchmark started each method in its own process and included explicit LIBXSMM initialization and dispatch as well as the product. On the fresh layouts, integrated COB averaged 1.150× its incumbent and 1.229× the fastest process-first peer. Each method had two such observations per shape. Dynamic loading completed before the clock started; OS caches were left in their natural state.

## Correctness and scope

Each build supplied **11,280 warm observations**, 188 process-first observations and 117,520,368 FP64 cell checks, including full products before the warm trials and during the final six rounds. The largest absolute error on the normalized timing fixtures was 0.00002186. The maximum task-thread count at the measured boundaries was one. The warm crossover used user-initiated QoS on macOS, without a guaranteed P-core affinity.

The integrated and prototype binaries each passed 98 independent cases with protected allocation boundaries, including 74 actual compact selections and 24 bypasses. All 5,217,900 outputs were bit-identical to the previous opt-in on the tested normalized, exact, cancellation, reciprocal-scale and correlated inputs. All 15 allocation-failure/retry, invalid-input and overlap controls passed. On correlated inputs scaled by 32, both binaries and the incumbent had 112,377 cells above 0.002 absolute error, of which 4,886 exceed that threshold even for correctly rounded FP32. The route changed performance while preserving this existing numerical limitation on the tested fixtures. The CMake build passed 4/4 tests. Make passed its three 667-shape configurations and the dedicated public API regression test.

The earlier 24–55-row direct-B candidate missed its promotion rule on a separate eight-shape panel, and the 53-row holdout regressed. Long separately buffered control timings had substantial same-function variation. This follow-up froze the narrower algorithm first, used independent seeds and geometries, and gave all competitors the identical buffers. The exact field includes the six external implementations named above. These results neither establish a global ranking nor cover padded inputs, larger row counts, other CPU families or multithreaded products.

## Reproduction

Run `python3 score.py initial` and `python3 score.py integrated` in the extracted evidence directory to reproduce both decisions. The scorers check the input panel, rotation balance, method presence, sample counts, CPU/thread receipts, normalized FP64 error and the frozen numerical thresholds. The native build-and-timing replay is `python3 native-replay.py RELEASE_REPO LIBXSMM_CHECKOUT PEERS.csv NEW_DIRECTORY` on a supported Apple Silicon host. The peer CSV uses `method,path` for Accelerate, OpenBLAS, ArmPL, MaxMulSK and MpGEMM, pointing to the serial row-major benchmark adapters. The archive contains the full ordinary public patch, independent panel receipt, raw timings, verification sources, immutable hashes and correctness results.
