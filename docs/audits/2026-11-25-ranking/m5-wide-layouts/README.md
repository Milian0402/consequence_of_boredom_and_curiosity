# Apple M5 Max compact SGEMM with 24–55 rows

**Streaming B directly from its input matrix failed the promotion rule on eight new compact layouts.** The first field comparison averaged 1.071× the existing COB opt-in and 0.972× the fastest external method per layout. Building both public libraries with identical CMake flags yielded 1.054× the incumbent and 0.959× the fastest external. Controlled comparisons narrowed the apparent gain over COB to 1.000× with longer batches and 1.038× with short, interleaved batches. The 24–55-row experiment remains unshipped.

All measurements use row-major FP32 on one Apple M5 Max and one caller thread. Candidate and incumbent both receive ordinary matrices through the public SGEMM entry point. Both pay their own allocation, packing, arithmetic and output costs. One-shot comparisons include serial Accelerate, OpenBLAS, ArmPL, MaxMulSK and MpGEMM where their public input contracts apply. Prepared LIBXSMM executes a generated kernel with its dispatch/setup cost excluded from warm timings and charged to first invocation. Compact-only MaxMulSK and MpGEMM were present on every fresh layout. The padded bypass control uses the supported subset.

## Candidate selection

The incumbent at `602a9dddce7e940a282b0716a7b2ca40bd2ad35e` packs each B panel while computing its first row tile. Three alternatives used only the three historical compact layouts for selection. The winning geometric mean was frozen before measuring the independent panel. Ratios above one favor the proposed kernel.

| Training candidate | 24 × 1280 × 131 | 39 × 2304 × 160 | 55 × 1536 × 384 | Geometric mean |
|---|---:|---:|---:|---:|
| Read B directly in 64-column panels | 1.067× | 1.045× | 0.987× | 1.032× |
| Read 16 B columns per vertical strip | 0.686× | 0.658× | 0.920× | 0.746× |
| Pack four neighboring B panels together | 0.634× | 0.834× | 0.956× | 0.797× |

The direct candidate keeps the packed A representation and reads original B for each 16-row output tile. It accepts compact calls with 24–55 rows, 1024–4096 columns, and inner dimension 96–384. Two seeds and three warm rounds per seed determined the training scores. The frozen panel contains three repeated anchors, eight fresh compact products and three untreated controls. The independent geometry scan found no matching triples for the eight new layouts in its saved table and archive inventory.

## Fresh layouts

Warm ratios are medians of eight within-round observations, four per input seed. “Matched” identifies the repeat using separately built and installed Release CMake archives with the same compiler, SDK, flags and public wrapper. The column labeled “fastest peer” names the leading independent method in that repeat. The long and rapid columns use the geometric mean of two incumbent observations divided by the candidate observation in each matched triple.

| M × N × K | Initial vs incumbent | Matched vs incumbent | Fastest peer, matched | Matched vs peer | Long vs incumbent | Rapid vs incumbent |
|---|---:|---:|---|---:|---:|---:|
| 28 × 1920 × 168 | 1.195× | 1.107× | Accelerate | 0.982× | 0.942× | 1.160× |
| 32 × 2496 × 184 | 1.154× | 0.962× | Accelerate | 1.011× | 0.889× | 1.082× |
| 40 × 3520 × 288 | 1.068× | 1.196× | MpGEMM | 1.052× | 0.862× | 0.992× |
| 48 × 3776 × 360 | 1.158× | 0.902× | Accelerate | 1.185× | 1.303× | 0.983× |
| 25 × 1761 × 315 | 1.169× | 1.062× | Accelerate | 0.913× | 0.927× | 1.081× |
| 31 × 2237 × 281 | 1.033× | 1.027× | Accelerate | 0.867× | 1.133× | 1.095× |
| 43 × 3301 × 167 | 0.970× | 1.136× | MpGEMM | 0.805× | 0.968× | 1.013× |
| 53 × 3757 × 129 | 0.868× | 1.064× | MpGEMM | 0.906× | 1.043× | 0.916× |

The rule required at least 1.04× the incumbent overall, six wins per seed and a 0.95× per-layout floor. It also required 1.01× the fastest independent method overall, five layouts beating that method by more than 1%, and a 0.92× floor. The matched run beat the fastest peer by more than 1% on three layouts, with a 0.805× floor. The two seeds recorded six and five incumbent wins. The matched first-call speedups averaged 1.195× over the incumbent and 1.160× over the fastest independent first call. There were two process-first observations per shape and method. Library loading occurred outside the timer.

On the historical 39 × 2304 × 160 case, the matched trial achieved 1.077× the incumbent and 1.169× MpGEMM, its fastest independent peer. That case was used for training. The 55-row historical anchor fell to 0.964× the incumbent and 0.882× its leading peer. Full timing tables retain every anchor and control.

## What the same-function controls show

An initial untreated 56-row call apparently gained 28.5%, which prompted the identical-toolchain rebuild and two same-function controls. The matched rebuild still showed an 8.6% difference on that untreated call. In every control process, two labels pointed to the identical incumbent function address and a third label pointed to the new candidate. Each label occupied every order position equally.

The **long-batch** control used 504 observations. Each pair of incumbent labels had its own output buffer. Their median time ratios ranged from 0.864× to 1.100× across the 14 layouts, and nine exceeded a 3% departure from unity. For eight fresh layouts, the candidate ratio against the paired incumbent averaged 1.000×. Only 48 × 3776 × 360 and 31 × 2237 × 281 cleared the predeclared corroboration rule on both seeds. No bypass control cleared it.

The **rapid crossover** used 5,040 observations, one shared output address and ten repetitions of every A/B/D ordering per seed. Incumbent-label wall-time medians ranged from 0.996× to 1.005×. Candidate speedups averaged 1.038× by wall time and 1.037× by caller CPU time. Three fresh layouts exceeded 3% on both seeds. The 31 × 2237 × 281 case was the only one supported under both diagnostics' respective rules. The candidate's rapid speedup there was 1.095× the incumbent, while the matched external field still put it at 0.867× Accelerate.

## Accuracy and replay

Each complete external-field run retained 880 warm observations, 220 separate first-call processes, 128,628,360 full-reference output-cell checks and maximum absolute error 0.000026455 on normalized inputs. An independent protected-memory verifier found all 7,183,072 outputs bit-identical between the incumbent, candidate and traced candidate. The trace observed 74 selections and 20 bypasses. All 15 allocation-failure, retry, alias and invalid-input checks passed. Both kernels retained the same errors on more strongly scaled inputs. In the correlated ×32 fixture, 177,437 outputs exceeded a 0.002 absolute tolerance, including 4,641 where even the nearest representable FP32 result exceeded that tolerance.

Run `python3 verify-study.py .` from the unpacked archive to verify checksums and reproduce the training selection, both field scores, both same-function diagnostics and the independent correctness summary. Source fragments and patches for all three treatments are included. `freeze.sha256` is the original freeze ledger. Recorded binaries are identified by hashes, and the evidence archive ships their sources and build scripts. The matching compiler was Apple clang 21.0.0 with macOS SDK 27.2. `matched/toolchain.json` confirms identical public driver object bytes across the two matched builds and identifies both SME objects and archives.

To repeat the native matched field with a local copy of the pinned base, a built LIBXSMM checkout and the five single-thread peer-adapter libraries, run `python3 replay.py COB_REPO LIBXSMM_CHECKOUT PEERS.csv NEW_BUILD_DIR`. Supply a CSV with columns `method,path`, and names Accelerate, OpenBLAS, ArmPL, MaxMulSK and MpGEMM. The separate native diagnostic command is `python3 replay-controls.py NEW_BUILD_DIR LIBXSMM_CHECKOUT NEW_CONTROL_DIR`. The raw scores describe the observed M5 Max, input range and methods.
