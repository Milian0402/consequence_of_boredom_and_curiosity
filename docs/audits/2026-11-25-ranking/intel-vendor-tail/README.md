# Single-thread x86 medium-wide FP32 multiplication with matched BLAS providers

The one-to-four-row COB tail did not improve the vendor path across the four shapes chosen to test it on the Intel Xeon 6975P-C. Relative to a full public call to the same BLAS, the split achieved **0.952× for OpenBLAS** and **0.970× for oneMKL**. Against matched direct CBLAS the results were **0.957×** and **0.973×**. The release keeps its existing dispatch.

Calling oneMKL for the entire result through COB did beat the standalone release by **1.100×** across all eight fresh, eligible layouts on a geometric mean, and beat stream-B by **1.039×**. It beat stream-B on just four of those eight. Against an explicit call to the very same sequential oneMKL provider, the measured ratio was **1.002×**. Its ratio to the strongest independent implementation was **0.990×**. The experiment provides a way to reproduce oneMKL performance through COB’s API. A direct single-thread oneMKL call already supplies that performance, and the measured regressions against stream-B do not justify a new maintained production option.


## Whole-call comparisons

Each candidate is the public COB entry point with a selected single-thread provider. The full variant delegates all eligible rows to that provider. The split variant uses the same provider for the initial rows and COB's AVX2/FMA kernel for a final one to four rows. The timed standalone COB and stream-B entries used COB's own kernels. Each matched direct comparator and both wrappers use the same LP64 provider binary and CPU. Values above 1 favor the named candidate.

| Candidate | Frozen group | vs matched direct | vs public full | vs standalone release | vs stream-B | vs best independent |
|---|---|---:|---:|---:|---:|---:|
| OpenBLAS full | Five original holdouts | 0.987× | 1.000× | 0.999× | 0.890× | 0.831× |
| OpenBLAS split | Five original holdouts | 0.987× | 0.999× | 0.998× | 0.889× | 0.831× |
| oneMKL full | Five original holdouts | 0.924× | 1.000× | 1.092× | 0.975× | 0.910× |
| oneMKL split | Five original holdouts | 0.929× | 1.001× | 1.097× | 0.978× | 0.915× |
| OpenBLAS full | Four new tail holdouts | 1.004× | 1.000× | 0.955× | 0.914× | 0.879× |
| OpenBLAS split | Four new tail holdouts | 0.957× | 0.952× | 0.910× | 0.870× | 0.838× |
| oneMKL full | Four new tail holdouts | 1.003× | 1.000× | 1.082× | 1.035× | 0.997× |
| oneMKL split | Four new tail holdouts | 0.973× | 0.970× | 1.047× | 1.003× | 0.968× |


## Whole oneMKL on every fresh eligible layout

The ratios compare each peer with the public COB full-call oneMKL route. Parentheses show how many of the six paired observations favored the public route. The original 37-row holdout bypassed this route and is excluded from this table.

| M × N × K | Padding | vs released COB | vs stream-B | vs direct oneMKL | vs fastest independent |
|---|---:|---:|---:|---:|---:|
| 61 × 3451 × 191 | 13 | 1.371× (6/6) | 1.174× (6/6) | 1.002× (4/6) | 1.002× (4/6) |
| 89 × 1933 × 257 | 0 | 1.159× (6/6) | 1.122× (6/6) | 1.001× (3/6) | 0.993× (0/6) |
| 108 × 1728 × 320 | 4 | 1.003× (4/6) | 0.967× (0/6) | 1.003× (4/6) | 0.979× (0/6) |
| 137 × 2179 × 347 | 7 | 0.980× (0/6) | 0.928× (0/6) | 1.002× (3/6) | 0.958× (0/6) |
| 65 × 1541 × 227 | 3 | 1.067× (6/6) | 1.056× (6/6) | 1.003× (4/6) | 0.993× (0/6) |
| 82 × 1859 × 313 | 0 | 1.188× (6/6) | 1.151× (6/6) | 1.002× (4/6) | 0.998× (2/6) |
| 99 × 2207 × 145 | 5 | 1.012× (6/6) | 0.943× (0/6) | 1.003× (5/6) | 1.000× (3/6) |
| 116 × 2717 × 291 | 9 | 1.068× (6/6) | 0.999× (3/6) | 1.003× (4/6) | 0.997× (1/6) |

The full oneMKL route led the released standalone implementation in seven of eight medians, and six times in both seed medians. At 137 × 2179 × 347 it lost all six observations against both standalone variants. It lost all six comparisons with stream-B at 108 × 1728 × 320 and 99 × 2207 × 145 as well. At 116 × 2717 × 291 the two were effectively level. The full oneMKL wrapper had a small positive pooled median against matched direct oneMKL for every row, but the sign repeated in both seed medians on only four.


## Four new tail cases

Each row gives the median of six paired ratios for the public split variant. The two comparisons isolate the cost or benefit of computing the remaining rows inside COB.

| Rows × columns × depth | Tail rows | OpenBLAS vs direct | OpenBLAS vs full | oneMKL vs direct | oneMKL vs full |
|---|---:|---:|---:|---:|---:|
| 65 × 1541 × 227 | 1 | 0.995× | 0.984× | 0.997× | 0.996× |
| 82 × 1859 × 313 | 2 | 0.913× | 0.911× | 0.922× | 0.923× |
| 99 × 2207 × 145 | 3 | 1.019× | 1.012× | 1.001× | 0.997× |
| 116 × 2717 × 291 | 4 | 0.905× | 0.907× | 0.972× | 0.967× |

Only OpenBLAS on 99 × 2207 × 145 improved against both matching interfaces in both input seeds. The paired ratios were 1.019× against direct OpenBLAS and 1.012× against the full public call. On that same shape the split achieved just 0.910× the speed of independent oneMKL.

## Verification

The measurement ran inside GitHub Actions on one pinned logical CPU of an Intel Xeon 6975P-C. The 15 participants included released standalone COB, the standalone streamed-B prototype, four vendor-assisted public builds, two matching direct CBLAS calls and seven additional external entries. External providers included OpenBLAS 0.3.34 and oneMKL 2026.2 LP64 sequential. oneDNN’s cached-plan and one-shot entries are identified separately in the raw comparison table.

The independent check validated **1,080 timed observations** (15 participants × 12 shapes × six paired rounds) and **720 full-output accuracy fixtures** comprising **126,912,000 FP64 reference values**. The maximum absolute error was **0.0000243740**, below the 0.002 threshold. Regular, cancellation and scaled inputs passed. Each process reported one thread on the pinned CPU. Separate traced and timed public-entry checks accounted for another **16,921,600 outputs**. All 48 dispatch observations matched the frozen gate. Forty selected the route and eight bypassed it. Ten calls assigned a real tail to COB. Each selected trace recorded one actual CBLAS call with the expected M, N and K. Input and output guards passed.

Both whole oneMKL and its direct comparator resolved to the same sequential LP64 library. The three OpenBLAS variants loaded the same shared object made from the pinned serial static archive. All manifest digests, exported entry identities and loader paths were reconciled against the worker records. The verification package contains the recorded hashes and evidence. From the extracted evidence directory, run `python3 audit.py . verified` to recalculate the comparisons and check the raw observations independently. Re-executing the native experiment requires the matching machine and provider libraries.


The experiment used an experimental source patch on top of b218beff1eaf8fd606d0934405cdc1e28ecdfe15. The CI experiment and per-provider recipes are in the measured revision. The report distinguishes vendor-assisted paths from standalone COB. These are native single-thread SGEMM measurements on the specified host. [Measurement run](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45691645697) · [Frozen experiment](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/tree/b2cdc6a2985726cb1658dc7cc730b5a1c4b99680/tools/world-rank/vendor-wide)
