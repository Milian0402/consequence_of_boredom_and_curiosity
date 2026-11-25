# Single-thread x86 medium-wide FP32 multiplication with matched BLAS providers

COB's own last-row kernel did not deliver a repeatable advantage over a single matched BLAS call. On four dedicated row-tail shapes, splitting the call achieved **0.952×** full OpenBLAS on Intel and **0.986×** on AMD. With oneMKL these figures were **0.970×** and **1.004×**. No tail shape beat both the matched direct and whole-public calls on all four host-and-seed combinations for either provider. These findings support keeping the current release and calling serial CBLAS directly when an application already selects that provider.

Each cell below gives the geometric-mean speedup in **attempt 1 / attempt 2**. A value above one favors the named COB variant. Each within-attempt ratio is a median of six matched timings. The whole-call path sends all eligible rows to the provider. The split path sends one to four final rows through COB's AVX2/FMA kernel when applicable. The matched direct comparator invokes CBLAS from the same library. Standalone COB and stream-B use their own kernels.

| Candidate | Independently frozen cohort | vs matched direct | vs full public call | vs standalone release | vs stream-B | vs fastest independent |
|---|---|---:|---:|---:|---:|---:|
| OpenBLAS full | Five original holdouts | 0.987× / 0.972× | — | 0.999× / 0.998× | 0.890× / 0.891× | 0.831× / 0.850× |
| OpenBLAS split | Five original holdouts | 0.987× / 0.973× | 0.999× / 1.004× | 0.998× / 0.999× | 0.889× / 0.896× | 0.831× / 0.854× |
| oneMKL full | Five original holdouts | 0.924× / 1.111× | — | 1.092× / 0.547× | 0.975× / 0.488× | 0.910× / 0.466× |
| oneMKL split | Five original holdouts | 0.929× / 1.110× | 1.001× / 1.000× | 1.097× / 0.547× | 0.978× / 0.488× | 0.915× / 0.465× |
| OpenBLAS full | Four new tail holdouts | 1.004× / 1.005× | — | 0.955× / 0.944× | 0.914× / 0.920× | 0.879× / 0.861× |
| OpenBLAS split | Four new tail holdouts | 0.957× / 0.991× | 0.952× / 0.986× | 0.910× / 0.934× | 0.870× / 0.908× | 0.838× / 0.849× |
| oneMKL full | Four new tail holdouts | 1.003× / 1.006× | — | 1.082× / 0.458× | 1.035× / 0.446× | 0.997× / 0.418× |
| oneMKL split | Four new tail holdouts | 0.973× / 1.008× | 0.970× / 1.004× | 1.047× / 0.461× | 1.003× / 0.447× | 0.968× / 0.420× |

## Whole oneMKL on the eight eligible holdout layouts

The original 37-row holdout bypasses the vendor path because eligibility begins at 48 rows. The eight eligible layouts comprise four shared with the primary panel and four additional tail shapes. Each appears once in this table. Padding adds the stated number of floats to the row strides of A, B and C.

| M × N × K | Pad | vs released COB | vs stream-B | vs direct oneMKL | vs fastest independent |
|---|---:|---:|---:|---:|---:|
| 61 × 3451 × 191 | 13 | 1.371× / 0.519× | 1.174× / 0.430× | 1.002× / 0.996× | 1.002× / 0.418× |
| 89 × 1933 × 257 | 0 | 1.159× / 0.460× | 1.122× / 0.442× | 1.001× / 1.009× | 0.993× / 0.417× |
| 108 × 1728 × 320 | 4 | 1.003× / 0.459× | 0.967× / 0.439× | 1.003× / 0.992× | 0.979× / 0.417× |
| 137 × 2179 × 347 | 7 | 0.980× / 0.447× | 0.928× / 0.434× | 1.002× / 0.998× | 0.958× / 0.420× |
| 65 × 1541 × 227 | 3 | 1.067× / 0.461× | 1.056× / 0.454× | 1.003× / 0.996× | 0.993× / 0.421× |
| 82 × 1859 × 313 | 0 | 1.188× / 0.469× | 1.151× / 0.461× | 1.002× / 1.020× | 0.998× / 0.416× |
| 99 × 2207 × 145 | 5 | 1.012× / 0.446× | 0.943× / 0.435× | 1.003× / 0.995× | 1.000× / 0.416× |
| 116 × 2717 × 291 | 9 | 1.068× / 0.456× | 0.999× / 0.434× | 1.003× / 1.012× | 0.997× / 0.419× |

Across those eight layouts, whole oneMKL averaged **1.100× released COB on Intel and 0.464× on AMD**, **1.039× / 0.441×** against stream-B, **1.002× / 1.002×** against the matched direct oneMKL call, and **0.990× / 0.418×** against the fastest independent method per shape. These are geometric means of per-shape paired speedups.

On Intel, whole oneMKL beat released COB on seven layouts, with 137 × 2179 × 347 the exception. It beat stream-B on four layouts and trailed on 108 × 1728 × 320, 137 × 2179 × 347, 99 × 2207 × 145 and 116 × 2717 × 291. The margin on the last shape was 0.05%. On AMD, released COB and stream-B each beat whole oneMKL on all eight layouts. AOCL and cached-plan oneDNN each led the independent AMD field on four layouts.

## Four fresh shapes with actual COB tails

These shapes isolate the tail treatment. The absolute row count sent to the matched vendor and the rows left to COB were observed in separate traced public builds on both attempts.

| M × N × K | COB tail rows | OpenBLAS split vs direct | OpenBLAS split vs full | oneMKL split vs direct | oneMKL split vs full |
|---|---:|---:|---:|---:|---:|
| 65 × 1541 × 227 | 1 | 0.995× / 0.977× | 0.984× / 0.991× | 0.997× / 0.993× | 0.996× / 0.998× |
| 82 × 1859 × 313 | 2 | 0.913× / 0.974× | 0.911× / 0.970× | 0.922× / 1.013× | 0.923× / 1.002× |
| 99 × 2207 × 145 | 3 | 1.019× / 1.013× | 1.012× / 1.013× | 1.001× / 1.024× | 0.997× / 1.019× |
| 116 × 2717 × 291 | 4 | 0.905× / 0.999× | 0.907× / 0.972× | 0.972× / 1.003× | 0.967× / 0.996× |

The same 12 layouts, two seeds and 15 methods ran on a single pinned logical CPU on an Intel Xeon 6975P-C in attempt 1 and an AMD EPYC 9V45 in attempt 2. The two completed native runs yielded **2,160 timing observations**, **1,440 full-output FP64 checks covering 253,824,000 cells**, and 96 public route checks covering a further 33,843,200 timed-build and trace-build output cells. The largest absolute error was 0.0000243741 in both attempts. The matching oneMKL binaries had identical hashes across machines. The native OpenBLAS builds had different hashes across machines and each run verified the same provider within its own direct and public calls. The field includes cached and JIT peers where labeled. No cross-host timing ratios enter the tables. These results establish the behavior of these host and workload combinations only.

The research patch applies to b218beff1eaf8fd606d0934405cdc1e28ecdfe15. To reproduce the numerical ranking from the evidence directory, run `python3 audit.py attempt-1 verified-1` and `python3 audit.py attempt-2 verified-2`. Use `python3 verify-repeat.py .` to verify all within-host ratios and both sets of cohort geometric means directly from the raw tables. The recorded library checksums and runtime messages identify the providers. Re-running the native timings requires the corresponding host and SDKs. [First native attempt](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45691645697/attempts/1) · [Second native attempt](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45691645697/attempts/2) · [Frozen workflow source](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/tree/b2cdc6a2985726cb1658dc7cc730b5a1c4b99680/tools/world-rank/vendor-wide)
