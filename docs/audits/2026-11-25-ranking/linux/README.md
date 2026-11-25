# Single-threaded FP32 matrix multiplication on Linux

On Arm Neoverse N2, COB’s two-level Winograd release with aligned OpenBLAS leaves beat the strongest independent competitor in all six trials on each of two preregistered odd shapes, by **5.1% and 3.8%** in paired speedup. On Intel Xeon 6975P-C, the equivalent oneMKL-backed release lost to direct oneMKL on both shapes. The performance claim is limited to the tested CPU models, library builds and matrix layouts.

| CPU | Layout, M × N × K | Fixed COB, ms | Fastest independent, ms | Paired speedup | Wins / 6 | Fastest one-shot entry |
| --- | --- | ---: | --- | ---: | ---: | --- |
| Neoverse N2 | Repeat · 1879 × 3437 × 3041 | 639.83 | FastMMW 663.46 | 1.0361× | 6 | Fixed COB release |
| Neoverse N2 | Deep · 2039 × 2969 × 3367 | 666.75 | FastMMW 700.01 | 1.0514× | 6 | Fixed COB release |
| Neoverse N2 | Wide · 1973 × 3319 × 2897 | 621.91 | FastMMW 647.45 | 1.0384× | 6 | Fixed COB release |
| Neoverse N2 | Control · 2624 × 2752 × 2688 | 649.23 | FastMMW 656.41 | 1.0109× | 5 | Earlier COB prototype with aligned ArmPL leaves |
| Xeon 6975P-C | Repeat · 1879 × 3437 × 3041 | 219.93 | oneMKL JIT 211.60 | 0.9590× | 0 | oneMKL JIT |
| Xeon 6975P-C | Deep · 2039 × 2969 × 3367 | 233.14 | oneMKL 221.63 | 0.9545× | 0 | oneMKL |
| Xeon 6975P-C | Wide · 1973 × 3319 × 2897 | 209.39 | oneMKL 208.76 | 0.9957× | 2 | Earlier COB prototype with aligned oneMKL leaves |
| Xeon 6975P-C | Control · 2624 × 2752 × 2688 | 210.54 | oneMKL 206.52 | 0.9855× | 2 | Earlier COB prototype with ordinary two-level oneMKL leaves |

The fixed provider was chosen on three earlier training shapes for each of AMD EPYC and Neoverse N2. Intel used the AMD choice as a preregistered extrapolation. The decision rule required a median paired speedup above 1 and at least five wins in six trials on **each** of the deep and wide shapes. The repeat and aligned control had separate scores. Independent competitors included direct BLAS, FastMMW with OpenBLAS leaves, strict one-shot oneDNN and, on Intel, two qualified TBLIS Strassen variants. The independently compiled standalone COB implementation also competed. It finished behind the fastest independent on all eight CPU/layout combinations. The oneDNN cached-plan measurements reuse a shape-specific primitive and are reported separately from the one-shot ranking. [Registered protocol](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/blob/7885da100a88fbc3a6f0ef11f486fece4eabee76/tools/world-rank/release-replay/protocol.md) · [Training run](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45672997916)

## Measurements and accuracy

Each worker performed a row-major FP32 product with disjoint operands, alpha 1 and beta 0, using one process thread pinned to one Linux CPU. Row padding in floats was 3 for the repeat, 11 for deep, 7 for wide and zero for the control. The timed release executable called `cob_sgemm_rowmajor` from the ordinary CMake-built library. The call included allocation, leaf multiplication, remainder handling and recombination. Six complete calls were timed after initialization, three per input seed, in counterbalanced order. Times in the table are medians. Speedup is the median of the six ratios of independent time to fixed COB time. The first call on each validation fixture is recorded separately.

The corrected run admitted 23 methods on Intel and 19 on Arm. Every admitted method passed four full-output fixtures per layout against FP64 (two random inputs, a scaled input and a cancellation input). The checks covered 4,415,300,232 output cells, operand bytes and output padding. Maximum observed absolute error was 0.00134679333777 on Intel and 0.00134334827844 on Arm, below the registered 0.002 limit. Results concern these four layouts, two CPU models and warm calls. [Intel job](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45682774110/job/139447504327) · [Arm job](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45682774110/job/139447504055)

## Which source ran

The benchmark revision is `7885da100a88fbc3a6f0ef11f486fece4eabee76`. The current timed production source is `518e7d77ea3d82d029f31dc10a3a705b42a7638b`. The previous-release control uses `3222dddb42944630f45c6864150cfaff54df257a`. Both are bound to Git blob manifests in the benchmark source. [Benchmark source and replay inputs](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/tree/7885da100a88fbc3a6f0ef11f486fece4eabee76/tools/world-rank/release-replay)

The first execution completed on Arm. Intel stopped before any ordinary release worker was measured because two tests expected the public dispatcher to choose Strassen on tiny matrices handled by an x86 small-matrix kernel. Fixture revision `b218beff1eaf8fd606d0934405cdc1e28ecdfe15` changed those two tests. The corrected execution reused the original shapes and seeds. All ten production library archives (five per CPU) were byte-identical before and after a clean rebuild with corrected fixtures. Each Make build was separately validated. [First execution](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45679590298) · [Correction and original artifact identifiers](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/blob/7885da100a88fbc3a6f0ef11f486fece4eabee76/tools/world-rank/release-replay/fixture-correction.md) · [Corrected execution](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45682774110)

## Recompute the rankings

Run `python3 verify.py` from this directory. The verifier uses the Python standard library. It checks the published files against `SHA256SUMS`, rebuilds the six paired comparisons from individual calls, checks the numerical fixtures and route counters, and compares the ten before-and-after library hashes. Reproducing the native elapsed times requires a matching machine and the benchmark workflow linked above.

| Files | Contents |
| --- | --- |
| `summary.csv`, `panel.csv`, `provenance.json` | Eight rankings, exact layouts, Git revisions, public artifact digests, observed oneDNN/TBLIS kernels and the fixed provider for each CPU |
| `timings.csv`, `accuracy.csv` | 1,008 warm calls with pair identifiers and elapsed seconds, plus 672 full-output fixture results with errors, input and guard checks, and first-call seconds |
| `methods.csv`, `coverage.csv` | Implementation roles and leaf providers, native inclusion and exclusions, and qualification for each method and layout |
| `builds.csv`, `release-routes.csv` | Public worker, Make object and CMake archive hashes, plus ordinary worker and instrumented route checks |
| `prototype-routes.csv`, `fastmmw-routes.csv`, `standalone-routes.csv`, `hybrid-leaves.csv` | Prototype, external and historical comparator route checks |
