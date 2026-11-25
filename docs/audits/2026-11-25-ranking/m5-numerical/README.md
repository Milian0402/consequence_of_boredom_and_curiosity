# Accuracy and speed of an experimental M5 Max FP32 kernel

**The candidate corrected the known numerical failure and missed its single-thread speed target.** On the previously losing 49 × 35 × 39 case, its warm public call took 332.1 ns, versus 363.5 ns for the released COB route and 346.0 ns for cached LIBXSMM 2.1.0. With four rotating operand sets, a direct call to a prepared LIBXSMM kernel took 333.4 ns versus 336.7 ns for COB. The paired median speedup was 0.980×. Across eight fresh shapes, four beat both independent comparisons. The promotion rule required six and a win on the original case. The patch was kept experimental.

The candidate adds a four-panel path for 49–64 rows, 17–48 columns and depth 16–48, excluding fully 16-lane-aligned shapes. It checks the magnitude of each active FP32 input as it feeds the SME kernel. If any magnitude reaches 2, it recomputes the full product with FP64 accumulators and rounds to FP32. The cost is visible when only one input crosses the threshold. For 49 × 35 × 39 with one value set to 64, four-buffer calls took 12.567 µs for the candidate and 0.374 µs for the released COB route. [Input families and timings](hazard-summary.csv)

## Reproduce the accuracy fix

On an SME2-capable Apple Silicon Mac with CMake, Git and Apple Clang, run the following from the directory containing this README in a checkout containing commit `b218beff1eaf8fd606d0934405cdc1e28ecdfe15`.

```sh
bash reproduce.sh "$(git rev-parse --show-toplevel)"
```

The script extracts that commit into a temporary directory and applies the included patch to a separate copy. The fixture calls both public entry points for a 64 × 25 × 29 product with both inputs sharing storage. Both input strides are 32, the output stride is 31 and the seed is 40096813. The original maximum error against FP64 was 0.00283442336513, above the 0.002 bound. The candidate reduced it to 0.00088129836513, with all 1,600 output values equal to the nearest FP32 reference. The script exits **0** when it reproduces both outcomes. [Fixture measurements](scale-control.csv) · [Experimental patch](candidate.patch)

An independent 350-case numerical suite covered 647,130 output cells. For 87,449 cells, the nearest FP32 value was itself more than 0.002 from the FP64 reference. The candidate returned that nearest value for all of them. Every remaining cell was within 0.002. Repository tests passed 4/4, and all 84 protected-memory cases passed. [Numerical criteria](accuracy-protocol.md) · [Raw numerical evidence](accuracy.csv) · [Numerical summaries](accuracy-summary.csv) · [Magnitude-check routing](wide-route.csv)

## Verify the recorded ranking

From this directory, run `python3 performance-verdict.py`. The expected exit code is **1**, reflecting the failed speed gate. The eight fresh shapes had geometric-mean speedups of 1.175× over the incumbent, 1.049× over the fastest independent public entry and 1.052× over direct prepared LIBXSMM with four rotating input sets. The public-field geometric mean also missed its 1.05× threshold. All raw observations were retained. [Frozen protocol](protocol.md) · [Shape panel](panel.csv) · [Four-design selection screen](screen-summary.csv)

| Comparison | Summary | Raw observations |
|---|---|---|
| Public entries, whole call | [Public results](summary.csv) | [1,240 observations](timings.csv) |
| Prepared LIBXSMM, one and four rotating sets | [Prepared results](prepared-summary.csv) | [1,600 observations](prepared.csv) |
| High-amplitude inputs, including recomputation | [High-amplitude results](hazard-summary.csv) | [300 observations](hazard.csv) |

The ordinary public field included Accelerate, OpenBLAS, ArmPL and LIBXSMM with and without last-kernel caching. Native MaxMulSK and MpGEMM also ran on compact inputs. Each ordinary observation timed 32,768 consecutive calls. The high-amplitude study used 8,192 calls per observation. Each comparator's thread probe recorded one thread before and after execution. Raw times are seconds and the summaries use nanoseconds and paired wall-time ratios. [Thread checks](threads.csv)
