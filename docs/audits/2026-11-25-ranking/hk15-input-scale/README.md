# Accuracy of the 15-product kernel on scaled and complementary inputs

The opt-in ragged Hopcroft–Kerr kernel at f322a9a exceeded the benchmark's 0.002 absolute error threshold on ten of 22 input families on each of two full-size M5 Max layouts. Every dot product in this test has an exactly representable FP32 result. Sequential OpenBLAS 0.3.34 returned every tested output exactly. The earlier ranking timed different input distributions.

| Shape, M × N × K | Outputs checked per implementation | Families over 0.002 with issued COB | Nonfinite issued outputs despite a finite exact result |
|---|---:|---:|---:|
| 1931 × 2903 × 2881 | 123,325,246 | 10/22 | 3,732,620 |
| 2117 × 3071 × 3013 | 143,028,754 | 10/22 | 4,329,336 |

At 1931 × 2903 × 2881, give the first half of A's rows and the first third of B's columns the value 1/256. Fill the remaining entries with one. At output (0,0) the exact result is 2881/65536 = 0.0439605712890625. COB returned 0.0197906494140625, an error of about 55%. With the smaller entries set to 1/65536, it returned a negative number for 933,155 positive outputs. A separate case used A = 2¹²⁷ and B = 2⁻¹²⁷ throughout. Every exact output was 2881; millions of outputs were nonfinite.

A second family places most mass in different reduction coordinates for different rows and columns. All input rows and columns can have the same maximum magnitude. At 1931 × 2903 × 2881 the interleaved complement case at shift 16 produced a maximum relative error of 72.4%.

Both shapes were cropped, padded matrices. The checker inspects every logical output, verifies the input data and all per-row output guards, and computes its oracle from powers of two and integer counts. It rejects any fixture whose expected entry cannot be represented exactly in FP32. The elapsed time in these CSVs is diagnostic; this study ranks numerical accuracy. The paper on diagonal scaling describes why operand scale matters for fast multiplication. [Ballard et al.](https://doi.org/10.1137/15M1032168)

## Reproduce

From a checkout at the tested source revision, configure CMake Release with `COB_ENABLE_HK15_RAGGED=ON`, `COB_ENABLE_PORTABLE_STRASSEN=ON` and `COB_ENABLE_RECURSIVE_STRASSEN=ON`. Compile `check.c` with the installed COB include path and library, then run `./check 1931 2903 2881 issued` and `./check 2117 3071 3013 issued`. For the control, define `WITH_OPENBLAS`, add the OpenBLAS headers, link its library, and set both `OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1`. Run `python3 -B replay.py` here to validate the recorded tables and recompute the counts. [Tested source](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/tree/f322a9ab58454762563b97323fdc5f29016cc39f)
