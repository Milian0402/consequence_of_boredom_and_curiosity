# Guard decision comparison

On Apple M5 Max with Apple Clang 21, the two compiled production predicates agreed on all 235 input pairs. Ubuntu GCC 13.3 on translated Linux x86-64 produced a byte-identical decision CSV. The earlier implementation is commit `85a007747ff54241f47c4754216e0a65748f2847`. The candidate guard source has SHA-256 `ca077ce185e05508eaff83edc68d6fdc7690bfb1328b46abcec4f3366ed4cad7`. All decisions agreed with independently computed or constructed expectations. Twelve public shape boundaries and three oversized, null-input early returns also passed on each compiler.

| New path | Regular HK15 | Classical | Wide |
|---|---:|---:|---:|
| Certificate passed, followed by the existing coherence test | 84 | 16 | 0 |
| Certificate declined, original predicate evaluated | 20 | 56 | 59 |

The test called the ordinary production predicate from a translation unit that includes `gemm.c`. Test-only accessors observed the same static certificate and coherence functions from outside the decision. Inputs included both real-size 1537 × 2306 × 2306 examples, unequal strides, independent starting offsets, exceptional padding, magnitude and support boundaries, all four unrolled B rows and each scalar depth tail. Within the test domain, the new predicate preserved the input bytes.

The sufficient test is conservative. If it certifies an operand using a proposed upper bound U, it has checked that all finite magnitudes are at most U. A witness of at least U/4 in every row and column therefore proves the original fourfold condition. Having at least ceil(3K/4) eligible depths at U/8 proves at least as many at the true maximum divided by eight. The original coherence check then determines the route. If a sample understates the true maximum or overstates the support threshold, the certificate may decline and the exact original predicate decides.

From this directory, reproduce with:

```sh
python3 -B check.py /path/to/candidate receipts/replay
```

The candidate checkout must contain the frozen baseline commit, or pass `--base-repo /path/to/baseline-history`. The runner requires a C compiler, Git and Python 3.10 or newer. `decisions.csv` records the two decisions, the independent expectation, each operand's nominated upper bound, its axis witnesses, and the certificate outcome for every input. `receipt.json` records source hashes, toolchain, field coverage and the CSV digest. This test does no multiplication timing.
