# HK15 input-guard shortcut

An exact shortcut to COB's input checks improved whole-call throughput by **1.5%** on four previously measured shapes and **2.0%** on four fresh shapes on Apple M5 Max. Both results missed the preregistered 3% gain. The experiment stays research-only.

| Single-threaded, warm, whole SGEMM | Previous four shapes | Fresh four shapes |
|---|---:|---:|
| Shortcut relative to the released numerical guard | 1.0151× | 1.0205× |
| Shortcut relative to the original unguarded HK15 | 0.9331× | 0.9380× |
| Two labels for the identical released function | 1.0088× | 0.9992× |

The original-throughput target was 0.95× in each group. The 512 timed calls used the same matrix addresses across methods, checked every output byte against the original and independently checked 96 cells per layout and seed against FP64. The scorer calculates each shape's median paired ratio and the geometric mean across four shapes. [Protocol](timing/protocol.md) · [Raw measurements](timing/times.csv) · [Per-shape ratios](timing/summary.json)

The shortcut attempts to prove the released magnitude and support conditions in one full pass per operand. A successful proof still runs the signed-coherence check. An inconclusive proof executes the original guard. [Derivation](analysis.md) · [Patch against 85a0077](certificate.patch)

**Correctness.** All 235 classifier decisions matched the released implementation and the independent expectations. Apple Clang 21 on M5 and GCC 13.3 on translated Linux x86-64 produced byte-identical decision records. The tests covered underestimated sample bounds, NaNs and infinities, support and magnitude boundaries, signed coherence, cropped inputs and two production-size matrices. All 225,731 values in the public-path numerical checks passed. A separate M5 pthread with exactly **16,384 writable stack bytes**, bounded by two inaccessible guard pages, successfully called public SGEMM 32 times and checked 128 output values and 64 padding values. The classification-only check also confirmed that the shortcut handled both seeds on all four older timing layouts. [Decision records](receipts/decisions.csv) · [Numerical checks](receipts/check-ragged.log) · [Small-stack receipt](receipts/small-stack.csv)

The kernel tested here is FP32, row-major, single-threaded SGEMM. Source revision `85a007747ff54241f47c4754216e0a65748f2847` supplies the released classifier. The candidate changes `src/cob_hk15.inc` only. The M5 native comparisons do not establish a ranking on another host.

## Offline replay

With Git, a C toolchain, CMake and Python 3.12 or newer, pass the path to a local COB repository containing the baseline commit and choose an output directory.

```sh
bash replay.sh /path/to/COB ./replayed
```

The script stages source in a temporary directory, runs the decision and numerical checks, exercises the 16 KiB stack on the recorded M5 configuration, and recomputes the recorded timing summary. On other platforms it uses the local minimum pthread stack rounded up to the page size. To recompute only the timing summary, run `python3 timing/score.py`.
