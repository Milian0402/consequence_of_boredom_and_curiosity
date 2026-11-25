# Signed SGEMM updates in two-level Winograd

This optional research patch changes two terminal products in the rectangular, two-level Winograd path that uses serial CBLAS leaves. It passed local correctness tests with serial OpenBLAS on Linux Arm64. On the two fresh Intel layouts, the signed oneMKL build gained 0.44% and 0.48% over the original, below the required 2% on each. [Native timing and accuracy results](results-linux/README.md)

The release computes P3 in a temporary and adds it to C12, then computes P4 in the same temporary and subtracts it from C21. At the point each product is issued, its destination already contains the partial sum. The patch asks SGEMM to compute `C12 ← (+1) S·B22 + C12` and `C21 ← (−1) A22·T + C21` directly. Each of the seven child products at the outer level uses both updates, removing fourteen terminal merge passes. The temporary still serves P7, which contributes to two outputs. An SGEMM accumulation may round differently or choose a different kernel.

Apply the patch to `b218beff1eaf8fd606d0934405cdc1e28ecdfe15` and compile with `-DCOB_ENABLE_SIGNED_WINOGRAD_LEAVES=1` to activate it. The default retains the earlier schedule. The standalone path, square recursion, outer-level merges and tail partitioning retain their original arithmetic. Each terminal update checks the current provider's serial status and uses the original overwrite/merge route if that check fails.

## Reproduce the correctness checks

On Linux Arm64 or x86-64, use Git, tar, patch, a native C11 compiler, and OpenBLAS development headers and libraries with `cblas.h` and `-lopenblas` on the compiler's default search paths. The recorded run used Ubuntu 24.04 Linux Arm64 with GCC and its serial OpenBLAS 0.3.26 package. From the research directory, run:

```sh
sh check.sh "$CHECKOUT"
```

`CHECKOUT` is a local Git repository containing the exact commit above. The script extracts that revision into a unique temporary directory, applies the patch there, and builds with the experimental flag off and on. It also compiles the production translation unit separately and invokes its ordinary public API for a ragged 2049 × 2051 × 2053 case. The script prints results directly; no checkout edits or network access are necessary.

| Local validation | Result per configuration |
| --- | --- |
| 16 dense FP64 fixtures | Eight combinations of bottom, right and reduction tails, each with random and cancellation inputs |
| 2 serial-status transition fixtures | The original fallback completed the output when the simulated provider status changed immediately before P3 or P4 |
| Public dispatch with tracing | 4,202,499 exact output values for 2049 × 2051 × 2053, with 49 core leaf calls plus three tails |
| Public dispatch through a separately linked production object | Same geometry, a different exact FP64 fixture with three nonzero reduction positions, 4,202,499 exact outputs |
| Existing release tests | FP64, depth-edge, stride, alias and allocation checks passed |

Across the 19 traced fixtures, each configuration checked 4,648,073 output cells. The maximum absolute error was 0.0000330852344632. Output guards and the complete input buffers were checked. With the flag on and a serial provider throughout, the trace found exactly seven positive and seven negative terminal updates at the previously initialized quadrant addresses and strides. The baseline made zero signed terminal updates. `validation.csv` gives every traced result, and `validation.txt` records the actual provider and the separately linked checks.

## Frozen performance follow-up

`panel.csv` and `protocol.md` specify five layouts, two seeds, six paired timings, whole-call measurement, numerical admission and a fixed success criterion. The Intel comparison included the original and patched public release and direct sequential oneMKL. oneMKL JIT is a separate comparator when qualified. The recorded measurements, qualification decisions and standalone verifier are in [results-linux](results-linux/README.md).
