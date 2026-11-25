# FP32 matrix multiplication using BF16 components

On the Intel Xeon 6975P-C, the fused AMX implementation passed the accuracy checks and ran **2.06× faster** than the six-product oneDNN implementation across eight layouts. It lost all 48 paired warm comparisons against the fastest eligible existing FP32 implementation and all 48 against standalone COB.

At 4096 × 4096, existing COB's two-level aligned Winograd with oneMKL leaves took **590.43 ms**. The fused AMX method took **948.38 ms**, and the six-product oneDNN method took **1,474.39 ms**. A hardware trace of the fused call recorded 50,331,648 BF16 dot-product instructions on one thread. Rounding both inputs to a single BF16 component took **318.81 ms**. Every checked output at 4096³ passed the componentwise γₖ bound relative to the sum of absolute products. The maximum absolute error against FP64 was **0.249**, so the method failed the separate **0.002 absolute-error ceiling for the bounded timing fixtures**. That ceiling is specific to their input scale. The raw method also failed the separate direct-FP32 accuracy checks on adversarial inputs.

On Arm Neoverse-N2, the six-product implementation used the Arm Compute Library's Advanced SIMD BFMMLA kernel. It passed on all eight layouts. At 4096 × 4096 it took **5,092.24 ms** against **2,166.11 ms** for FastMMW with its 1536 cutoff. Rounded single-product BF16 took **905.03 ms** at reduced accuracy. Sequential oneDNN selected its reference BF16 implementation on Arm, which excluded that path from the hardware comparison.

| Pinned host | Accuracy-qualified mixed methods | Paired warm wins versus fastest qualifying FP32 | Geometric mean of paired speed ratios |
| --- | --- | ---: | --- |
| Intel Xeon 6975P-C | oneDNN six, grouped six, nine, custom fused AMX | 0/192 | 0.329×, 0.242×, 0.261×, 0.675× |
| Arm Neoverse-N2 | ACL six, grouped six, nine | 0/144 | 0.410×, 0.388×, 0.295× |

A speed ratio above one favors the mixed method. The accuracy-qualified methods reached the registered 1.5× and 2× thresholds on zero layouts for first and warm calls. Existing standalone COB led the N2 field at 1024³ with **40.34 ms** against **41.85 ms** for the fastest independent comparator, FastMMW, winning six of six pairs. Existing standalone COB also led on the 1793 × 1151 × 2779 cropped case. BLASFEO's worker failed on that layout and on the 1537 × 2053 × 1789 cropped rectangle on both hosts; each unavailable measurement is recorded in the CSV. Two further N2 rectangles favored COB's existing vendor-assisted OpenBLAS configuration. The eight shapes on each of these two machines establish a bounded comparison.

Each host used one pinned CPU thread. Each method that qualified for timing had four input checks and six paired warm observations per eligible layout. The timer covered conversion, packing, workspace allocation, multiplication and final FP32 reconstruction. First-call measurements have their own columns. The cached oneDNN FP32 plan is identified separately. Every tested output was checked against signed FP64 reference sums and the sum of absolute products. Three-product and raw BF16 methods remain reduced-precision comparisons. The registered six-product method uses only six of the nine possible cross-products and makes no universal correct-rounding guarantee.

The supplemental long-reduction check compared actual COB and the Arm BF16 kernel on local Apple hardware with an exact integer oracle. With K = 4096 and 8191, all methods passed the componentwise γₖ test on this small-output panel. Maximum and RMS comparisons against direct FP32 rejected raw BF16 and three-product arithmetic on all eight cases. Six products, grouped six, and nine passed on all eight. The archive includes the complete output words, native route receipts and earlier scalar and emulated checks.

[Every method and layout, including exclusions](outcomes.csv) · [Full offline evidence and verification source](observations.zip) · [Frozen benchmark source](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/tree/a23ab20dae16111c0604f92864bf6d072d6752f9/tools/world-rank/precision) · [Original hosted jobs](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45717780617).

Run the following from the repository root with Python 3.11 or later to verify both hosted rankings and both numerical supplements offline.

```sh
python3 -B docs/audits/2026-11-25-ranking/bf16-components/verify.py
```
