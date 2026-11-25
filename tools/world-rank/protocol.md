# Independent single-thread FP32 CPU GEMM ranking

Compare one-shot row-major C = A × B on one CPU thread, using identical FP32 inputs and row strides, including all per-call setup, packing, workspace and recombination. Start from COB 6dc1265 (complete local source) and 828109b (native Linux build). The result sought is leadership against independent implementations. Speedups over earlier COB and any hybrid with a vendor leaf are reported separately.

## Linux input sets

Dimensions are M × N × K. The suffix /p adds p elements to each row stride. Freeze these sets before viewing any new vendor ranking. A balanced grid here gives a portfolio result for the listed shapes, not a mathematical proof over all positive integer dimensions.

Discovery: 64³, 192×512×384, 5×3073×1537/5, 64×3072×256, 1024³, 3072³, 2560×4096×2560, 2305×3073×2113/7.

Confirmation: 48³, 96×256×512, 97×641×193/5, 2×3073×1703/3, 11×513×385, 128×4096×256, 1024×128×2048, 2048³, 4608³, 3072×4480×2688/7, 3713×2049×3329. Confirmation uses two new seeds, three counterbalanced timing rounds per seed, the complete one-shot API and an independent FP64 product on every logical output. Accuracy requires finite differences with maximum 0.002 on the bounded workload and intact padding. Thread/affinity and actual host model must be recorded.

## Ranking decision

Freeze one COB configuration and any dispatch rule before opening confirmation results. Distinguish native standalone and any externally linked variants. A per-shape oracle that takes the fastest candidate after the run is not a deployable multiplier. Compare each frozen candidate on each shape with the fastest independent, numerically valid single-thread implementation measured there. Use medians of paired ratios and equal-weight geometric means. Show the raw per-shape times and near-ties. Treat margins close to noise as inconclusive. Keep adverse results, untested competitors and failed builds visible. If a documented rival repeatedly wins a valid input, a claim that COB wins everywhere fails. To assess prospective leadership over a defined portfolio, replicate on independent native hosts and give the exact input and competitor scope.

The additional short-N and short-K discovery controls are 1536×1×1025/3 and 257×1025×4/5. Their independent confirmation cases are 2049×3×1023/7 and 513×769×7/3. LIBXSMM is an independent eligible peer for M×N×K ≤ 262144. Warm timings include every shape lookup. First SGEMM latency additionally includes the first JIT compilation, after library initialization.

## Frozen Linux confirmation configuration

Keep the prior standalone path for matrices with all three dimensions at most 128. The four-row no-pack implementation lost on both native discovery CPUs. Use the new blocked classical kernel on eligible larger matrices, short-N and short-K kernels on their supported shapes, and fused one-level Strassen from harmonic dimension 1024. Compact square matrices eligible for the existing two-level route retain that route. Both discovery CPUs favored two levels at 3072. The confirmation policy uses matrix geometry only.

The small kernel challenger is scored independently on its own nine-layout panel. It has two separately declared seeds, direct-only peers and a wider explicit LIBXSMM JIT range. Main-panel results keep the previously declared LIBXSMM size range. The one-thread whole-call timing contract applies throughout.

## Follow-up replication

On the 13 previously confirmed layouts, compare the frozen 3cd1efc standalone entry to the revised candidate and to the independent libraries on new seeds 59147207 and 81729353. Seven additional layouts test the changed entry for small inputs and for nine to fifteen output rows: 40³, 72³, 29×61×43/3, 57×85×37/5, 9×397×293/1, 13×743×479/5, 15×953×613/3. Distinguish the repeated cases from these seven in the ranking. Retain the original 13-case confirmation scores and CPU model. Use pure standalone candidates here; initial hybrid results remain available in the first confirmation artifacts.

## Short-row dispatch verification

At 13 and 15 output rows a split NEON row tile regressed against the packed incumbent. The final Arm dispatch caps the new path at one 12-row tile. The x86 panel stays enabled through 15 rows after all four tested heights beat the incumbent on AMD. Test all six registered 9–15-row layouts in short.csv with new inputs and both timing orders against current independent libraries and the original frozen entry. Keep this follow-up separate from the preceding 20-shape replication, which used the full 15-row range on both CPUs.

## Medium rectangular and small Arm experiments

The medium pilot compares changing the depth and lifetime of packed panels on both architectures. On x86 it also tests wider register tiles and a kernel that packs only B. The Arm small pilot compares a native 6-by-16 NEON kernel with COB and a warmed LIBXSMM lookup on the entire small panel. Capture independent BLAS results on each case and keep the full six timings and FP64 checks. The incumbent is the distributed configuration at 22bbdbd. These are one-shot calls with reusable libraries already initialized.

Freeze the pilot and held-out dimensions together. Select at most one policy per architecture and workload family from the pilot, with the incumbent as the default. Favor reproducible improvements across related dimensions and inspect regressions. Confirm the selected policy against the incumbent and strongest eligible independent competitor on the untouched dimensions with fresh seeds. Report losses and actual runner models, including any change between phases. Report vendor-backed recursion separately from standalone COB.
