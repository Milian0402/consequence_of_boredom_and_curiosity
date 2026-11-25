# Independent single-thread FP32 CPU GEMM ranking

Compare one-shot row-major C = A × B on one CPU thread, using identical FP32 inputs and row strides, including all per-call setup, packing, workspace and recombination. Start from COB 6dc1265 (complete local source) and 828109b (native Linux build). The result sought is leadership against independent implementations. Speedups over earlier COB and any hybrid with a vendor leaf are reported separately.

## Linux input sets

Dimensions are M × N × K. The suffix /p adds p elements to each row stride. Freeze these sets before viewing any new vendor ranking. A balanced grid here gives a portfolio result for the listed shapes, not a mathematical proof over all positive integer dimensions.

Discovery: 64³, 192×512×384, 5×3073×1537/5, 64×3072×256, 1024³, 3072³, 2560×4096×2560, 2305×3073×2113/7.

Confirmation: 48³, 96×256×512, 97×641×193/5, 2×3073×1703/3, 11×513×385, 128×4096×256, 1024×128×2048, 2048³, 4608³, 3072×4480×2688/7, 3713×2049×3329. Confirmation uses two new seeds, three counterbalanced timing rounds per seed, the complete one-shot API and an independent FP64 product on every logical output. Accuracy requires finite differences with maximum 0.002 on the bounded workload and intact padding. Thread/affinity and actual host model must be recorded.

## Ranking decision

Freeze one COB configuration and any dispatch rule before opening confirmation results. Distinguish native standalone and any externally linked variants. A per-shape oracle that takes the fastest candidate after the run is not a deployable multiplier. Compare each frozen candidate on each shape with the fastest independent, numerically valid single-thread implementation measured there. Use medians of paired ratios and equal-weight geometric means. Show the raw per-shape times and near-ties. Treat margins close to noise as inconclusive. Keep adverse results, untested competitors and failed builds visible. If a documented rival repeatedly wins a valid input, a claim that COB wins everywhere fails. To assess prospective leadership over a defined portfolio, replicate on independent native hosts and give the exact input and competitor scope.

The additional short-N and short-K discovery controls are 1536×1×1025/3 and 257×1025×4/5. Their independent confirmation cases are 2049×3×1023/7 and 513×769×7/3. LIBXSMM is an independent eligible peer for M×N×K ≤ 262144. Warm timings include every shape lookup. First SGEMM latency additionally includes the first JIT compilation, after library initialization.
