# Exact signed-byte matrix multiplication on Apple M5 Max

On the original 32×32 UCI handwriting scans, the checked multiplier took **2.862 ms**. It beat standalone COB in 23 of 24 paired rounds and single-thread Accelerate in all 24. On the reduced 8×8 representation, standalone COB won all 24 rounds. The same 1,797 test examples appear in both comparisons. The original archive supplies different training-set sizes for these two representations. [UCI Optical Digits, E. Alpaydin and C. Kaynak](https://doi.org/10.24432/C50P49)

| Fresh-process confirmation | M × N × K | Checked | Standalone COB | Accelerate | Binary pack + popcount |
|---|---|---:|---:|---:|---:|
| Original binary images | 1,797 × 1,934 × 1,024 | 2.862 ms | 4.024 ms | 4.736 ms | 12.153 ms |
| Reduced feature counts | 1,797 × 3,823 × 64 | 1.045 ms | 0.660 ms | 1.277 ms | Ineligible |

Times are medians of complete calls. For each round, the paired speedup divides the control time by the checked time. On the original images those median ratios were **1.399×** against COB, **1.634×** against Accelerate and **4.154×** against this pack-and-popcount implementation. On the reduced features they were **0.632×** against COB and **1.232×** against Accelerate. Each method received FP32 inputs and returned the full row-major FP32 output. The custom methods allocated and packed their inputs within each call. [All 168 confirmation calls](confirm-results/timings.csv) · [Paired medians and wins](confirm-results/summary.json)

The preceding pilot used nine paired rounds per reduced or synthetic workload and twelve per original-image workload. The checked path beat COB by 1.205× on a synthetic signed 1,024³ product. On ordinary synthetic FP32 at the reduced-data geometry, the input check selected Accelerate. The wrapper ran at 0.940× the speed of calling Accelerate directly. The pilot ratios against COB on original and reduced handwriting were 1.482× and 0.658× respectively. [All 129 pilot calls](results/timings.csv) · [Pilot accuracy against FP64](results/accuracy.csv)

## What qualified the product

The experimental entry point `apple_exact` admits live FP32 values that are exactly integers in [−63,63] and depths through 4,096. It checks both operands while packing signed bytes. The Arm SME `SMOPA` instruction accumulates signed products into INT32. For every output, the sum of absolute product magnitudes is bounded by 4,096 × 63² = **16,257,024**. All integer partial sums therefore fit in INT32 and exactly in FP32. The guarantee covers numerical equality, including positive and negative zero. Unqualified data, a missing hardware feature or a failed scratch allocation selects ordinary Accelerate. Invalid strides or overlapping output have a separate error return. [Source and exact ABI](exact.c) · [Observed instructions in the measured binary](instructions.txt)

The pilot compared the full outputs of its integral workloads with a single-thread OpenBLAS FP64 reference. It also checked three scalar dots per workload. The confirmation verified **every timed output** on both real workloads against an independent full-matrix fingerprint, checked guards after each call and rechecked both inputs in all five storage sets. Each round used fresh output poisoning. Across methods and rounds this covered 828,273,240 output entries, including repetitions. The independent reference also used the two products in a nearest-neighbor classifier, recognizing 1,731 of 1,797 original images and 1,761 of 1,797 reduced images. The timings cover the matrix products alone. [Reference program](reference.py) · [Reference fingerprints](reference.json) · [Storage checks and input fingerprints](confirm-results/checks.csv)

Separate native tests covered 54 combinations of dimensions and values, including late failures, nonfinite values, subnormal values and depth 4,096/4,097. Eight cases used integer arithmetic and 46 used the fallback. Three additional checks exercised aliasing, invalid stride and zero depth. The portable boundary test checked 3,780 FP32 bit patterns. [Boundary tests](boundary.py) · [Recorded native results](boundary.json)

One M5 Max ran both experiments. The confirmation used a fresh process, a separately rebuilt candidate with the same binary SHA-256, and five disjoint storage sets per representation. Each round used the same views for every method. The 24 rounds balanced the order of all methods. The custom kernels ran on the calling thread, Accelerate reported single-thread mode, and OpenBLAS used one thread for the pilot reference. macOS supplied wall and calling-thread clocks. CPU affinity, independent system-wide thread counts and hardware counter traces were unavailable. The two representations come from one dataset and share a test population. [Pilot protocol](protocol.md) · [Confirmation protocol](confirm-protocol.md) · [Binary and host identity](confirm-results/identity.json)

## Offline verification

From this directory, Python 3.10 or newer verifies the manifest, recomputes each paired statistic from its raw rounds, matches the recorded layouts and fingerprints, and reruns the portable numerical boundary test.

```sh
python3 -B verify.py
```

To regenerate the two independent reference fingerprints, obtain the Optical Digits ZIP from UCI and install NumPy. The preparation script checks the original ZIP SHA-256 and extracts the four operands. Compare the regenerated reference JSON with the archived copy. [Download and attribution](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits)

```sh
python3 -B prepare.py optdigits.zip operands
python3 -B reference.py optdigits.zip > fresh-reference.json
diff -u reference.json fresh-reference.json
```

## Native reproduction

On compatible Apple hardware, use an SME2-enabled Apple Clang, Accelerate with its single-thread setting, CMake, NumPy and a single-thread-capable OpenBLAS providing CBLAS DGEMM. Set `COB_SRC` to the source checkout at the revision linked below and `OPENBLAS` to the OpenBLAS dylib. The historical standalone control enabled the guarded HK15 route, portable and recursive Strassen, automatic Apple SME leaves, and no BLAS leaf provider. Toolchain and linker differences can affect binary hashes. [COB source used for the control](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/commit/85a007747ff54241f47c4754216e0a65748f2847) · [Recorded library hashes](results/identity.json)

```sh
cmake -S "$COB_SRC" -B cob-build -DCMAKE_BUILD_TYPE=Release \
    -DCOB_ENABLE_PORTABLE_STRASSEN=ON -DCOB_ENABLE_RECURSIVE_STRASSEN=ON \
    -DCOB_ENABLE_HK15_RAGGED=ON -DCOB_STRASSEN_SME=AUTO -DCOB_BLAS_LEAF=NONE
cmake --build cob-build --target cob_matmul
cc -dynamiclib -Wl,-force_load,cob-build/libcob_matmul.a -o cob.dylib
bash build.sh build
python3 -B boundary.py --library build/exact.dylib
python3 -B study.py --kernel build/exact.dylib --cob cob.dylib \
    --oracle "$OPENBLAS" --corpus operands --out fresh-pilot
bash build.sh rebuild
python3 -B confirm.py --kernel rebuild/exact.dylib --cob cob.dylib \
    --corpus operands --out fresh-confirm
```

The original kernel, controls, drivers and verification code in this directory are supplied under the repository MIT license. UCI distributes its dataset under CC BY 4.0. Reproduction uses the original data from UCI and separately installed system and numerical libraries. [MIT license](LICENSE) · [Dataset license and creators](https://doi.org/10.24432/C50P49)
