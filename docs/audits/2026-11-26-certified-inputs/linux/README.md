# Single-thread Linux results on certified integer inputs

The checked signed-INT8 entry returned exact FP32 results on the fresh 3584³ square. It beat the fastest qualified ordinary FP32 implementation by **2.98× on the Intel Xeon 6975P-C** and **4.81× on Arm Neoverse-N2**, winning all six paired warm calls on each CPU. Against the strongest measured competitor of any kind, a signed-INT8 implementation, the paired gains were **1.055×** (6/6) and **1.020×** (5/6) respectively. The unrounded Arm gain of 1.01970× missed the prespecified threshold of strictly more than 1.02×. The first-call comparisons against that specialist were 0.910× on Intel and 1.018× on Arm. [Checked INT8 results](checked-int8/README.md) · [Raw peer comparisons](checked-int8/peer-comparisons.csv)

The earlier checked BF16 entry beat ordinary FP32 on the real 1024-pixel handwriting product by **1.76× on Intel** and **2.42× on Arm**. The binary-image specialist packs the same FP32 inputs and computes the dot product with AND and popcount. Against that specialist, BF16 achieved 1.003× on Intel and 0.488× on Arm. BLASFEO exited with signal 11 on this image workload on both hosts in both studies. [Checked BF16 results](certified-bf16/README.md) · [Raw peer comparisons](certified-bf16/peer-comparisons.csv)

The Intel and Arm timings charge the certifying scans, conversions, packing, multiplication and FP32 output stores. Each acceleration requires exact integer inputs within the method’s stated bound and a qualifying native backend. The checked signed-INT8 entry immediately bypasses depths at or below 128. A rejected input uses the supplied FP32 callback. The tested accelerated entries use external numerical libraries. The independently timed standalone COB, vendor-assisted COB, ordinary one-shot FP32 providers, explicit retained plans, unchecked BF16 reference and integer specialists have distinct labels in each report. The scope is one pinned Linux thread, these input distributions and these sampled CPUs. [BF16 research code](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/tree/c609245c8d7359b1e484c7d0896e3ad9a32a97a9/tools/world-rank/exact) · [Signed-INT8 research code](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/tree/6a9c6a696903926b83caf19364b4d19b23f8f2f6/tools/world-rank/exact/s8)

## Reproduce the rankings

From this directory, run the following command with Python 3.11 or later. The standard-library verifier checks both studies, all four host and source receipts, input and oracle hashes, every measured first and warm call, output/guard checks, fallback equality, qualification, paired comparisons and rendered tables.

```sh
python3 -B verify.py
```

Each study has a readable report, a CSV of individual peer comparisons and a ZIP containing its raw tables, build and route fingerprints, and frozen experiment source. The two studies share a single [fixture archive](fixtures/matrices.zip). Each evidence packet includes an optional NumPy checker that rebuilds the two signed real-corpus product hashes from the four prepared operand matrices. The sample inputs and method dimensions are documented with the [UCI attribution and transformations](fixtures/ATTRIBUTION.md).

The checked-INT8 JIT evidence uses a compact, human-readable record of the original kernel and disassembly digests, operation names, site counts, offset hashes and representative operands. The full kernel binaries and vendor SDKs are omitted. An independently authenticated verifier decoded the matching signed instructions from the original bytes, including Intel AMX TDPBSSD and Arm SVE SMMLA. Public replay checks the recorded metadata against the actual per-geometry execution and output receipts. Providing the original GitHub ZIPs additionally replays every recorded instruction site against the original kernel bytes.

```sh
python3 -B verify.py \
  --certified-bf16-x86_64 path/to/bf16-x86.zip \
  --certified-bf16-aarch64 path/to/bf16-arm.zip \
  --checked-int8-x86_64 path/to/int8-x86.zip \
  --checked-int8-aarch64 path/to/int8-arm.zip
```

Each original ZIP is optional. Its download digest and identity are in the study report and packaged host receipt. [BF16 hosted run](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45788542642) · [Checked INT8 hosted run](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45792579553)

## Source and notices

The frozen COB code carries the repository’s [MIT license](LICENSE). Ethem Alpaydin and Cenk Kaynak’s UCI handwriting data and the four prepared matrices are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Qualification used oneDNN v3.14.2 at commit `4db284662a37cf46401655b5ee99391d9de52c00`, whose runtime reports version 3.14.1. [oneDNN license](notices/oneDNN/LICENSE) · [Third-party program notices](notices/oneDNN/THIRD-PARTY-PROGRAMS) · [Pinned oneDNN source](https://github.com/uxlfoundation/oneDNN/tree/4db284662a37cf46401655b5ee99391d9de52c00)
