# Sequential oneDNN BF16 bridge

The restored upstream v3.14.2 source is `4db284662a37cf46401655b5ee99391d9de52c00`. That tag identifies itself at run time as 3.14.1. `build.sh` binds the platform-specific shared library to its previously authenticated SHA-256. The bridge requires a sequential CPU build with zero primitive-cache capacity.

```sh
bash backend/build.sh ONE_DNN_PREFIX OUT
python3 backend/qualify.py OUT/bf16-probe PROOF.json M N K LDA LDB LDC
```

Each plan owns one stream, fresh primitive descriptor and matmul primitive. The shared lazy engine belongs to the calling thread. Every product binds fresh memory objects to the supplied BF16 buffers and FP32 destination, waits for completion and overwrites the output. Destruction releases all per-call state. A caller can reuse its plan for the component products of one multiply. Ordinary execution leaves tracing disabled.

The separate qualification process checks the mapped providers, complete signed products on two disjoint buffer sets, neighboring output guards, unchanged inputs, affinity, task count and lifecycle. Native admission also requires matching geometry and datatypes in oneDNN's execution log, a supported CPU and actual BF16 dot-product instructions in an independently disassembled computational kernel emitted for this descriptor. On Intel the required instruction is `tdpbf16ps`. On Arm it is SVE `bfdot` or `bfmmla`. A kernel dump supplies generated code evidence. It does not count retired instructions. CPU or reference fallbacks retain their exact status and implementation labels. `--diagnostic` runs these fallbacks for correctness. `--emulate-sve-128` labels the Neoverse N2 QEMU process as ineligible for native timing.
