# Exact integer products through an FP32 matrix interface

A checked signed-INT8 implementation multiplied a fresh 3584 × 3584 pair of bounded integer matrices **2.98× faster on Intel Xeon 6975P-C** and **4.81× faster on Arm Neoverse N2** than the fastest qualifying ordinary FP32 implementation tested on each CPU. It won all six warm comparisons on both. Each call started with FP32 inputs and included content checks, conversion, allocation, oneDNN primitive construction, multiplication and writing FP32 output. The Linux experiment used one pinned CPU thread. [Linux results and raw evidence](linux/README.md) · [Original confirmation run](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45792579553)

| Checked INT8 case | Intel Xeon 6975P-C | Arm Neoverse N2 |
|---|---:|---:|
| Fresh 3584³ signed integers | 137.65 ms vs 410.58 ms · **2.98×** | 300.25 ms vs 1,447.28 ms · **4.81×** |
| Repeated 4096³ signed integers | 2.73× | 4.52× |
| Cropped 1079 × 1553 × 1201 signed integers | 0.85× | 4.01× |
| Original binary handwriting images, depth 1024 | 1.28× | 4.23× |
| Handwriting feature counts, depth 64, FP32 bypass | 0.99× | 1.00× |

Ratios divide the fastest qualifying ordinary FP32 competitor's elapsed time by the checked implementation's elapsed time in the corresponding round. Each table entry reports the median ratio over six pairs. The millisecond values are separate warm medians. The fresh-square FP32 leaders were COB's two-level Winograd with sequential oneMKL on Intel and FastMMW with its 768 cutoff on Arm. The first call after library initialization on the fresh square gained 2.52× and 4.92×, respectively. [Frozen policy, source and C API](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/tree/6a9c6a696903926b83caf19364b4d19b23f8f2f6/tools/world-rank/exact/s8)

**Integer specialists narrow the margin.** On the fresh square the checked entry also had the lowest warm median among all measured methods. It gained 5.51% over the direct signed-INT8 oneDNN comparator on Intel (six of six pairs) and 1.97% on Arm (five of six). The registered rule required more than 2% and at least five wins. On the real binary images a freshly packed AND/popcount method was fastest on both Linux CPUs. BLASFEO's worker crashed on that image layout on both hosts and was excluded there. [Complete peer comparisons and exclusions](linux/README.md)

**Why exact arithmetic is possible.** The Linux signed path admits integral FP32 inputs in [−128,127], with 129 ≤ K ≤ 4096 and K × max|A| × max|B| ≤ 2²⁴. This bounds every possible partial sum. Inputs convert exactly to signed bytes, the signed 32-bit accumulator stays in range, and all result integers convert exactly to FP32. The code uses Intel AMX INT8 or Arm SVE I8MM through sequential oneDNN. On a rejected input or unavailable backend, the caller's FP32 routine produces the output. A fractional value at the start of A gave near-parity with that fallback. A single fractional value at the end of B added 43% on Intel and 6.6% on Arm because the call had already inspected and packed nearly both matrices. All 1,079,871,888 output-cell comparisons for the checked entry across both jobs matched the reference, counting repeated calls. [Linux method and verification](linux/README.md)

## What transferred

- **Apple M5 Max.** A separate SME signed-INT8 kernel with a narrower integer certificate processed the same original handwriting images in 2.862 ms. It beat standalone COB by 1.399× in 23 of 24 paired rounds and single-thread Accelerate by 1.634× in 24 of 24. On the short-depth handwriting counts, standalone COB won all 24 rounds. macOS did not provide core pinning or independent per-call thread counts. [Apple sources and measurements](apple/README.md)
- **Single-product BF16 on Linux.** An earlier certificate allowed larger integer inputs with magnitude up to 256 under the same 2²⁴ sum bound. At 4096³ the checked BF16 path beat the strongest measured FP32 method by 1.78× on Intel and 2.55× on Arm. Direct signed INT8 on the same FP32 inputs, including conversion, reached 3.91× and 4.91× respectively. That comparison motivated the checked signed-INT8 confirmation above. [Original Linux run](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45788542642) · [Linux replay](linux/README.md)
- **Application coverage.** Three of eight deliberately selected operations from four public datasets met the wider integer certificate. They involved binary handwriting images, handwriting feature counts, and uncentered MovieLens ratings. Two of the eight also met the signed-INT8 selector’s depth requirement. In a trace of a pretrained two-layer Google BERT model, zero of 63 dense products qualified. Three sequence lengths contributed 75,055,104 scalar multiply-accumulates. The eight selected operations and single model trace measure eligibility on those workloads. [Application inputs, rules and provenance](applications/README.md)

For arbitrary FP32 matrices, a separate experiment tested BF16 component methods that preserve more precision. Its accuracy-qualified methods won zero of 336 paired warm comparisons with the fastest measured FP32 methods. [Unrestricted FP32 experiment](../2026-11-25-ranking/bf16-components/README.md)

## Replay offline

With Python 3.11 or later, run this command from the repository root. It verifies the archived observations, recomputes the comparisons and checks the application records. Rerunning native hardware timings and recreating the original datasets have separate requirements in each study.

```sh
python3 -B docs/audits/2026-11-26-certified-inputs/verify.py
```

The original UCI handwriting data were contributed by Ethem Alpaydin and Cenk Kaynak under CC BY 4.0. The application study documents the exact partitions, transformations and hashes. The BERT trace uses the identified Google checkpoint and sequence lengths. [Optical Recognition of Handwritten Digits](https://doi.org/10.24432/C50P49) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) · [Applications and model revision](applications/README.md)
