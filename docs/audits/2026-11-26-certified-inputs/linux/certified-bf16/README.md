# Exact integer products through an FP32 SGEMM interface

A content check lets an ordinary FP32 matrix multiply use one BF16 product when all input values and partial sums are exactly representable. The registered Linux experiment compares the complete operation with qualified FP32 providers and domain-specific integer implementations on the same input buffers.

- **Intel(R) Xeon(R) 6975P-C.** Of 4 admitted integer fixtures, the candidate cleared the paired rule against the fastest qualified one-shot FP32 method on 2/4 first calls and 2/4 warm calls. Against the quickest eligible peer of any kind, the warm count was 0/4. Warm ratios on UCI digits were 0.307× versus `mkl` on the 64-input representation and 1.764× versus `fp32-policy` on the 1024-input representation.
- **Neoverse-N2.** Of 4 admitted integer fixtures, the candidate cleared the paired rule against the fastest qualified one-shot FP32 method on 4/4 first calls and 4/4 warm calls. Against the quickest eligible peer of any kind, the warm count was 0/4. Warm ratios on UCI digits were 1.203× versus `cob-standalone` on the 64-input representation and 2.424× versus `cob-standalone` on the 1024-input representation.

### What the six fixtures measure

Two signed synthetic products use values from −63 to 63 on a 4096³ cube and a cropped 1079×1553×1201 rectangle. Two copies of the rectangle add ½ to the first A value or the last B value. The real products compare handwritten digits across the original UCI test and training partitions. One computes 64-feature dot products across 1,797×3,823 pairs. The other counts shared set pixels across 1,797×1,934 pairs of 1,024-pixel binary images. These are two representations from one collection. [UCI dataset](https://doi.org/10.24432/C50P49) · [Input transformations and hashes](../fixtures/ATTRIBUTION.md)

The certificate requires integral inputs with magnitude at most 256, K ≤ 4,096 and K × max|A| × max|B| ≤ 2²⁴. A qualifying call scans the current inputs and uses vendor BF16 arithmetic through oneDNN or Arm Compute Library. Each rejected call delegates to the registered FP32 policy. Standalone COB has its own timing entry. Benchmark initialization fixes native availability for each tested geometry. Admission frequency on other applications remains unmeasured.

A fresh process times its first multiply after library initialization, then its next multiply. Content inspection, allocations, conversions and packing inside each multiply are charged. Each ratio is peer time divided by candidate time, paired within a block and scheduling round. The reported speedup is the median of six paired ratios. Values above one favor the candidate. The registered improvement rule requires a median above 1.02× and at least five strict wins. The six pairs are within one VM per architecture. Two seeds change the synthetic inputs and leave both real datasets fixed.

Every ordinary call is screened on every output against an exact binary64 reference and an absolute-product sum. Numerical qualification requires finite outputs, absolute error ≤0.002 and error ≤ K/(2²⁴−K) times the absolute-product sum. Exact qualification requires zero deviation. The fallback must also match a separate call to the same provider bit-for-bit. The known full-output hashes of both UCI products provide independent checks. [All individual peer ratios and raw medians](peer-comparisons.csv)

## Intel(R) Xeon(R) 6975P-C

The original job recorded 534 process runs on CPU 0. [Hosted job](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45788542642/job/139801417544) · Artifact receipt (`x86_64/provenance.json` in [evidence.zip](evidence.zip)) · Timing records (`x86_64/raw/observations.jsonl` in [evidence.zip](evidence.zip)) · Admission and provider evidence (`x86_64/audit.json` in [evidence.zip](evidence.zip))

BF16 qualification required exact full-output executions and a matching instruction witness. The signed oneDNN INT8 qualification used s8×s8→s32 execution. Instruction names below come from the linked or generated arithmetic kernel. A generic descriptor alone cannot identify its instruction set.

| Independently probed geometry | Executed BF16 route and instruction | Signed INT8 primitive and instruction evidence |
|---|---|---|
| Signed cube | `brg_matmul:avx10_1_512_amx` · `tdpbf16ps` | `brg_matmul:avx10_1_512_amx_fp16` · `tdpbssd` |
| Signed rectangle | `brg_matmul:avx10_1_512_amx` · `tdpbf16ps` | `brg_matmul:avx10_1_512_amx_fp16` · `tdpbssd` |
| UCI feature counts | `brg_matmul:avx10_1_512_amx` · `tdpbf16ps` | `brg_matmul:avx10_1_512_amx_fp16` · `tdpbssd` |
| UCI binary pixels | `brg_matmul:avx10_1_512_amx` · `tdpbf16ps` | `brg_matmul:avx10_1_512_amx_fp16` · `tdpbssd` |

- Ordinary backing provider `release-mkl-aligned-two` on Signed cube.
- Ordinary backing provider `mkl-jit` on Signed rectangle, First A + ½, Last B + ½, UCI feature counts, UCI binary pixels.

### Complete-call timings

The FP32 opponent has the lowest raw median among numerically qualified one-shot FP32 methods. The last column gives the lowest raw median across the whole qualified field, including this candidate, the unchecked BF16 control, exact integer specialists and retained plans when eligible.

| Fixture and call | Candidate ms | One-shot FP32 leader, ms | Paired speedup (wins) | Full-field raw leader, ms |
|---|---:|---|---:|---|
| Signed cube 4096×4096×4096 · first | 316.370 | `fp32-policy` 584.148 | 1.843× (6/6) | `onednn-int8` 193.824 |
| Signed cube 4096×4096×4096 · warm | 325.311 | `fp32-policy` 579.685 | 1.778× (6/6) | `onednn-int8` 148.299 |
| Signed rectangle 1079×1553×1201 · first | 21.758 | `openblas` 19.625 | 0.900× (0/6) | `openblas` 19.625 |
| Signed rectangle 1079×1553×1201 · warm | 20.531 | `openblas` 18.986 | 0.923× (0/6) | `onednn-int8` 18.818 |
| First A + ½ 1079×1553×1201 · first | 21.264 | `openblas` 19.605 | 0.920× (0/6) | `openblas` 19.605 |
| First A + ½ 1079×1553×1201 · warm | 19.256 | `openblas` 18.982 | 0.986× (0/6) | `openblas` 18.982 |
| Last B + ½ 1079×1553×1201 · first | 21.925 | `fastmmw-1536` 19.566 | 0.895× (0/6) | `fastmmw-1536` 19.566 |
| Last B + ½ 1079×1553×1201 · warm | 20.032 | `fastmmw-1536` 19.076 | 0.951× (0/6) | `fastmmw-1536` 19.076 |
| UCI feature counts 1797×3823×64 · first | 17.819 | `cob-standalone` 5.526 | 0.310× (0/6) | `cob-standalone` 5.526 |
| UCI feature counts 1797×3823×64 · warm | 15.838 | `mkl` 4.869 | 0.307× (0/6) | `mkl` 4.869 |
| UCI binary pixels 1797×1934×1024 · first | 26.578 | `openblas` 34.064 | 1.283× (6/6) | `bitset-popcount` 19.776 |
| UCI binary pixels 1797×1934×1024 · warm | 18.785 | `fp32-policy` 33.075 | 1.764× (6/6) | `exact-dispatch` 18.785 |

On the signed 4096³ cube, exact `onednn-int8` took 148.299 ms warm against 579.685 ms for `fp32-policy`, the fastest qualified one-shot FP32 method. Its directly paired speedup was **3.914× (6/6)**. The six FP32/INT8 time ratios were 3.904, 3.937, 3.726, 3.964, 3.925, 3.874. Each INT8 call validated and converted the incoming FP32 values to signed bytes, created its oneDNN primitive, multiplied and copied the signed 32-bit result back to FP32. The specialist aborts on inputs outside its certified domain.

### Exact, independent and specialist comparisons

Each entry gives the named raw-median leader of its field and the candidate’s paired speedup against it. Exact FP32 requires every output to equal its reference. The independent fields exclude the candidate, its backing wrapper and released COB methods. The inclusive fields admit the retained oneDNN plan.

| Fixture and call | Exact FP32, any | Ordinary FP32, any | Independent FP32, one-shot | Independent FP32, any | Exact integer or binary |
|---|---|---|---|---|---|
| Signed cube · first | `fp32-policy` 1.843× (6/6) | `fp32-policy` 1.843× (6/6) | `fastmmw-1536` 1.926× (6/6) | `fastmmw-1536` 1.926× (6/6) | `onednn-int8` 0.614× (0/6) |
| Signed cube · warm | `fp32-policy` 1.778× (6/6) | `fp32-policy` 1.778× (6/6) | `fastmmw-1536` 1.831× (6/6) | `fastmmw-1536` 1.831× (6/6) | `onednn-int8` 0.456× (0/6) |
| Signed rectangle · first | `openblas` 0.900× (0/6) | `openblas` 0.900× (0/6) | `openblas` 0.900× (0/6) | `openblas` 0.900× (0/6) | `onednn-int8` 0.920× (0/6) |
| Signed rectangle · warm | `openblas` 0.923× (0/6) | `openblas` 0.923× (0/6) | `openblas` 0.923× (0/6) | `openblas` 0.923× (0/6) | `onednn-int8` 0.914× (0/6) |
| First A + ½ · first | `openblas` 0.920× (0/6) | `openblas` 0.920× (0/6) | `openblas` 0.920× (0/6) | `openblas` 0.920× (0/6) | — |
| First A + ½ · warm | `openblas` 0.986× (0/6) | `openblas` 0.986× (0/6) | `openblas` 0.986× (0/6) | `openblas` 0.986× (0/6) | — |
| Last B + ½ · first | `fastmmw-1536` 0.895× (0/6) | `fastmmw-1536` 0.895× (0/6) | `fastmmw-1536` 0.895× (0/6) | `fastmmw-1536` 0.895× (0/6) | — |
| Last B + ½ · warm | `fastmmw-1536` 0.951× (0/6) | `fastmmw-1536` 0.951× (0/6) | `fastmmw-1536` 0.951× (0/6) | `fastmmw-1536` 0.951× (0/6) | — |
| UCI feature counts · first | `cob-standalone` 0.310× (0/6) | `cob-standalone` 0.310× (0/6) | `fastmmw-1536` 0.322× (0/6) | `fastmmw-1536` 0.322× (0/6) | `onednn-int8` 0.980× (2/6) |
| UCI feature counts · warm | `mkl` 0.307× (0/6) | `mkl` 0.307× (0/6) | `mkl` 0.307× (0/6) | `mkl` 0.307× (0/6) | `onednn-int8` 0.903× (0/6) |
| UCI binary pixels · first | `openblas` 1.283× (6/6) | `openblas` 1.283× (6/6) | `openblas` 1.283× (6/6) | `openblas` 1.283× (6/6) | `bitset-popcount` 0.745× (0/6) |
| UCI binary pixels · warm | `fp32-policy` 1.764× (6/6) | `fp32-policy` 1.764× (6/6) | `mkl-jit` 1.760× (6/6) | `mkl-jit` 1.760× (6/6) | `bitset-popcount` 1.003× (4/6) |

### Backing provider, standalone COB and unchecked arithmetic

Ratios compare the candidate with a separately timed call on the same input. The unchecked BF16 control omits the integer certificate. The signed INT8 provider and binary AND/popcount provider accept ordinary FP32 buffers and include conversion or packing. A dash means the comparison did not qualify or was outside the registered domain.

| Fixture and call | Backing FP32 | Standalone COB | Raw BF16 | oneDNN INT8 | Binary AND/popcount |
|---|---:|---:|---:|---:|---:|
| Signed cube · first | 1.843× (6/6) | 2.019× (6/6) | 0.959× (0/6) | 0.614× (0/6) | — |
| Signed cube · warm | 1.778× (6/6) | 1.953× (6/6) | 0.951× (0/6) | 0.456× (0/6) | — |
| Signed rectangle · first | 0.973× (0/6) | 1.092× (6/6) | 0.977× (1/6) | 0.920× (0/6) | — |
| Signed rectangle · warm | 0.941× (0/6) | 1.154× (6/6) | 0.959× (0/6) | 0.914× (0/6) | — |
| First A + ½ · first | 1.000× (3/6) | 1.120× (6/6) | 1.014× (6/6) | — | — |
| First A + ½ · warm | 1.007× (6/6) | 1.228× (6/6) | 1.032× (6/6) | — | — |
| Last B + ½ · first | 0.968× (0/6) | 1.091× (6/6) | 0.984× (0/6) | — | — |
| Last B + ½ · warm | 0.965× (0/6) | 1.179× (6/6) | 0.999× (1/6) | — | — |
| UCI feature counts · first | 0.361× (0/6) | 0.310× (0/6) | 1.009× (3/6) | 0.980× (2/6) | — |
| UCI feature counts · warm | 0.306× (0/6) | 0.346× (0/6) | 0.997× (2/6) | 0.903× (0/6) | — |
| UCI binary pixels · first | 1.323× (6/6) | 1.417× (6/6) | 0.977× (3/6) | 0.841× (0/6) | 0.745× (0/6) |
| UCI binary pixels · warm | 1.764× (6/6) | 1.990× (6/6) | 1.262× (6/6) | 1.042× (6/6) | 1.003× (4/6) |

### Output and fallback checks

| Fixture | Timed route | Inexact outputs / checked | Largest absolute error | Ordinary FP32 methods exact / qualified |
|---|---|---:|---:|---:|
| Signed cube | one certified BF16 product | 0 / 201,326,592 | 0 | 12/12 |
| Signed rectangle | one certified BF16 product | 0 / 20,108,244 | 0 | 12/12 |
| First A + ½ | FP32, fractional value | 0 / 20,108,244 | 0 | 12/12 |
| Last B + ½ | FP32, fractional value | 0 / 20,108,244 | 0 | 12/12 |
| UCI feature counts | one certified BF16 product | 0 / 82,439,172 | 0 | 12/12 |
| UCI binary pixels | one certified BF16 product | 0 / 41,704,776 | 0 | 11/11 |

The route records show 64 on first A and 3,161,032 on last B input values inspected. The first and warmed results matched the selected FP32 backing call bit-for-bit in every qualified pair.

### Exclusions

- `bitset-popcount` — outside registered binary image workload (Signed cube, Signed rectangle, First A + ½, Last B + ½, UCI feature counts).
- `onednn-int8` — fractional input outside INT8 domain (First A + ½, Last B + ½).
- `blasfeo` — worker exit -11: BLASFEO target=X64_INTEL_HASWELL API=standard-SGEMM workspace=initialized inputs=unpacked timed_output_clear=0 (UCI binary pixels).

All scored methods and raw medians (`x86_64/raw/score.json` in [evidence.zip](evidence.zip)) · Six pairs for every registered comparison group (`x86_64/raw/comparisons.csv` in [evidence.zip](evidence.zip)) · Per-geometry admissions (`x86_64/raw/admissions.json` in [evidence.zip](evidence.zip))

## Neoverse-N2

The original job recorded 426 process runs on CPU 0. [Hosted job](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45788542642/job/139801417850) · Artifact receipt (`aarch64/provenance.json` in [evidence.zip](evidence.zip)) · Timing records (`aarch64/raw/observations.jsonl` in [evidence.zip](evidence.zip)) · Admission and provider evidence (`aarch64/audit.json` in [evidence.zip](evidence.zip))

BF16 qualification required exact full-output executions and a matching instruction witness. The signed oneDNN INT8 qualification used s8×s8→s32 execution. Instruction names below come from the linked or generated arithmetic kernel. A generic descriptor alone cannot identify its instruction set.

| Independently probed geometry | Executed BF16 route and instruction | Signed INT8 primitive and instruction evidence |
|---|---|---|
| Signed cube | `a64_interleaved_bf16fp32_mmla_8x12` · `bfmmla` | `jit:int8` · `smmla` |
| Signed rectangle | `a64_interleaved_bf16fp32_mmla_8x12` · `bfmmla` | `jit:int8` · `smmla` |
| UCI feature counts | `a64_interleaved_bf16fp32_mmla_8x12` · `bfmmla` | `jit:int8` · `smmla` |
| UCI binary pixels | `a64_interleaved_bf16fp32_mmla_8x12` · `bfmmla` | `jit:int8` · `smmla` |

- Ordinary backing provider `fastmmw-1536` on Signed cube.
- Ordinary backing provider `armpl` on Signed rectangle, First A + ½, Last B + ½, UCI feature counts, UCI binary pixels.

### Complete-call timings

The FP32 opponent has the lowest raw median among numerically qualified one-shot FP32 methods. The last column gives the lowest raw median across the whole qualified field, including this candidate, the unchecked BF16 control, exact integer specialists and retained plans when eligible.

| Fixture and call | Candidate ms | One-shot FP32 leader, ms | Paired speedup (wins) | Full-field raw leader, ms |
|---|---:|---|---:|---|
| Signed cube 4096×4096×4096 · first | 826.428 | `fastmmw-1536` 2,117.932 | 2.563× (6/6) | `onednn-int8` 432.280 |
| Signed cube 4096×4096×4096 · warm | 822.257 | `fastmmw-1536` 2,098.952 | 2.554× (6/6) | `onednn-int8` 427.863 |
| Signed rectangle 1079×1553×1201 · first | 34.242 | `cob-standalone` 74.246 | 2.168× (6/6) | `onednn-int8` 20.501 |
| Signed rectangle 1079×1553×1201 · warm | 32.221 | `cob-standalone` 73.863 | 2.291× (6/6) | `onednn-int8` 18.687 |
| First A + ½ 1079×1553×1201 · first | 79.787 | `cob-standalone` 74.093 | 0.929× (0/6) | `raw-bf16` 32.658 |
| First A + ½ 1079×1553×1201 · warm | 77.627 | `cob-standalone` 73.824 | 0.950× (0/6) | `raw-bf16` 30.316 |
| Last B + ½ 1079×1553×1201 · first | 81.607 | `cob-standalone` 74.198 | 0.909× (0/6) | `raw-bf16` 32.703 |
| Last B + ½ 1079×1553×1201 · warm | 79.462 | `cob-standalone` 73.772 | 0.928× (0/6) | `raw-bf16` 30.460 |
| UCI feature counts 1797×3823×64 · first | 16.956 | `cob-standalone` 17.741 | 1.044× (6/6) | `onednn-int8` 14.579 |
| UCI feature counts 1797×3823×64 · warm | 14.662 | `cob-standalone` 17.603 | 1.203× (6/6) | `onednn-int8` 12.010 |
| UCI binary pixels 1797×1934×1024 · first | 54.978 | `cob-standalone` 127.019 | 2.309× (6/6) | `bitset-popcount` 25.796 |
| UCI binary pixels 1797×1934×1024 · warm | 52.078 | `cob-standalone` 126.268 | 2.424× (6/6) | `bitset-popcount` 25.406 |

On the signed 4096³ cube, exact `onednn-int8` took 427.863 ms warm against 2,098.952 ms for `fastmmw-1536`, the fastest qualified one-shot FP32 method. Its directly paired speedup was **4.905× (6/6)**. The six FP32/INT8 time ratios were 4.913, 4.904, 4.876, 4.906, 4.903, 4.910. Each INT8 call validated and converted the incoming FP32 values to signed bytes, created its oneDNN primitive, multiplied and copied the signed 32-bit result back to FP32. The specialist aborts on inputs outside its certified domain.

### Exact, independent and specialist comparisons

Each entry gives the named raw-median leader of its field and the candidate’s paired speedup against it. Exact FP32 requires every output to equal its reference. The independent fields exclude the candidate, its backing wrapper and released COB methods. The inclusive fields admit the retained oneDNN plan.

| Fixture and call | Exact FP32, any | Ordinary FP32, any | Independent FP32, one-shot | Independent FP32, any | Exact integer or binary |
|---|---|---|---|---|---|
| Signed cube · first | `fastmmw-1536` 2.563× (6/6) | `fastmmw-1536` 2.563× (6/6) | `fastmmw-1536` 2.563× (6/6) | `fastmmw-1536` 2.563× (6/6) | `onednn-int8` 0.523× (0/6) |
| Signed cube · warm | `fastmmw-1536` 2.554× (6/6) | `fastmmw-1536` 2.554× (6/6) | `fastmmw-1536` 2.554× (6/6) | `fastmmw-1536` 2.554× (6/6) | `onednn-int8` 0.520× (0/6) |
| Signed rectangle · first | `cob-standalone` 2.168× (6/6) | `cob-standalone` 2.168× (6/6) | `openblas` 2.275× (6/6) | `openblas` 2.275× (6/6) | `onednn-int8` 0.594× (0/6) |
| Signed rectangle · warm | `cob-standalone` 2.291× (6/6) | `cob-standalone` 2.291× (6/6) | `openblas` 2.398× (6/6) | `openblas` 2.398× (6/6) | `onednn-int8` 0.581× (0/6) |
| First A + ½ · first | `cob-standalone` 0.929× (0/6) | `cob-standalone` 0.929× (0/6) | `openblas` 0.976× (0/6) | `openblas` 0.976× (0/6) | — |
| First A + ½ · warm | `cob-standalone` 0.950× (0/6) | `cob-standalone` 0.950× (0/6) | `openblas` 0.995× (0/6) | `openblas` 0.995× (0/6) | — |
| Last B + ½ · first | `cob-standalone` 0.909× (0/6) | `cob-standalone` 0.909× (0/6) | `openblas` 0.954× (0/6) | `openblas` 0.954× (0/6) | — |
| Last B + ½ · warm | `cob-standalone` 0.928× (0/6) | `cob-standalone` 0.928× (0/6) | `openblas` 0.973× (0/6) | `openblas` 0.973× (0/6) | — |
| UCI feature counts · first | `cob-standalone` 1.044× (6/6) | `cob-standalone` 1.044× (6/6) | `armpl` 1.063× (6/6) | `armpl` 1.063× (6/6) | `onednn-int8` 0.862× (0/6) |
| UCI feature counts · warm | `cob-standalone` 1.203× (6/6) | `cob-standalone` 1.203× (6/6) | `armpl` 1.208× (6/6) | `armpl` 1.208× (6/6) | `onednn-int8` 0.819× (0/6) |
| UCI binary pixels · first | `cob-standalone` 2.309× (6/6) | `cob-standalone` 2.309× (6/6) | `openblas` 2.493× (6/6) | `openblas` 2.493× (6/6) | `bitset-popcount` 0.469× (0/6) |
| UCI binary pixels · warm | `cob-standalone` 2.424× (6/6) | `cob-standalone` 2.424× (6/6) | `openblas` 2.610× (6/6) | `openblas` 2.610× (6/6) | `bitset-popcount` 0.488× (0/6) |

### Backing provider, standalone COB and unchecked arithmetic

Ratios compare the candidate with a separately timed call on the same input. The unchecked BF16 control omits the integer certificate. The signed INT8 provider and binary AND/popcount provider accept ordinary FP32 buffers and include conversion or packing. A dash means the comparison did not qualify or was outside the registered domain.

| Fixture and call | Backing FP32 | Standalone COB | Raw BF16 | oneDNN INT8 | Binary AND/popcount |
|---|---:|---:|---:|---:|---:|
| Signed cube · first | 2.564× (6/6) | 2.593× (6/6) | 0.975× (0/6) | 0.523× (0/6) | — |
| Signed cube · warm | 2.555× (6/6) | 2.607× (6/6) | 0.976× (0/6) | 0.520× (0/6) | — |
| Signed rectangle · first | 2.332× (6/6) | 2.168× (6/6) | 0.953× (0/6) | 0.594× (0/6) | — |
| Signed rectangle · warm | 2.406× (6/6) | 2.291× (6/6) | 0.941× (0/6) | 0.581× (0/6) | — |
| First A + ½ · first | 1.002× (3/6) | 0.929× (0/6) | 0.409× (0/6) | — | — |
| First A + ½ · warm | 0.998× (2/6) | 0.950× (0/6) | 0.391× (0/6) | — | — |
| Last B + ½ · first | 0.973× (0/6) | 0.909× (0/6) | 0.401× (0/6) | — | — |
| Last B + ½ · warm | 0.977× (0/6) | 0.928× (0/6) | 0.383× (0/6) | — | — |
| UCI feature counts · first | 1.065× (6/6) | 1.044× (6/6) | 0.994× (2/6) | 0.862× (0/6) | — |
| UCI feature counts · warm | 1.209× (6/6) | 1.203× (6/6) | 0.979× (0/6) | 0.819× (0/6) | — |
| UCI binary pixels · first | 2.529× (6/6) | 2.309× (6/6) | 0.958× (0/6) | 0.592× (0/6) | 0.469× (0/6) |
| UCI binary pixels · warm | 2.631× (6/6) | 2.424× (6/6) | 0.950× (0/6) | 0.575× (0/6) | 0.488× (0/6) |

### Output and fallback checks

| Fixture | Timed route | Inexact outputs / checked | Largest absolute error | Ordinary FP32 methods exact / qualified |
|---|---|---:|---:|---:|
| Signed cube | one certified BF16 product | 0 / 201,326,592 | 0 | 9/9 |
| Signed rectangle | one certified BF16 product | 0 / 20,108,244 | 0 | 9/9 |
| First A + ½ | FP32, fractional value | 0 / 20,108,244 | 0 | 9/9 |
| Last B + ½ | FP32, fractional value | 0 / 20,108,244 | 0 | 9/9 |
| UCI feature counts | one certified BF16 product | 0 / 82,439,172 | 0 | 9/9 |
| UCI binary pixels | one certified BF16 product | 0 / 41,704,776 | 0 | 8/8 |

The route records show 64 on first A and 3,161,032 on last B input values inspected. The first and warmed results matched the selected FP32 backing call bit-for-bit in every qualified pair.

### Exclusions

- `bitset-popcount` — outside registered binary image workload (Signed cube, Signed rectangle, First A + ½, Last B + ½, UCI feature counts).
- `onednn-int8` — fractional input outside INT8 domain (First A + ½, Last B + ½).
- `blasfeo` — worker exit -11: BLASFEO target=ARMV8A_ARM_CORTEX_A76 API=standard-SGEMM workspace=initialized inputs=unpacked timed_output_clear=1 (UCI binary pixels).

All scored methods and raw medians (`aarch64/raw/score.json` in [evidence.zip](evidence.zip)) · Six pairs for every registered comparison group (`aarch64/raw/comparisons.csv` in [evidence.zip](evidence.zip)) · Per-geometry admissions (`aarch64/raw/admissions.json` in [evidence.zip](evidence.zip))

## Reproduction

Run `python3 -B verify.py` from the [Linux directory](../README.md). The [study archive](evidence.zip) contains frozen source, raw observations, qualification and comparison records. [All six paired ratios by peer](peer-comparisons.csv)

- x86_64 · [Run](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45788542642) · [Job 139801417544](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45788542642/job/139801417544) · artifact 15101368713, SHA-256 `03c29409b3249a76b86624a6fcacca258f427d0de36ace7fb6e8ff386ce5ccac`.

- aarch64 · [Run](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45788542642) · [Job 139801417850](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs/45788542642/job/139801417850) · artifact 15101803846, SHA-256 `b6c5a310c8fece0ed9b33f743379d086d9c679f7b9772c1e3fcae9323a3221c3`.

[Frozen benchmark source and protocol](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/tree/c609245c8d7359b1e484c7d0896e3ad9a32a97a9/tools/world-rank/exact) · [Timed COB release source](https://github.com/Milian0402/consequence_of_boredom_and_curiosity/tree/e4b7ce19cdf49f9175c86d3e91ea8d3a756eae9b)
