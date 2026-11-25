# Arm KleidiAI single-thread FP32 comparison on Apple M5 Max

These two benchmark adapters call public Arm KleidiAI FP32 kernels from stable v1.33.0, commit `ab3f4b24a75f0667a1af9ebfe27e94efa7ce97e1`. `kleidiai-fixed` uses its two-vector SME2 MOPA kernel. `kleidiai-elastic` uses the public SME2 dispatcher for six tile shapes. Both accept ordinary row-major matrices, including offset pointers and padded row strides. Positive K calls allocate and align their own scratch, initialize the required column bias to zero, pack A and B with the corresponding upstream packers, write `C = A × B` directly and release the scratch. The timed ABI is `mtmm_sgemm_rowmajor` with the leading config field `num_threads=1`. Each invocation assumes one caller with streaming mode and ZA inactive. [Fixed SME2 public API](https://github.com/ARM-software/kleidiai/blob/ab3f4b24a75f0667a1af9ebfe27e94efa7ce97e1/kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa.h) · [Elastic SME2 public API](https://github.com/ARM-software/kleidiai/blob/ab3f4b24a75f0667a1af9ebfe27e94efa7ce97e1/kai/ukernels/matmul/kai_matmul.h#L121-L133)

## Reproduce the M5 Max build and checks

Run from the directory containing this README using Xcode tools and Python 3 on an arm64 M5 Max. Build output and third-party files go in a fresh temporary directory. The builder verifies the immutable revision and the pristine upstream `kai/` tree. It writes product and input hashes to `identity.json` and the local compile/link commands to `commands.txt`.

```sh
scratch=$(mktemp -d)
git clone --quiet --depth 1 --branch v1.33.0 https://github.com/ARM-software/kleidiai.git "$scratch/upstream"
python3 -B build.py "$scratch/upstream" "$scratch/build"
python3 -B qualify.py "$scratch/build"
```

`$scratch/build/kleidiai-fixed.dylib` and `$scratch/build/kleidiai-elastic.dylib` are ready for the single-thread harness. Each depends on Apple's libSystem. Xcode links the SME lazy-save support into the dylib. The upstream C wrappers require a compiler target advertising SVE2, so the recipe disables automatic vectorization for those translation units. The independent checker disassembles all ordinary C objects and rejects scalable-vector instructions or registers outside the hand-written assembly. Apple's nonstreaming FP32 SVE kernel has no advertised SVE execution capability on this host. Upstream handles SME kernel entry, lazy ZA save and exit in its Apple-compatible assembly. [Upstream macOS feature map](https://github.com/ARM-software/kleidiai/blob/ab3f4b24a75f0667a1af9ebfe27e94efa7ce97e1/test/common/cpu_info.cpp#L254-L272) · [Arm ZA-save and SVL implementation](https://github.com/ARM-software/kleidiai/blob/ab3f4b24a75f0667a1af9ebfe27e94efa7ce97e1/kai/kai_common_sme_asm.S)

## Recorded qualification

Each ordinary dylib passed 39 finite random, cancellation and scaled fixtures over 13 matrix sizes with shifted pointers, independently padded row strides, read-only A/B, inaccessible end pages, full C padding checks and an FP64 reference. Zero M and zero N smoke calls also passed. The native streaming vector was 64 bytes. The caller observed one Mach task thread and inactive streaming state around each call. All local qualification calls were untimed.

| Adapter | Output elements checked | C guard cells | Maximum absolute error |
| --- | ---: | ---: | ---: |
| Fixed SME2 | 122,382 | 111,090 | 0.000001369 |
| Elastic SME2 | 122,382 | 111,090 | 0.000024895 |

`recorded/qualification.json` gives the exact errors, binary hashes, platform and assembly audit counts. `recorded/identity.json` lists the direct source hashes. `eligibility/` documents the Apple feature probes, the 41 matching full-FP32 files in stable and the subsequent release candidate, and the tested kernel selection. An independent FP32 FMOPA probe succeeded. The exact FP32 FMOP4A opcode required by the optional SME2.1 quarter-tile dispatcher raised SIGILL. Unmeasured FP32 alternatives in this KleidiAI release include the older fixed SME MOPA kernel and raw-left-input NEON kernels. These comparisons cover the two named SME2 implementations.

The selected upstream source, assembly and API files carry Apache-2.0 SPDX notices. The source-only replay kit contains the benchmark glue, test receipts and source identities. The build fetches the unmodified upstream sources separately. [Official release](https://github.com/ARM-software/kleidiai/releases/tag/v1.33.0) · [Upstream license text](https://github.com/ARM-software/kleidiai/blob/ab3f4b24a75f0667a1af9ebfe27e94efa7ce97e1/LICENSES/Apache-2.0.txt)
