# Local exact-dispatch checks

The final guard passed all 17 cases on a physical AArch64 host in an Arm LinuxKit guest. The native path executed one ACL 53.3.2 `a64_interleaved_bf16fp32_mmla_8x12` multiply on each of the eight admitted cases. All finite outputs agreed exactly with whole-matrix double GEMM. For the two genuine UCI matrices, both the candidate and the narrowed double reference also reproduced independent FP32 output hashes. The serial ordinary provider and DGEMM reference used OpenBLAS 0.3.34 VORTEX. On the same host, translated Linux x86-64 passed 11 smaller controls with native dispatch disabled and OpenBLAS 0.3.34 NEHALEM. Both adapters reported one thread. All checks were untimed.

The fractional 1079×1553×1201 cases used identical signed integers except for one half-integer. The first A value triggered fallback after the first 64 values. The last B value required scanning all 3,161,032 values. Both calls matched the separate ordinary call bit-for-bit on all 1,675,687 outputs, with zero error against the double reference.

The checker protects both padded, cropped input parents read-only. It compares the full output and checks surrounding sentinel bits. NaN and infinity use bit-for-bit ordinary-path parity. Their absolute error field is empty in the final tables. The `integer_spots` column counts additional exact checks of both signed and absolute-product references. The output SHA-256s use logical row-major little-endian FP32. Route codes 1, 2 and 3 mean native, content/depth rejection, and native unavailable, respectively. A maximum of −1 means the corresponding input did not complete validation.

[Final Arm](arm.csv) · [Final x86-64](x86.csv) · [Arm source and provider identities](aarch64-identity.txt) · [x86-64 source and provider identities](x86_64-identity.txt)

The [initial Arm](initial-arm.csv) and [initial x86-64](initial-x86.csv) records preserve a preliminary version that finished a full logical scan after a bad value. Their headerless columns are case, M, N, K, route, reason, inspected, products, integer spots, max absolute error, implementation. The preliminary numeric field for NaN and infinity was initialized to zero and has no error interpretation. The initial invocation did not capture the compiled dispatch digest. The final run captured its sources and libraries when building.

To reproduce the final local check on Linux, set `EXACT_FALLBACK_PLUGIN` to a single-thread OpenBLAS plugin exposing both row-major SGEMM and DGEMM. For native Arm, set `EXACT_NATIVE_ENABLED=1` and `EXACT_FAST_PLUGIN` to the registered mode-1 mixed precision plugin linked to ACL. On x86 without a verified backend, set `EXACT_NATIVE_ENABLED=0` and select `small`. Resolve the shared-library paths before entering the fixture directory. The checker took seed `0x84fd2117` and used one logical CPU.

```sh
cc -O3 -std=c11 -Wall -Wextra -Werror -fno-fast-math plugin_check.c exact_fixture.c -ldl -lcrypto -lm -o check
cc -O3 -std=c11 -shared -fPIC ../dispatch/dispatch.c -pthread -ldl -o dispatch.so
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c 0 ./check ./dispatch.so "$EXACT_FALLBACK_PLUGIN" ./data full
```
