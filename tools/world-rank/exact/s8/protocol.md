# Checked signed INT8 confirmation

The experimental C entry accepts ordinary, disjoint, strided, row-major FP32 input and overwrites FP32 output. Its last argument is the caller's normal SGEMM callback. The standalone source and header are `s8/checked-int8.cpp` and `s8/checked-int8.h`. Link the object with `-ldl -pthread`, use the pinned oneDNN 3.14.1 sequential SDK headers and provide the runtime through the dynamic loader or `EXACT_INT8_DNNL_PATH`. Compile the adapter at the architecture baseline with C++17, -O3 and -fno-fast-math. The hosted ordinary adapter uses no -march=native requirement. CPU and arithmetic selection occur within the real public call. No data, packed buffer, primitive, validation or per-shape permission is retained across calls. A per-thread engine and loaded library functions may persist. The experiment requires a valid callback. This adapter ships only on the separate research branch.

K ≤ 128 bypasses immediately on both architectures. K > 4096 also bypasses. Eligible inputs must contain exact binary32 integers in [−128,127], with K × max|A| × max|B| ≤ 2²⁴. A bitwise validator tests and packs each signed byte in a single pass, inspecting at most 64 logical values beyond the first failure. Every possible true partial sum fits s32 and the final conversion is exact binary32. An eligible x86 call needs AMX INT8 and usable tile state. An eligible Arm call needs SVE I8MM. A compatible executed oneDNN plan and version are required. The default path has no pre-probed shape table. The independent audit verifies the real generated arithmetic kernel and exact outputs separately for every tested geometry. Hardware, loader, domain, workspace or primitive refusal invokes ordinary FP32. The accelerated result uses a private s32 output buffer until successful completion. The proof concerns finite integer inputs, with positive and negative zero numerically equivalent.

## Fixed panel and comparisons

One freely allocated native Linux x86 and one Arm runner are accepted regardless of model. The 4096³ anchor and fresh 3584³ square use independent signed random integers in [−63,63]. The existing cropped 1079×1553×1201 rectangle and early or late half-integer rejections check strides and fallback cost. The two original UCI digit matrices measure the short-depth FP32 selector (64) and true binary pixels (1024). Seeds 2341279043 and 916583729 select fresh synthetic values. Each real corpus is unchanged between scheduling blocks. The registered algorithms, dimensions and rules are fixed before timing.

The ordinary callback in the ranking wrapper uses released two-level COB with sequential oneMKL on x86 or FastMMW-1536 on Arm when all dimensions are at least 4096 and the aspect ratio is at most two. Its other shapes use oneMKL's JIT-capable SGEMM or ArmPL. All prior registered FP32 implementations are measured separately, including the ordinary large baseline on the fresh square. The earlier checked BF16 method, raw BF16 reference, pre-existing signed INT8 specialist, and a fresh packed AND/popcount implementation for the bitmap form separately named fields. A complete comparison counts per-call scans, input conversion, primitive creation, allocation, packing, multiplication and final stores. Ordinary DSO initialization and the selected callback setup precede timing.

Use two seeds × three matched hardware rounds, each timing a fresh-process first call and the immediately subsequent warm call. One pinned OS thread runs every implementation with protected input parents, poisoned C, surrounding output guards and independent full FP64 and scaled-integer reference checks. Previous UCI output digests must reproduce exactly. The two rejection controls and shallow bypass must equal the separately called matching fallback bit-for-bit. Report source and binary digests, direct and traced native identity and all failed peers. Report per-shape median ratios, all pair outcomes, first and warm timings, the fastest qualifying FP32 and specialist. The scoped improvement gate is a median above 1.02× and at least five wins in six. A large effect on the fresh 3584³ shape requires at least 2× over the strongest qualified ordinary FP32 under both paired and separate warm medians, five or more pair wins, and exact output. Both host results are published. Performance on unrestricted floating inputs and other machines is outside this test.

## Use the research entry from C

With the frozen sequential oneDNN headers installed, build the library from this directory.

```sh
c++ -std=c++17 -O3 -fno-fast-math -fPIC -shared \
  -I"$DNNL_PREFIX/include" checked-int8.cpp -ldl -pthread -o libchecked_int8.so
export EXACT_INT8_DNNL_PATH="$DNNL_PREFIX/lib/libdnnl.so.3"
```

Include `checked-int8.h` and link the new library. Calling code supplies an ordinary single-threaded row-major FP32 fallback, such as an existing SGEMM wrapper with beta zero.

```c
int route = exact_int8_checked_sgemm(m, n, k, A, lda, B, ldb, C, ldc, ordinary_sgemm);
```

Return value 1 confirms a completed signed INT8 product, 2 reports that a domain or depth check selected ordinary SGEMM, and 3 reports an unavailable accelerated operation. Both fallback routes have already run ordinary SGEMM. A missing callback returns −1. Production callers own valid buffer lifetimes and disjoint output just as for ordinary SGEMM. A caller using only this typed API can omit the benchmark adapter environment variables.
