# Fused six-product AMX-BF16 SGEMM

This research kernel accepts row-major FP32 inputs and overwrites a strided FP32 output. A complete call allocates scratch, splits and packs each input, executes the tile products, restores the output and frees scratch. It retains no operand packs or plans between calls.

Each eligible scalar is decomposed into three BF16 values by rounding the input to nearest-even, subtracting that value and repeating on the residual. Nonzero input magnitudes must lie in the inclusive range [2⁻³⁰,2³⁰]. Each A row tile has 16 rows and 32 depth values. Each B tile interleaves adjacent depth pairs for 16 output columns. All three parts are packed into separate contiguous planes, including zero-padded tails. A vectorized converter runs on supported AVX-512 BF16 hosts. The scalar converter implements the same layout.

One 16×16 output tile uses three distinct FP32 AMX registers across the full reduction. The hardware dot products accumulate these degrees separately.

| Register | Terms accumulated |
|---|---|
| 0 | A-high × B-high |
| 1 | A-high × B-mid, A-mid × B-high |
| 2 | A-high × B-low, A-mid × B-mid, A-low × B-high |

The kernel adds register 2 to register 1, then adds register 0. Each 32-deep step executes six TDPBF16PS instructions, using six tile loads. The three omitted products are A-mid × B-low, A-low × B-mid and A-low × B-low. This is an approximation to an FP32-input product. Accuracy depends on input values and conditioning.

## Building and checking

Compile `amx_fused6.c` and `amx_compute.c` as separate C11 translation units at `-O3` without fast-math. No global ISA selection is needed. Link the existing `precision_range_rescue` from the companion precision core. The complete call uses that binary64 dot-product fallback for exceptional inputs or scratch exhaustion. Padded C entries are preserved.

Run `bash tests/check.sh` to build and check a temporary shared library, the dedicated benchmark ABI and a separate trace version. A host without AMX still checks component conversion, layout and reference arithmetic and explicitly skips the hardware checks. The ordinary precision suite can instead link both sources into its existing binary and call the namespaced entry points.

| API | Meaning |
|---|---|
| `amx_fused6_init()` | Zero when CPU AMX-TILE/BF16 and Linux extended-state support and permissions are present. Call from the thread doing the multiplication. |
| `amx_fused6_sgemm(m,n,k,a,lda,b,ldb,c,ldc)` | Zero after a completed call, −2 for unavailable native hardware and −1 for invalid arguments. Empty outputs and zero K are handled without AMX. |
| `amx_fused6_last_route()` | 1 for completed native tile computation, 2 for the range rescue and 0 for an empty, invalid or unsupported call. |
| `amx_fused6_last_products()` | Six component products for the native route, zero otherwise. |
| `amx_fused6_last_reason()` | Per-thread explanation for the most recent call. |
| `cob_amx_six_trace(i)` | In a separately compiled `-DCOB_AMX_TRACE=1` binary, per-thread cumulative counts for output tiles (0), depth blocks (1), and hardware dot instructions (2). Compare the before and after values. |

A dedicated DSO can enable `-DAMX_FUSED6_BENCH_ABI=1` to export `precision_sgemm` mode 6 and the matching getters. The fused loop creates zero vendor plans.
