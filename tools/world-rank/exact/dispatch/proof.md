# An exact domain for single-product BF16 matrix multiplication

For a positive row-major `M × N × K` overwrite multiplication, let `a` and `b` be the largest absolute values in the logical A and B views. The dispatcher admits a native BF16 product when all entries are integers in `[−256,256]`, `K ≤ 4096`, and `K a b ≤ 16,777,216`. Both zero encodings are accepted. The guarantee concerns the numerical value of an FP32 output cell, including zero. The sign bit of a zero result is outside the guarantee.

## Proof

1. A normal BF16 number has eight significant binary digits. Every integer from −256 through +256 is exactly representable. In particular, 256 itself requires just one significant bit. Rounding an admitted FP32 input to nearest BF16 therefore preserves its value.
2. For one cell, each of the `K` products is an integer with magnitude at most `a b`. For any subset of the terms, the absolute value of its mathematical sum is at most the sum of its absolute products, which is at most `K a b`. This applies to pair sums, even/odd lanes, tile accumulators and combinations of partial sums, independent of reduction order and cancellation. Zero-padded depth positions add zero.
3. Binary32 represents every integer in the closed interval `[−2²⁴,2²⁴]`. By induction, every intermediate product and sum in the ordinary dot-product reduction is already representable. Rounding returns the same value. All nonzero intermediates have magnitude at least one and remain far from underflow. The output of the single classical BF16 dot product with binary32 accumulation is consequently the exact integer sum. This proof assumes partials combine disjoint original terms with no scaling or algebraic transformation.

Intel documents two FP32 partial sums and their final addition into the tile accumulator for `TDPBF16PS`. All three additions fall inside this bound. [Intel instruction reference, TDPBF16PS](https://cdrdv2-public.intel.com/774492/325383-sdm-vol-2abcd.pdf)

Arm describes `BFMMLA` as pairs of `BFDOT` operations accumulating into FP32. The original architecture uses round-to-odd on *inexact* results and flush-to-zero for subnormals. Neither changes the integer operations proved above. Extended BF16 fused evaluation also gives an exactly representable result. [Arm matrix instruction description](https://developer.arm.com/community/arm-community-blogs/b/ai-blog/posts/bfloat16-processing-for-neural-networks-on-armv8_2d00_a), [Arm BFDotAdd pseudocode](https://documentation-service.arm.com/static/6526e1bd9e189a266cef8412)

At the inclusive boundary, 256 terms of `256 × 256` sum to exactly 16,777,216. A depth of 257 with the same maxima exceeds the bound and goes to FP32. The guard is sufficient and conservative. It uses one maximum per entire input and can reject matrices whose individual products or dot products would all be exact. The fixed depth cap restricts the registered experiment further. At depth 4096 all integers between −63 and +63 pass, since `4096 × 63² = 16,257,024`. Magnitude 64 on both sides also passes exactly at the boundary.

## Dispatch evidence

`dispatch.c` accepts the existing `rank_vendor_init` and `rank_vendor_sgemm` ABI. The public operation assumes valid, nonoverlapping output and input views, ordinary row-major strides in elements, and overwrite semantics. The memory holding the inputs must remain stable during the call. Each worker process selects its two DSO paths with `EXACT_FALLBACK_PLUGIN` and `EXACT_FAST_PLUGIN`. The loader checks the actual defining objects of the two public entry points and prevents self-recursion. `EXACT_NATIVE_ENABLED=1` is supplied only after the external harness verifies the real instruction route on this host and underlying geometry. Initialization and paths are per process. Measurements on different certified geometries start fresh workers.

On a hardware-enabled, well-formed call with `1 ≤ K ≤ 4096`, the wrapper scans the logical views in A-then-B order. It checks up to 64 contiguous elements per chunk, ending each chunk at a row boundary and excluding all padding. A failed chunk ends the scan immediately. For example, a fractional first element of A costs `min(64,K)` inspections and leaves B untouched. A failing final B element costs `M K + K N` inspections. It inspects the unsigned binary32 representation so that probing a signaling NaN does not itself perform floating-point arithmetic. For an absolute value from 1 through 256 and stored exponent `E`, integrality requires the low `150−E` fraction bits to be zero. A nonzero value below one and any word above the encoding of 256 are rejected. An operand with any rejected word has observed maximum `−1`. A skipped operand also returns `−1`. A completed operand retains its maximum. Inspected words and completed maxima are recomputed on every call, even when the pointers are reused. The bound multiplies in `uint64_t`.

A successful fast dispatch calls the existing mixed1 operation once and observes one native plan and one completed component product, with a nonempty implementation name. The underlying implementation includes FP32-to-BF16 conversion and allocation in that public call. When hardware or the verified fast plugin is unavailable, the wrapper forwards the same pointers, dimensions and strides directly to the ordinary FP32 provider. An observed mismatch after the fast call also recomputes the complete output with the ordinary provider. The fallback assumes the selected ordinary provider's original numerical behavior. Its choice and source identity belong in the host record. The wrapper requires a valid ordinary provider before serving requests.

| Route | Reasons | Input observations |
| --- | --- | --- |
| 1, native | `native-exact` | Both maxima and exactly `M K + K N` logical words |
| 2, FP32 | `integer-domain` | All completed chunks through the first failing chunk, with the rejected operand and any skipped operand marked −1 |
| 2, FP32 | `accumulator-bound` | Both matrices scanned and both maxima recorded |
| 2, FP32 | `depth-outside`, `empty-or-invalid-view` | Zero scanned, both maxima −1 |
| 3, FP32 | `native-disabled`, `fast-plugin-unavailable`, `fast-native-unverified` | Zero scanned if rejected at initialization, or both matrices scanned if post-call metadata failed |

`exact_last_route`, `exact_last_reason`, `exact_last_inspected`, `exact_last_amax`, `exact_last_bmax`, `exact_last_fast_calls`, `exact_last_fallback_calls`, `exact_last_products`, `exact_last_native` and `exact_last_impl` expose the last call on the current thread. `exact_last_inspected` returns `unsigned long long`, `reason` and `impl` return `const char *`, and the other accessors return `int`.

## Portable regression

Run `python3 smoke.py` with a C11 compiler. It checks 116,571 binary32 encodings against an independent Python classifier, including all 65,536 bit patterns with zero low 16 bits and values one FP32 word to either side of every small integer. Six fresh-process configurations exercise 78 public calls with cropped, independently padded views, dirty C, first-chunk, second-chunk and final-row fractional entries, NaN, infinity, signed zero, exact-limit and over-limit reductions. Test-only DSOs verify the ABI, counts and fallback bytes. Native hardware admission and performance use the actual DSOs and their independent per-geometry instruction receipts.
