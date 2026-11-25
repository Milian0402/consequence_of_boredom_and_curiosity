# Eight-column prefix experiment

This kit builds three ordinary single-threaded row-major FP32 wrappers for the pinned sequential LP64 oneMKL adapter. Each worker initializes its provider once using the absolute path in `RANK_PREFIX_MKL`. The oneMKL adapter validates the resolved SGEMM library and its one-thread setting. All operands belong to the caller. The timed route needs no scratch and retains no packed operand or execution plan.

| Method | Complete product on an eligible view |
| --- | --- |
| host-forward | One call to the original oneMKL adapter for all columns |
| host-split8 | One oneMKL call for the first eight columns, then one oneMKL call for the rest |
| host-prefix8 | A four-row, eight-column AVX2/FMA loop computes the first eight columns, then one oneMKL call computes the rest |

The last two methods split only on genuine Intel CPUs with AVX2 and FMA when B begins on a 64-byte boundary and the input geometry passes the scope check. The scope is 24–160 rows, 1024–4096 columns, depth 96–384, at least eight times as many columns as rows, and at most 49,152 entries in A. All other supported views forward unchanged to the same oneMKL adapter. Eligible bodies read B and write C eight floats past the start of their respective views and use the three original leading dimensions. This also changes the body's column count and which kernel computes the first eight columns.

Run `bash build.sh VENDOR_DIR BUILD_DIR` on Linux x86-64. Export `RANK_PREFIX_MKL` as the absolute path to `VENDOR_DIR/mkl.so` for timing. Run `taskset -c CPU python3 routes.py PANEL.csv VENDOR_DIR BUILD_DIR OUT` for separate route checks at each registered layout. The route checker loads the ordinary plugin against the original provider and separately runs the diagnostic plugin through a witness at the provider boundary. The witness records the actual forwarded addresses, leading dimensions, extents and call counts and invokes the same original provider. The independent FP64 oracle checks each complete output, read-only mappings protect A and B, and sentinel values cover C outside the view. The build also verifies the absence of explicit copy and allocator imports in each ordinary wrapper, alongside source review of the timed path.

For local qualification with a stand-in FP32 provider, run `bash local-test.sh OUT`. The fixture covers all four row remainders, nonzero cropped offsets, unequal leading dimensions and bypass paths. Those results describe correctness only.
