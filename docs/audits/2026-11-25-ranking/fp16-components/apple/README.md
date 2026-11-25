# Widening FP16 outer products on the M5 Max

This research kernel accepts FP32 inputs, forms binary16 component planes inside each call, and uses Arm Streaming Matrix Extension `FMOPA` to accumulate pairs of binary16 products into binary32 cells. The hardware tests ran on Apple M5 Max, Darwin 27.2.0, with a streaming vector length of 512 bits. Every result tile covers up to 32 rows and 32 columns. The kernel uses four ZA tiles and admits independent leading dimensions, cropped pointers, odd reduction lengths and ragged output edges. [Hardware and operating system](receipts/platform.txt) · [Emitted instructions](receipts/sme-instructions.txt)

The raw mode uses one component product. The corrected mode uses three. Each A row and B column is normalized by an exact power of two, placing its maximum nonzero exponent at 12. Let h be the nearest binary16 input and l the nearest binary16 representation of the residual multiplied by 2¹¹. The corrected result combines the two cross products, scales their sum by 2⁻¹¹, adds the high-component product and restores the row and column scales. Conversion, allocation, packing, all multiplies, rescaling and copying the logical result to C occur within the public call. Nonfinite inputs, exponent spans above 26 and overflowing restored results are refused before modifying C.

On 86 successful public calls, 27,932 output cells were checked against binary64 with 324,324 padding checks. Ordinary signed random inputs had maximum absolute errors of 0.0025732 for raw half and 0.000002125 for three terms. On the separate registered corpus, all 36 ordinary integrated outputs matched the numerical study's common SME frontend bit-for-bit. The three-term implementation still missed its stricter direct-error gate on one same-sign input. [Protected checks](qa/native-m5.json) · [Corpus comparisons](qa/integrated-corpus.csv)

In a separate register-only experiment, matched half- and single-precision instructions produced 2.135 and 2.146 TFLOP/s respectively. Widening FP16 processed two depths per instruction and took about twice as long per instruction on this schedule. The paired FP16/FP32 throughput ratio was 0.9982 across six trials. Input conversion and packing add to the cost of a whole call. [Instruction experiment](ceiling/README.md)

## Reproduce

On an SME-capable M5 Max with Apple's C compiler and Python 3, run:

```sh
bash build.sh build
python3 -B qa/check.py build/apple-fp16.dylib --output replay.json
python3 -B ceiling/summarize.py
```

The public entry points are `apple_fp16_raw` and `apple_fp16_three`. Both return 1 on success and 0 on refusal. They take `(m,n,k,A,lda,B,ldb,C,ldc)` and overwrite the logical output. For a successful positive-dimensional product, `apple_fp16_last_route()` reports 1 or 3 executed component products. The numerically screened single-plane adapter `apple_half_plane` accepts already-converted binary16 operands. The measured library's three compiled objects and full Mach-O binary were independently rebuilt byte-for-byte, using the original library install name for the full binary comparison. [Source and binary identity](receipts/identity.json)
