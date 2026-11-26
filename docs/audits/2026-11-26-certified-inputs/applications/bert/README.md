# Can the checked integer route serve BERT-Tiny inference?

**Zero of 63 dense products qualified, covering zero of 75,055,104 multiply–accumulates (MACs).** Every product had fractional values in both input matrices. The six feed-forward output projections with contraction depth above the signed INT8 dispatcher's shallow bypass accounted for 22,020,096 MACs. All six also failed the input test. These observations cover one pinned FP32 model and three batch-one inputs. They measure eligibility only.

Google identifies this two-layer, width-128 checkpoint as BERT-Tiny, part of its English uncased BERT Miniatures collection. Its card lists Apache-2.0. The three local input files are identified by SHA-256 in the source manifest. [Pinned Google model card](https://huggingface.co/google/bert_uncased_L-2_H-128_A-2/blob/30b0a37ccaaa32f332884b96992754e246e48c5f/README.md) · [Source manifest](model-source.json)

## Which products were observed?

Each input starts with `[CLS]` and ends with `[SEP]`. The intervening tokens repeat and truncate the same fixed English phrase using direct vocabulary lookup. The NumPy forward pass uses the checkpoint's FP32 embeddings, two encoder layers with two width-64 attention heads each, width-512 feed-forward blocks, and the first-token pooler. It runs with token type zero, no padding, and dropout disabled. Embedding lookups, bias additions, normalization, softmax, GELU and pooling activation lie outside the dense MAC count. The trace ends at the pooler and covers inference only.

| Product | Calls per input | Output rows × output columns × contraction depth, for length L |
| --- | ---: | --- |
| Query, key, value and attention output projections | 8 | L × 128 × 128 |
| Attention scores | 4 | L × L × 64 |
| Attention probabilities times values | 4 | L × 64 × L |
| Feed-forward expansion | 2 | L × 512 × 128 |
| Feed-forward output | 2 | L × 128 × 512 |
| Pooler projection | 1 | 1 × 128 × 128 |

| Input tokens | Dense products | Dense MACs | Accepted products | Accepted MACs |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 21 | 3,194,880 | 0 | 0 |
| 32 | 21 | 13,123,584 | 0 | 0 |
| 128 | 21 | 58,736,640 | 0 | 0 |

For a product with dimensions `m,n,k`, the CSV counts `m*n*k` MACs. Its historical `accepts` column tests a broader bounded-integer certificate. Both matrices must contain only finite integers, each magnitude at most 256, with `k <= 4096` and `k * max_abs(A) * max_abs(B) <= 2^24`. Fractional values alone rule out both that certificate and lossless conversion to signed INT8. The signed INT8 dispatcher's separate `k <= 128` bypass applies to 57 of the 63 products. The trace computes these observations from input values without running the native dispatcher. [Per-product observations](bert-products.csv)

The two `*_scaling_witnesses` columns count normal FP32 values with at least one nonzero bit among the lowest 16 fraction bits. After removing all powers of two from such a value's 24-bit significand, its odd part is at least 257. Even a separately chosen power-of-two scale for that single value could not leave an integer of magnitude at most 256. Each of the 63 products contained at least one such witness on **each** side. Approximate quantization and other model checkpoints would require separate measurements.

## Reproduce the observations

From this directory, supply the three local files listed in the manifest and a Python environment with NumPy. The script checks all three digests before inference. It writes per-product counts and prints an aggregate summary with three pooled-vector norms.

```sh
python3 -B trace.py --models /path/to/pinned-model-files --csv reproduced.csv
cmp bert-products.csv reproduced.csv
```

An independent offline rerun of the original trace and a second run of the packaged script each reproduced the deposited CSV byte for byte, including every fractional and scaling-witness count. The full summaries also matched on Python 3.14.10 with NumPy 2.4.4. Different FP32 math libraries can change low bits of activations and their witness counts. The SHA-256 of the deposited CSV is `ac15e6bd9f809da1a401240d337af55bb1c4598928819eaf4cfd170903fe9764`. [Offline trace](trace.py)
