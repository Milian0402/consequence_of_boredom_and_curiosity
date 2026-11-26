# Which application inputs qualify?

Three of eight deliberately chosen matrix products satisfy the original bounded-integer BF16 condition. Two satisfy the input and depth checks of the later signed INT8 selector. In a separate FP32 trace of Google's two-layer small BERT model, all 63 dense products contained a value that prevented exact integer conversion. These observations identify examples where an ordinary FP32 entry can take a certified route. They do not estimate how often applications supply such inputs. [Corpus observations](corpus/coverage.csv) · [Encoder observations](bert/bert-products.csv)

| Operation on the supplied FP32 values | M × N × K | BF16 integer | BF16 with exact power-of-two scaling | Signed INT8 after depth check |
|---|---:|:---:|:---:|:---:|
| Handwriting, 64 pixel-count features, test by training | 1797 × 3823 × 64 | Yes | Yes | Shallow bypass |
| Same counts divided by 16 | 1797 × 3823 × 64 | No | Yes | Shallow bypass |
| Handwriting, original 32×32 binary images, test by training | 1797 × 1934 × 1024 | Yes | Yes | Yes |
| MovieLens, user-by-user rating dot products | 943 × 943 × 1682 | Yes | Yes | Yes |
| MovieLens, observed ratings centered per user | 943 × 943 × 1682 | No | No | No |
| Smartphone activity, normalized test-by-training sensor features | 2947 × 7352 × 561 | No | No | No |
| Red wine, covariance numerator for centered measured features | 11 × 11 × 1599 | No | No | No |
| White wine, same feature operation | 11 × 11 × 4898 | No | No | Depth exceeds 4096 |

The BF16 integer column gives the same decisions for two rules. The original requires integral FP32 values of magnitude at most 63 and K ≤ 4,096. The wider arithmetic rule permits magnitude 256 when K × max|A| × max|B| ≤ 2²⁴. Every eligible integer and every partial product sum is exactly representable. The measured BF16 dispatcher also caps K at 4,096. The experimental scaling column factors one exact power of two from all of A and another from all of B. The scaled integers must satisfy the wider bound, and the summed exponent must lie between −126 and 103. This restores a finite, normal FP32 result whenever the integer dot is nonzero. The native signed INT8 selector accepts signed bytes, applies the same sum bound, and bypasses at K ≤ 128 or K > 4,096. Its entries above describe data and shape eligibility. Actual execution additionally depends on the provider and instruction set.

## Why the representation matters

The two handwriting products feed an actual one-nearest-neighbor classifier. Squared Euclidean distances come from the integer test-by-training dot products and the two row norms. The first training index wins an exact distance tie. Pixel counts recognized **1,761 of 1,797** test digits (98.00%). Original bitmaps recognized **1,731 of 1,797** (96.33%). The training partitions differ between representations. The independent FP32 output fingerprints also bind the two Linux benchmark fixtures to these exact products. [Classifier records](corpus/digit-classifier.json) · [Classifier replay](corpus/digit_knn.py)

For MovieLens, unobserved ratings contribute zero, and centering subtracts only the mean of a user's observed ratings. The original matrix contains 100,000 ratings in 943 × 1,682 cells (6.30%). A dense user-by-user Gram matrix uses 1,495,716,818 multiply-accumulate terms. Grouping ratings by movie yields 16,807,190 co-rated ordered user-pair contributions, including self-pairs. The 89-to-1 ratio counts terms. Elapsed time for a sparse application remains unmeasured. [Term counts](corpus/movielens-work.json) · [Sparse count replay](corpus/movielens_work.py)

For the smartphone features, only 1.23% of test values and 1.34% of training values were represented exactly in BF16. Centering each wine feature in binary64 before casting to FP32 left zero BF16-exact entries in both wine matrices. For the BERT trace, each call had at least one finite normal FP32 operand value with more than eight significant binary digits. A power-of-two factor, even chosen per value, cannot turn it into an exact integer with magnitude ≤256. The 63 observed products accounted for 75,055,104 dense multiply-accumulates. The experiment measures applicability on three specified inputs, with no inference speed or accuracy comparison. [Encoder trace and reproduction](bert/README.md)

## Replay

Run `python3 -B verify.py` in this directory. This standard-library check validates the observation fingerprints, reconstructs the certificate decisions from the recorded summaries, checks each encoder product shape and MAC count, and verifies the classifier counts and exact output fingerprints. To regenerate the observations themselves, use Python with NumPy and the system `gzip` utility. The optional download fetches the four original archives from their publishers and verifies their SHA-256 hashes.

```sh
python3 -B corpus/audit.py --data ./downloads --fetch --output ./coverage-replayed.csv
python3 -B corpus/digit_knn.py ./downloads/optdigits.zip
python3 -B corpus/movielens_work.py ./downloads/ml-100k.zip
```

## Sources and attribution

The application audit and replay scripts use the project’s MIT license. The original datasets and pretrained model are obtained separately under their respective publisher terms. [Script license](LICENSE) · [Model provenance](bert/model-source.json)

- E. Alpaydin and C. Kaynak, *Optical Recognition of Handwritten Digits* (1998). The 64-feature product uses the 1,797 test and 3,823 training records. The 1,024-pixel product uses all 1,797 original writer-independent images and 1,934 original training images. Matrix multiplication omits the class label. Classification uses the original labels. UCI distributes the dataset under CC BY 4.0. [Dataset](https://doi.org/10.24432/C50P49) · [License](https://creativecommons.org/licenses/by/4.0/)
- F. Maxwell Harper and Joseph A. Konstan, *The MovieLens Datasets: History and Context* (2015). The MovieLens 100K dataset was collected by the GroupLens Research Project at the University of Minnesota. Its terms cover research use and require separate permission for commercial use or redistribution. This package provides aggregate counts and fingerprints. [Paper](https://doi.org/10.1145/2827872) · [Original dataset and usage terms](https://files.grouplens.org/datasets/movielens/ml-100k-README.txt)
- J. Reyes-Ortiz, D. Anguita, A. Ghio, L. Oneto and X. Parra, *Human Activity Recognition Using Smartphones* (2013). The product pairs the published test and training feature vectors across all 561 normalized features. UCI distributes the dataset under CC BY 4.0. [Dataset](https://doi.org/10.24432/C54S4K) · [License](https://creativecommons.org/licenses/by/4.0/)
- P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis, *Wine Quality* (2009). Products use the 11 physicochemical measurements, omit the quality label, and center features separately within each wine type. UCI distributes the dataset under CC BY 4.0. [Dataset](https://doi.org/10.24432/C56S3T) · [License](https://creativecommons.org/licenses/by/4.0/)
