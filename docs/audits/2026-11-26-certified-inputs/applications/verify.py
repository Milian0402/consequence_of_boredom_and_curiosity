"""Check the recorded aggregate observations offline with Python's standard library."""
import csv
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
DIGESTS = {
    "corpus/coverage.csv": "6f0b7f28de6e0348f93980c4c1ba4eb430d9163fc60db93d3c58ee86c35f6cc6",
    "corpus/digit-classifier.json": "c10e11b97f1976427cf34302b25af129397e09d6a50de588b6d4b9668c565bf8",
    "bert/bert-products.csv": "ac15e6bd9f809da1a401240d337af55bb1c4598928819eaf4cfd170903fe9764",
}
PANEL = {
    "optdigits_counts": (1797, 3823, 64),
    "optdigits_unit_intensity": (1797, 3823, 64),
    "optdigits_original_binary": (1797, 1934, 1024),
    "movielens_user_raw_crossproducts": (943, 943, 1682),
    "movielens_user_centered_crossproducts": (943, 943, 1682),
    "har_inertial_linear_kernel": (2947, 7352, 561),
    "wine_red_covariance_numerator": (11, 11, 1599),
    "wine_white_covariance_numerator": (11, 11, 4898),
}


def check(ok, why):
    if not ok:
        raise ValueError(why)


def rows(name):
    with (ROOT / name).open(newline="") as source:
        return list(csv.DictReader(source))


def json_file(name):
    return json.loads((ROOT / name).read_bytes())


def corpus():
    observed = rows("corpus/coverage.csv")
    check(len(observed) == len(PANEL) and {r["workload"] for r in observed} == set(PANEL), "corpus panel")
    hits = [0, 0, 0, 0]
    for r in observed:
        dims = tuple(int(r[key]) for key in ("m", "n", "k"))
        name = r["workload"]
        check(PANEL[name] == dims, f"{name} shape")
        k = dims[2]
        maximum = [float(r["max_abs_"+tag]) for tag in "ab"]
        powers = [int(r["grid_exponent_"+tag]) for tag in "ab"]
        widened = [math.ldexp(v, -e) for v, e in zip(maximum, powers)]
        check(all(v.is_integer() and v >= 0 for v in widened), f"{name} recovered dyadic integers")
        integer = all(int(r["all_integer_"+tag]) == 1 for tag in "ab")
        mag_bound = k * maximum[0] * maximum[1] <= 2**24
        strict = integer and k <= 4096 and max(maximum) <= 63
        bounded = integer and max(maximum) <= 256 and mag_bound
        dyadic = max(widened) <= 256 and k*int(widened[0])*int(widened[1]) <= 2**24 and -126 <= sum(powers) <= 103
        accepted = [strict, bounded, dyadic]
        recorded = [int(r[key]) for key in ("strict_integer", "bounded_integer", "scaled_dyadic")]
        check(accepted == recorded, f"{name} certificate")
        # All candidates near the signed byte boundary in this panel are fractional;
        # every admitted integral case has magnitude <=63, so sign is immaterial here.
        signed = integer and 128 < k <= 4096 and max(maximum) <= 127 and mag_bound
        for i, yes in enumerate([*accepted, signed]):
            hits[i] += int(yes)
        for tag in "ab":
            check(bool(re.fullmatch(r"[0-9a-f]{64}", r["sha256_"+tag+"_f32le"])), f"{name} input digest")
            for field in ("bf16_exact_fraction_", "nonzero_fraction_"):
                check(0 <= float(r[field+tag]) <= 1, f"{name} input fraction")
    check(hits == [3, 3, 4, 2], "corpus certificate totals")

    ml = json_file("corpus/movielens-work.json")
    check((ml["users"], ml["movies"], ml["ratings"]) == (943, 1682, 100000), "MovieLens counts")
    check(ml["dense_terms"] == ml["users"]**2*ml["movies"] and ml["sparse_ordered_terms"] == 16807190,
          "MovieLens Gram product term counts")
    classifiers = json_file("corpus/digit-classifier.json")
    expected = [("pixel_count_features", 3823, 64, 1761, "ec47b13672085cc2096a990dd166c76e2e29843c7f79ea88d1d483db94b05db0"),
                ("original_bitmaps", 1934, 1024, 1731, "eec353b47234bfe0fa3544331dc16aceee6a9b1ee3b7165b51e30fe565a9e825")]
    check(len(classifiers) == 2, "classifier count")
    for r, values in zip(classifiers, expected):
        check(r["m"] == r["test_records"] == 1797 and tuple(r[key] for key in
              ("workload", "n", "k", "recognized", "f32_row_major_product_sha256")) == values, "classifier result")


def bert():
    evidence = rows("bert/bert-products.csv")
    expected = {}
    for t in (8, 32, 128):
        expected[t, "bert.pooler.dense"] = (1, 128, 128)
        for layer in range(2):
            prefix = f"bert.encoder.layer.{layer}."
            for role in ("attention.self.query", "attention.self.key", "attention.self.value", "attention.output.dense"):
                expected[t, prefix+role] = (t, 128, 128)
            for head in range(2):
                expected[t, prefix+f"head{head}.scores"] = (t, t, 64)
                expected[t, prefix+f"head{head}.values"] = (t, 64, t)
            expected[t, prefix+"intermediate.dense"] = (t, 512, 128)
            expected[t, prefix+"output.dense"] = (t, 128, 512)
    check(len(evidence) == len(expected) == 63, "encoder call count")
    macs_by_length = {8: 0, 32: 0, 128: 0}
    deeper_calls = deeper_macs = 0
    for r in evidence:
        key = int(r["tokens"]), r["product"]
        m,n,k = (int(r[d]) for d in ("m","n","k"))
        check(expected.pop(key, None) == (m,n,k), f"{key} tensor dimensions")
        check(int(r["macs"]) == m*n*k, f"{key} dense MAC count")
        fractional = (int(r["a_fractional"]),int(r["b_fractional"]))
        witnesses = (int(r["a_scaling_witnesses"]),int(r["b_scaling_witnesses"]))
        for counts in (fractional, witnesses):
            check(0 < counts[0] <= m*k and 0 < counts[1] <= k*n, f"{key} counterexample on each operand")
        check(int(r["accepts"]) == 0, f"{key} eligibility")
        macs_by_length[key[0]] += m*n*k
        if k > 128:
            deeper_calls += 1
            deeper_macs += m*n*k
    check(not expected and macs_by_length == {8: 3194880, 32: 13123584, 128: 58736640} and
          sum(macs_by_length.values()) == 75055104, "encoder MACs")
    check(deeper_calls == 6 and deeper_macs == 22020096, "encoder calls above shallow depth")


if __name__ == "__main__":
    for name, digest in DIGESTS.items():
        check(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, f"{name} recorded digest")
    corpus()
    bert()
    print("Verified eight corpus products (3 strict, 3 bounded, 4 scaled, 2 signed after depth selection), "
          "two classifiers, MovieLens term counts, and 63 rejected BERT products totaling 75,055,104 MACs.")
