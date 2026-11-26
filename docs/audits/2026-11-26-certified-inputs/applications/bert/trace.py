"""Audit all dense products in BERT-Tiny encoder inference using its pinned FP32 weights."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np


FILES = {
    "model.safetensors": "7fb69ad9f6866d8983183c930e33828f326470bf6ad8bbb2ad4ed957a92e9414",
    "config.json": "508e1f01aae55d73355cbdd82609be2f43ba5a0d3428837adbe56cf8391f8b39",
    "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
}


def weights_from_file(path):
    raw = path.read_bytes()
    offset = struct.unpack("<Q", raw[:8])[0] + 8
    headers = json.loads(raw[8:offset])
    result = {}
    for name, spec in headers.items():
        if name == "__metadata__":
            continue
        assert spec["dtype"] == "F32"
        start, end = spec["data_offsets"]
        result[name] = np.frombuffer(raw, dtype="<f4", count=(end-start)//4,
                                     offset=offset+start).reshape(spec["shape"])
    return result


def audit_operand(x):
    assert x.dtype == np.float32 and np.isfinite(x).all()
    # A normal FP32 value with more than eight significant bits cannot be
    # s * 2**e for any integer |s| <= 256, even with a separate e per value.
    bits = x.view(np.uint32)
    exponent = (bits >> 23) & 255
    witnesses = np.count_nonzero((exponent > 0) & (exponent < 255) & ((bits & 65535) != 0))
    return np.max(np.abs(x)), np.equal(x, np.trunc(x)).all(), int(witnesses)


def forward(tokens, weights, config, rows):
    def matmul(name, a, b):
        m, k = a.shape
        n = b.shape[1]
        amax, a_integers, a_witnesses = audit_operand(a)
        bmax, b_integers, b_witnesses = audit_operand(b)
        eligible = (k <= 4096 and a_integers and b_integers and
                    max(amax, bmax) <= 256 and k * float(amax) * float(bmax) <= 2**24)
        rows.append(dict(tokens=len(tokens), product=name, m=m, n=n, k=k,
                         macs=m*n*k, accepts=int(eligible),
                         a_fractional=int(np.count_nonzero(a != np.trunc(a))),
                         b_fractional=int(np.count_nonzero(b != np.trunc(b))),
                         a_scaling_witnesses=a_witnesses, b_scaling_witnesses=b_witnesses))
        return a @ b

    def dense(name, x):
        return matmul(name, x, weights[name + ".weight"].T) + weights[name + ".bias"]

    def norm(name, x):
        centered = x - x.mean(axis=-1, keepdims=True)
        scale = np.sqrt((centered**2).mean(axis=-1, keepdims=True) + config["layer_norm_eps"])
        return centered / scale * weights[name + ".weight"] + weights[name + ".bias"]

    prefix = "bert.embeddings."
    x = norm(prefix + "LayerNorm", weights[prefix + "word_embeddings.weight"][tokens] +
             weights[prefix + "position_embeddings.weight"][:len(tokens)] +
             weights[prefix + "token_type_embeddings.weight"][0])
    for layer in range(config["num_hidden_layers"]):
        prefix = f"bert.encoder.layer.{layer}."
        q, k, v = (dense(prefix + "attention.self." + role, x) for role in ("query", "key", "value"))
        heads = []
        for h in range(config["num_attention_heads"]):
            width = config["hidden_size"] // config["num_attention_heads"]
            part = slice(h * width, (h + 1) * width)
            scores = matmul(f"{prefix}head{h}.scores", q[:, part], k[:, part].T) / np.float32(math.sqrt(width))
            prob = np.exp(scores - scores.max(axis=1, keepdims=True))
            prob /= prob.sum(axis=1, keepdims=True)
            heads.append(matmul(f"{prefix}head{h}.values", prob, v[:, part]))
        x = norm(prefix + "attention.output.LayerNorm",
                 x + dense(prefix + "attention.output.dense", np.concatenate(heads, axis=1)))
        hidden = dense(prefix + "intermediate.dense", x)
        erf = np.vectorize(math.erf, otypes=[np.float32])
        hidden = np.float32(0.5) * hidden * (1 + erf(hidden / np.float32(math.sqrt(2))))
        x = norm(prefix + "output.LayerNorm", x + dense(prefix + "output.dense", hidden))
    pooled = np.tanh(dense("bert.pooler.dense", x[:1]))
    assert np.isfinite(pooled).all()
    return float(np.linalg.norm(pooled))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True, help="directory containing the three pinned input files")
    parser.add_argument("--csv", type=Path, required=True, help="path for the regenerated observations")
    args = parser.parse_args()
    for name, digest in FILES.items():
        path = args.models / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Unexpected SHA256 for {name}")
    weights = weights_from_file(args.models / "model.safetensors")
    config = json.loads((args.models / "config.json").read_text())
    vocab = {word: i for i, word in enumerate((args.models / "vocab.txt").read_text().splitlines())}
    words = "the library measured the time to read and write each part of the large table".split()
    rows = []
    norms = {}
    for length in (8, 32, 128):
        sequence = ["[CLS]"] + [words[i % len(words)] for i in range(length - 2)] + ["[SEP]"]
        norms[length] = forward([vocab[word] for word in sequence], weights, config, rows)
    with args.csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    macs = sum(row["macs"] for row in rows)
    print(json.dumps(dict(calls=len(rows), accepted=sum(row["accepts"] for row in rows), macs=macs,
                          accepted_macs=sum(row["macs"]*row["accepts"] for row in rows),
                          impossible_even_after_dyadic_rescaling=sum(
                              row["a_scaling_witnesses"] + row["b_scaling_witnesses"] > 0 for row in rows),
                          pooled_norms=norms), indent=2))


if __name__ == "__main__":
    main()
