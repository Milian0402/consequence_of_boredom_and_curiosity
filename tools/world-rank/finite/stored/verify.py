"""Certify the 2x3x3 tensor over the integers and generate its C expressions."""
from hashlib import sha256
from itertools import chain
from pathlib import Path
import sys

here = Path(__file__).resolve().parent
source = here.parent / "hk323-15-94"
if sha256(source.read_bytes()).hexdigest() != "5a76ffd8d97651df709843cdd7f1a2b1892b3efdfba44a72333698a0b54859e9":
    raise ValueError("unexpected upstream coefficient table")
sections = [[]]
for line in source.read_text().splitlines():
    if line.startswith("#"):
        if sections[-1]:
            sections.append([])
    elif line.strip():
        sections[-1].append(list(map(int, line.split())))
old_u, old_v, old_w = sections
if [len(t) for t in sections] != [6, 6, 9] or any(len(r) != 15 for t in sections for r in t):
    raise ValueError("unexpected coefficient shape")
if any(c not in (-1, 0, 1) for t in sections for r in t for c in r):
    raise ValueError("expected signed units")
# Cyclic permutation of trace(A B C^T), with output made row-major again.
u = old_v
v = [old_w[j * 3 + p] for p in range(3) for j in range(3)]
w = [old_u[j * 2 + i] for i in range(2) for j in range(3)]
for a in range(6):
    for b in range(9):
        for c in range(6):
            got = sum(x * y * z for x, y, z in zip(u[a], v[b], w[c]))
            correct = int(a // 3 == c // 3 and a % 3 == b // 3 and b % 3 == c % 3)
            if got != correct:
                raise ValueError((a, b, c, got, correct))

# These two triplets reuse x across successive left operand forms. Each tuple
# has (prior product, coefficient multiplying previous x, new A block, sign).
chains = {8: (6, 1, 3, 1), 15: (8, 1, 5, -1),
          14: (9, 1, 5, 1), 13: (14, -1, 2, 1)}
order = [1, 2, 3, 4, 5, 6, 8, 15, 7, 9, 14, 13, 10, 11, 12]
# Use the eventual destination as one of the products that it will consume.
inplace = {2: 0, 3: 1, 7: 2, 1: 3, 9: 4, 12: 5}
scratch = {r: i for i, r in enumerate(r for r in order if r not in inplace)}
if set(order) != set(range(1, 16)) or len(scratch) != 9:
    raise ValueError("product schedule mismatch")

def form(side, r):
    table, rows, cols, stride, workspace = ((u, "mh", "kh", "lda", "x") if side == "a"
                                           else (v, "kh", "nh", "ldb", "y"))
    terms = [(q, a[r - 1]) for q, a in enumerate(table) if a[r - 1]]
    if len(terms) == 1:
        return [], f"{side}[{terms[0][0]}], {stride}", terms[0][1], 0, 0
    expr, body = "", [f"    for (int i = 0; i < {rows}; ++i) {{"]
    if side == "a" and r in chains:
        prior, prior_sign, q, sign = chains[r]
        if prior != order[order.index(r) - 1]:
            raise ValueError("live x was overwritten")
        expected = [prior_sign * coeff[prior - 1] + sign * (i == q) for i, coeff in enumerate(u)]
        if expected != [row[r - 1] for row in u]:
            raise ValueError("incorrect in-place substitution")
        terms = [(q, sign)]
        expr = "dst[j]" if prior_sign == 1 else "-dst[j]"
    body.append(f"        float *dst = {workspace} + (size_t)i * {cols};")
    for q, sign in terms:
        ptr = f"{side}{q}"
        body.append(f"        const float *{ptr} = {side}[{q}] + (size_t)i * {stride};")
        if sign > 0:
            operator = " + " if expr else ""
        else:
            operator = " - " if expr else "-"
        expr += f"{operator}{ptr}[j]"
    body += [f"        for (int j = 0; j < {cols}; ++j) dst[j] = {expr};", "    }"]
    reads = len(terms) + int(side == "a" and r in chains)
    return body, f"{workspace}, {cols}", 1, reads, reads - 1

# Every terminal expression is exactly binary. The program is also interpreted
# symbolically to prove the emitted common-subexpression graph reconstructs W.
ops = [
    ("d", "p8", "-", "p6"), ("e", "p11", "+", "p14"),
    ("f", "p1", "+", "d"), ("g", "e", "+", "p13"),
    ("h", "p2", "+", "p4"), ("i0", "h", "+", "p5"), ("z0", "f", "+", "i0"),
    ("j1", "p3", "+", "p4"), ("z1", "j1", "-", "g"),
    ("j2", "p6", "+", "p7"), ("z2", "j2", "+", "g"),
    ("j3", "p10", "-", "p15"), ("z3", "f", "+", "j3"),
    ("j4", "p3", "+", "p9"), ("k4", "p10", "-", "p5"),
    ("l4", "j4", "+", "k4"), ("z4", "l4", "-", "e"),
    ("j5", "p11", "+", "p12"), ("k5", "p15", "-", "d"), ("z5", "j5", "+", "k5"),
]
symbolic = {f"p{r}": [int(r == i + 1) for i in range(15)] for r in order}
depth = {name: 0 for name in symbolic}
for dest, a, op, b in ops:
    symbolic[dest] = [x + y if op == "+" else x - y for x, y in zip(symbolic[a], symbolic[b])]
    depth[dest] = 1 + max(depth[a], depth[b])
if [symbolic[f"z{i}"] for i in range(6)] != w:
    raise ValueError("output reduction differs from certified tensor")

header = ["/* Generated from Sandia's BSD-2-licensed Hopcroft-Kerr table.",
          " * Copyright (c) 2014, Sandia Corporation. See LICENSE.fast-matmul. */"]
body = header[:]
factor = {}
reads = {"a": 0, "b": 0}
additions = {"a": 0, "b": 0}
for r in order:
    body.append(f"    /* Product {r}. */")
    left, a_operand, sa, ra, aa = form("a", r)
    right, b_operand, sb, rb, ab = form("b", r)
    factor[r] = sa * sb
    if r in inplace:
        out = f"c[{inplace[r]}], ldc"
    else:
        out = f"product + {scratch[r]} * pc, nh"
    body += left + right + [f"    cob_sgemm_rowmajor(mh, nh, kh, {a_operand}, {b_operand}, {out});"]
    reads["a"] += ra
    reads["b"] += rb
    additions["a"] += aa
    additions["b"] += ab
body += ["    hk_reduce(mh, nh, ldc, pc, product, c[0], c[1], c[2], c[3], c[4], c[5]);"]

reduce = header + ["static void hk_reduce(int mh, int nh, int ldc, size_t pc,",
                   "                      const float *restrict product,",
                   "                      float *restrict c0, float *restrict c1, float *restrict c2,",
                   "                      float *restrict c3, float *restrict c4, float *restrict c5)",
                   "{", "    for (int i = 0; i < mh; ++i) {",
                   "        const size_t ci = (size_t)i * ldc, pi = (size_t)i * nh;",
                   "        for (int j = 0; j < nh; ++j) {"]
for r in range(1, 16):
    ref = f"c{inplace[r]}[ci + j]" if r in inplace else f"product[{scratch[r]} * pc + pi + j]"
    reduce.append(f"            const float p{r} = {'' if factor[r] == 1 else '-'}{ref};")
for dest, a, op, b in ops:
    reduce.append(f"            const float {dest} = {a} {op} {b};")
for i in range(6):
    reduce.append(f"            c{i}[ci + j] = z{i};")
reduce += ["        }", "    }", "}"]
for name, lines in [("hk233.inc", body), ("hk233_reduce.inc", reduce)]:
    path = here / name
    content = "\n".join(lines) + "\n"
    if sys.argv[1:] == ["--generate"]:
        path.write_text(content)
    elif sys.argv[1:] or path.read_text() != content:
        raise SystemExit(f"{name} differs from certified schedule")
print("324/324 exact integer coefficients, verified four A substitutions and all 90 output coefficients")
print(f"15 leaves, 6 directly in output; additive counts A/B/C {additions['a']}/{additions['b']}/{len(ops)}")
print("Explicit array reads/writes beyond leaf GEMMs, counted per scalar across one block:")
print(f"A {reads['a']}/11; B {reads['b']}/9; C and products 15/6; max output add depth {max(depth.values())}")
