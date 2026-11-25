"""Certify the 2x3x3 tensor over the integers and generate its C expressions."""
from hashlib import sha256
from itertools import chain
from pathlib import Path
import sys

here = Path(__file__).resolve().parent
source = here / "hk323-15-94"
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

def terms(matrix, r):
    return [(i, row[r]) for i, row in enumerate(matrix) if row[r]]

def expression(xs, symbol):
    result = ""
    for q, (i, sign) in enumerate(xs):
        operator = (" + " if q else "") if sign > 0 else (" - " if q else "-")
        result += operator + f"{symbol}{i}[j]"
    return result

def input_form(side, xs):
    # A singleton sign can always be moved to the product's output coefficients.
    sign_factor = -1 if xs[0][1] < 0 else 1
    xs = [(i, sign * sign_factor) for i, sign in xs]
    if side == "a":
        rows, cols, stride, workspace = "mh", "kh", "lda", "x"
    else:
        rows, cols, stride, workspace = "kh", "nh", "ldb", "y"
    if len(xs) == 1:
        return [], f"{side}[{xs[0][0]}], {stride}", sign_factor
    lines = [f"    for (int i = 0; i < {rows}; ++i) {{"]
    for block, _ in xs:
        lines += [f"        const float *{side}{block} = {side}[{block}] + (size_t)i * {stride};"]
    lines += [f"        float *dst = {workspace} + (size_t)i * {cols};",
              f"        for (int j = 0; j < {cols}; ++j) dst[j] = {expression(xs, side)};", "    }"]
    return lines, f"{workspace}, {cols}", sign_factor

def generate():
    code = ["/* Generated from Sandia's BSD-2-licensed Hopcroft-Kerr table.",
            " * Copyright (c) 2014, Sandia Corporation. See LICENSE.fast-matmul. */"]
    initialized = set()
    nontrivial = [0, 0]
    for r in range(15):
        aa, left, s1 = input_form("a", terms(u, r))
        bb, right, s2 = input_form("b", terms(v, r))
        nontrivial[0] += bool(aa)
        nontrivial[1] += bool(bb)
        out = [(i, s * s1 * s2) for i, s in terms(w, r)]
        direct = next((i for i, s in out if i not in initialized and s == 1), None)
        target, ld = ("product", "nh") if direct is None else (f"c[{direct}]", "ldc")
        code.append(f"    /* Product {r + 1} of 15. */")
        rhs = [(i, s*s2) for i,s in terms(v,r)]
        lazy = not aa and len(rhs) == 3
        if lazy:
            pointers = ", ".join(f"b[{i}]" for i,_ in rhs)
            flags = ", ".join(str(int(s>0)) for _,s in rhs[1:])
            left_pointer = left.split(",")[0]
            code.extend(["#ifdef COB_HK_PACK_RIGHT", f"    if (!COB_HK_PACK_RIGHT({left_pointer}, {pointers}, {flags}, {target}, {ld}))", "#endif", "    {"])
        body = [*aa,*bb,f"    cob_sgemm_rowmajor(mh, nh, kh, {left}, {right}, {target}, {ld});"]
        code.extend(("    "+line if lazy else line) for line in body)
        if lazy:
            code.append("    }")
        rest = [(i, s) for i, s in out if i != direct]
        if rest:
            code.append("    for (int i = 0; i < mh; ++i) {")
            code.append(f"        const float *src = {target} + (size_t)i * {ld};")
            for i, _ in rest:
                code.append(f"        float *c{i} = c[{i}] + (size_t)i * ldc;")
            code.append("        for (int j = 0; j < nh; ++j) {")
            for i, s in rest:
                op = ("+=" if s == 1 else "-=") if i in initialized else ("=" if s == 1 else "= -")
                code.append(f"            c{i}[j] {op} src[j];")
            code.extend(["        }", "    }"])
        initialized.update(i for i, _ in out)
    print(f"materialized input forms A/B={nontrivial}, others are direct quadrant pointers")
    return "\n".join(code) + "\n"

add_a = sum(len(terms(u, r)) - 1 for r in range(15))
add_b = sum(len(terms(v, r)) - 1 for r in range(15))
add_c = sum(sum(x != 0 for x in row) - 1 for row in w)
print(f"324 exact integer tensor coefficients passed, 15 vs 18 products, "
      f"direct schedule additions A/B/C={add_a}/{add_b}/{add_c}; "
      f"nonzero coefficients={sum(c != 0 for t in (u,v,w) for c in chain.from_iterable(t))}")
body = generate()
generated = here / "hk233.inc"
if sys.argv[1:] == ["--generate"]:
    generated.write_text(body)
elif sys.argv[1:] or generated.read_text() != body:
    raise SystemExit("generated kernel differs from certified table; run verify.py --generate")
