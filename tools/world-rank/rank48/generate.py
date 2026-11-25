"""Certify the published straight-line programs over the rationals, then emit their C kernels."""
import argparse
import ast
import operator
import re
from fractions import Fraction
from pathlib import Path

STEMS = {48: "4x4x4_48_201-balanced", 49: "4x4x4_49_156"}
OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def evaluate(node, values):
    if isinstance(node, ast.Name):
        return values[node.id]
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return Fraction(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -evaluate(node.operand, values)
    if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
        return OPERATORS[type(node.op)](evaluate(node.left, values), evaluate(node.right, values))
    raise ValueError(f"unexpected arithmetic: {ast.dump(node)}")


def read_schedule(path, inputs, outputs):
    rows = []
    for line in path.read_text().splitlines():
        match = re.fullmatch(r"([a-z][0-9]+):=([a-z0-9+*/() -]+);", line.strip())
        if match is None:
            raise ValueError(f"unrecognized line in {path.name}: {line}")
        name, expr = match.groups()
        rows.append((name, expr, ast.parse(expr, mode="eval").body))
    matrix = [[] for _ in range(outputs)]
    for basis in range(inputs):
        variables = {f"i{index}": Fraction(index == basis) for index in range(inputs)}
        for name, _, expr in rows:
            variables[name] = evaluate(expr, variables)
        for j in range(outputs):
            matrix[j].append(variables[f"o{j}"])
    return rows, matrix


def certify(left, right, output):
    # For each pair of input matrix units, the coefficient of every output must equal A*B.
    for a in range(16):
        for b in range(16):
            terms = [(i, la[a] * rb[b]) for i, (la, rb) in enumerate(zip(left, right)) if la[a] and rb[b]]
            for c, row in enumerate(output):
                actual = sum(row[i] * value for i, value in terms)
                expected = int(a % 4 == b // 4 and c == 4 * (a // 4) + b % 4)
                if actual != expected:
                    raise ValueError(f"invalid coefficient A{a} B{b} C{c}: {actual} != {expected}")


def transform(name, schedule, inputs, outputs, is_output):
    rows, matrix = schedule
    aliases = []
    for weights in matrix:
        nonzero = [j for j, v in enumerate(weights) if v]
        aliases.append(nonzero[0] if len(nonzero) == 1 and weights[nonzero[0]] == 1 else -1)
    code = []
    if not is_output:
        code.append(f"static const int {name}_aliases[] = {{{', '.join(map(str, aliases))}}};")
    code += [f"static void {name}(int rows, int cols, const struct view *input, const struct target *out)",
             "{", "    for (int i = 0; i < rows; ++i) {", "        int j = 0;"]
    for is_vector in (True, False):
        typ = "vec" if is_vector else "float"
        code.append("        for (; j + 4 <= cols; j += 4) {" if is_vector else "        for (; j < cols; ++j) {")
        for i in range(inputs):
            addr = f"input[{i}].p + (size_t)i * input[{i}].ld + j"
            code.append(f"            {typ} i{i} = " + (f"load_vec({addr});" if is_vector else f"*({addr});"))
        seen = set()
        for var, expr, _ in rows:
            declaration = "" if var in seen else typ + " "
            code.append(f"            {declaration}{var} = {expr};")
            seen.add(var)
        for i in range(outputs):
            if is_output or aliases[i] < 0:
                addr = f"out[{i}].p + (size_t)i * out[{i}].ld + j"
                code.append("            " + (f"store_vec({addr}, o{i});" if is_vector else f"*({addr}) = o{i};"))
        code.append("        }")
    code.extend(["    }", "}"])
    return "\n".join(code)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("rank", type=int, choices=STEMS)
    cli.add_argument("header", type=Path)
    args = cli.parse_args()
    schedules = Path(__file__).resolve().parent / "schedules"
    rank = args.rank
    parts = [read_schedule(schedules / f"{STEMS[rank]}_{key}.slp", ni, no)
             for key, ni, no in (("L", 16, rank), ("R", 16, rank), ("P", rank, 16))]
    certify(*(s[1] for s in parts))
    with args.header.open("w") as stream:
        print(f"#define PRODUCT_COUNT {rank}", file=stream)
        for name, part, ni, no in zip(("left", "right", "recombine"), parts, (16, 16, rank), (rank, rank, 16)):
            print(transform(name, part, ni, no, name == "recombine"), file=stream)
    print(f"{STEMS[rank]}: verified all 4096 bilinear tensor coefficients over the rationals")


if __name__ == "__main__":
    main()
