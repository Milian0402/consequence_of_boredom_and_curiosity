"""Check all coefficients of both published 4x4 block products using exact vectors."""
import ast
from fractions import Fraction as Q
from pathlib import Path


def expand(path, ni, no):
    zeros = (Q(0),)*ni
    env = {f'i{i}': zeros[:i]+(Q(1),)+zeros[i+1:] for i in range(ni)}
    def linear(e):
        if isinstance(e, ast.Name): return env[e.id]
        if isinstance(e, ast.Constant) and type(e.value) is int: return Q(e.value)
        if isinstance(e, ast.UnaryOp) and isinstance(e.op, ast.USub): return tuple(-v for v in linear(e.operand))
        if not isinstance(e, ast.BinOp): raise ValueError(ast.dump(e))
        a,b=linear(e.left),linear(e.right)
        if isinstance(e.op, ast.Add): return tuple(x+y for x,y in zip(a,b))
        if isinstance(e.op, ast.Sub): return tuple(x-y for x,y in zip(a,b))
        if isinstance(e.op, ast.Div) and isinstance(b,Q): return tuple(x/b for x in a)
        if isinstance(e.op, ast.Mult) and isinstance(b,Q): return tuple(x*b for x in a)
        if isinstance(e.op, ast.Mult) and isinstance(a,Q): return tuple(a*x for x in b)
        raise ValueError(ast.dump(e))
    for line in path.read_text().splitlines():
        name, expr = line.strip().removesuffix(';').split(':=')
        env[name] = linear(ast.parse(expr,mode='eval').body)
    return [env[f'o{i}'] for i in range(no)]


def verify(schedules):
    verified={}
    for rank,stem in ((48,'4x4x4_48_201-balanced'),(49,'4x4x4_49_156')):
        a,b,c=[expand(schedules/f'{stem}_{s}.slp',ni,no) for s,ni,no in [('L',16,rank),('R',16,rank),('P',rank,16)]]
        tensor=[[[Q(0) for _ in range(16)] for _ in range(16)] for _ in range(16)]
        for t in range(rank):
            for ai,av in enumerate(a[t]):
                if av:
                    for bi,bv in enumerate(b[t]):
                        if bv:
                            for ci,cv in enumerate(c):
                                if cv[t]: tensor[ai][bi][ci] += av*bv*cv[t]
        for ai,by_b in enumerate(tensor):
            for bi,by_c in enumerate(by_b):
                for ci,actual in enumerate(by_c):
                    if actual != int(ai%4==bi//4 and ci//4==ai//4 and ci%4==bi%4):
                        raise ValueError((rank,ai,bi,ci,actual))
        verified[rank]=4096
    return verified

if __name__=='__main__':
    import sys
    print(verify(Path(sys.argv[1])))
