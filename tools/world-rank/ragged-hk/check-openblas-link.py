"""Verify GNU ld's static symbol origins and the ordinary Linux executable's definitions."""
from pathlib import Path
import re
import sys


def member(lines, archive, relation, symbol):
    suffix = ': ' + relation + ' ' + symbol
    records = {line for line in lines if line.endswith(suffix)}
    pattern = re.compile(r'(?:^|: )' + re.escape(str(archive)) + r'\(([^()]+)\)' + re.escape(suffix) + '$')
    matches = {match[1] for line in records if (match := pattern.search(line))}
    if len(matches) != 1 or (relation == 'definition of' and len(records) != 1):
        raise ValueError(f'Expected one {archive.name} member with {relation} {symbol}, found {sorted(matches)}')
    return matches.pop()


def verify(cob, openblas, trace, symbols):
    lines = trace.read_text().splitlines()
    functions = {row[-1] for line in symbols.read_text().splitlines()
                 if len(row := line.split()) == 3 and row[1] in ('T', 't')}
    for api in ('cblas_sgemm', 'openblas_get_num_threads'):
        if api not in functions:
            raise ValueError(f'Missing {api} in the statically linked ordinary worker')
        caller = member(lines, cob, 'reference to', api)
        provider = member(lines, openblas, 'definition of', api)
        print(f'{cob.name}({caller}) -> {openblas.name}({provider}): {api}')


if __name__ == '__main__':
    verify(*(Path(arg) for arg in sys.argv[1:]))
