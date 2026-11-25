"""Replay measured matrix layouts through an independently compiled dispatch trace."""
import argparse
import csv
from pathlib import Path
import subprocess
import tempfile


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError(f"expected one source marker: {before[:80]}")
    return source.replace(before, after)


def instrument(source):
    key = '#include <stdint.h>'
    source = replace_once(source, key, key + '\nstatic const char* replay_route;')
    entry = source.index('void cob_sgemm_rowmajor(\n')
    brace = source.index('\n{\n', entry) + 3
    source = source[:brace] + '    replay_route = "native-dispatch";\n' + source[brace:]
    call = '        cob_sgemm_few_rows(m, n, k, a, lda, b, ldb, c, ldc);'
    source = replace_once(source, call, call + '\n        replay_route = "thin-stream";')
    call = '        cob_sgemm_vector_matrix(n, k, a, b, ldb, c);'
    source = source[:entry] + replace_once(source[entry:], call, call + '\n        replay_route = "vector-stream";')
    # Record success after fringe recursion has returned. A declined schedule
    # or failed workspace allocation must not appear as a completed Strassen call.
    start = source.index('static int cob_sgemm_rowmajor_strassen1(')
    end = source.index('\n}\n', start)
    body = source[start:end]
    mark = '''    replay_route = "balanced-strassen";
#if defined(COB_USE_RECTANGULAR_STRASSEN)
    if (!legacy_geometry) replay_route = "rectangular-strassen";
#endif
'''
    if 'child_values' in body:
        mark += '''#if defined(COB_USE_RECURSIVE_STRASSEN)
    if (child_values != 0) replay_route = "two-level-winograd";
#endif
'''
    body = replace_once(body, '    return 1;', mark + '    return 1;')
    source = source[:start] + body + source[end:]
    # On Apple the compact padded dispatcher runs first. Capture its actual
    # nested outcome before the parent overwrites the label.
    marker = 'static int cob_sgemm_rowmajor_padded('
    if marker in source:
        start = source.index(marker)
        end = source.index('\n}\n', start)
        body = replace_once(source[start:end], '    return 1;', '''    if (strcmp(replay_route, "balanced-strassen") == 0)
        replay_route = "padded-balanced-strassen";
    else if (strcmp(replay_route, "rectangular-strassen") == 0)
        replay_route = "padded-rectangular-strassen";
    else
        replay_route = "padded-native-dispatch";
    return 1;''')
        source = source[:start] + body + source[end:]
    call = 'if (cob_sgemm_336(m,n,k,a,lda,b,ldb,c,ldc)) return;'
    if call in source:
        source = replace_once(source, call, call.replace('return;', '{ replay_route = "rank40"; return; }'))
    return source


DRIVER = '''
#include <limits.h>
#include <stdio.h>

static float* replay_matrix(int rows, int cols)
{
    if ((size_t)rows > SIZE_MAX / sizeof(float) / (size_t)cols) return NULL;
    const size_t bytes = (size_t)rows * cols * sizeof(float);
    void* matrix = NULL;
    if (posix_memalign(&matrix, 128, bytes) != 0) return NULL;
    memset(matrix, 0, bytes);
    return (float*)matrix;
}

int main(void)
{
    int m, n, k, pad, seen = 0;
    puts("m,n,k,pad,route");
    while (scanf("%d %d %d %d", &m, &n, &k, &pad) == 4) {
        if (m <= 0 || n <= 0 || k <= 0 || pad < 0 ||
            n > INT_MAX - pad || k > INT_MAX - pad) return 1;
        const int lda = k + pad, ldb = n + pad, ldc = n + pad;
        float* a = replay_matrix(m, lda);
        float* b = replay_matrix(k, ldb);
        float* c = replay_matrix(m, ldc);
        if (!a || !b || !c) return 2;
        a[0] = 0.25f; b[0] = 2.0f;
        cob_sgemm_rowmajor(m, n, k, a, lda, b, ldb, c, ldc);
        if (c[0] != 0.5f || !replay_route) return 3;
        printf("%d,%d,%d,%d,%s\\n", m, n, k, pad, replay_route);
        free(c); free(b); free(a); ++seen;
    }
    return seen == 0 || !feof(stdin);
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--pairs', required=True, action='append', type=Path)
    parser.add_argument('--cc', default='cc')
    parser.add_argument('flags', nargs=argparse.REMAINDER)
    opts = parser.parse_args()
    layouts = set()
    for pairs in opts.pairs:
        with pairs.open(newline='') as stream:
            for row in csv.DictReader(stream):
                layouts.add(tuple(int(row[key]) for key in ('m', 'n', 'k', 'pad')))
    if not layouts:
        raise ValueError('paired timing CSV contains no matrix layouts')
    with tempfile.TemporaryDirectory() as scratch:
        trace = Path(scratch) / 'routes.c'
        binary = Path(scratch) / 'replay'
        trace.write_text(instrument(opts.source.read_text()) + DRIVER)
        flags = opts.flags[1:] if opts.flags[:1] == ['--'] else opts.flags
        subprocess.run([opts.cc, '-I'+str(opts.source.resolve().parent), str(trace),
                        *flags, '-lm', '-o', str(binary)], check=True)
        inputs = ''.join(' '.join(map(str, dims))+'\n' for dims in sorted(layouts))
        subprocess.run([str(binary)], input=inputs, text=True, check=True)


if __name__ == '__main__':
    main()
