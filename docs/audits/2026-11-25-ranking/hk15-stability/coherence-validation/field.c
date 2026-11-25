#define _POSIX_C_SOURCE 200809L
#include <cblas.h>
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef void (*multiply)(int, int, int, const float *, int, const float *, int, float *, int);
enum fixture { CENTERED, POSITIVE, ROWCOL, HALF_BIASED, STRONG_BIAS, ONE_SIDED, TWO_PROFILES, THREE_PROFILES, SMALL_GROUP, WEAK_BIAS, TILTED };
static const char *name[] = {"centered","positive","row-col-signs","partial-bias","biased-signed","one-sided","two-depth-profiles","three-depth-profiles","rare-group","weak-bias","depth-signs"};
static uint32_t step(uint32_t *s) {
    *s ^= *s << 13;
    *s ^= *s >> 17;
    *s ^= *s << 5;
    return *s;
}
static uint32_t mix(uint32_t s) {
    s ^= s >> 16;
    s *= 0x7feb352dU;
    s ^= s >> 15;
    s *= 0x846ca68bU;
    return s ^ (s >> 16);
}
static float value(enum fixture f, int right, int i, int j, uint32_t seed, uint32_t *state) {
    const uint32_t u = step(state);
    float x = (float)(u >> 9) * 0x1p-23f;
    int out = right ? j : i, depth = right ? i : j;
    uint32_t tag = mix(seed ^ (right ? 77383u : 23917u) ^ (unsigned)out);
    float sign = tag & 1 ? -1.f : 1.f;
    if (f == CENTERED) return x - .5f;
    if (f == POSITIVE) return x;
    if (f == ROWCOL) return sign*x;
    if (f == HALF_BIASED) return (tag & 2) ? sign*x : x - .5f;
    if (f == STRONG_BIAS) return sign*(x - 0.09375f);
    if (f == ONE_SIDED) return right ? x - .5f : sign*x;
    if (f == TWO_PROFILES) {
        uint32_t pattern = tag & 4 ? 33241u : 19121u;
        return (mix(seed ^ pattern ^ (unsigned)depth) & 1 ? x : -x) * sign;
    }
    if (f == THREE_PROFILES) {
        uint32_t pattern = (tag%3+3)*11591u;
        return (mix(seed ^ pattern ^ (unsigned)depth) & 1 ? x : -x) * sign;
    }
    if (f == SMALL_GROUP) {
        if ((tag % 20) == 0) return x*sign;
        return x - .5f;
    }
    if (f == WEAK_BIAS) return (x - .25f) * sign;
    return (mix(seed ^ (unsigned)depth) & 1 ? x : -x) * sign;

}
static void *array(size_t cells, size_t width) {
    void *p = calloc(cells, width);
    if (!p) {
        perror("calloc");
        exit(2);
    }
    return p;
}
int main(int argc, char **argv) {
    if (argc != 6 || openblas_get_num_threads() != 1)
        return 1;
    const int m = atoi(argv[1]), n = atoi(argv[2]), k = atoi(argv[3]);
    uint32_t seed = (uint32_t)strtoul(argv[4], NULL, 10);
    if (m <= 0 || n <= 0 || k <= 0)
        return 2;
    void *lib = NULL;
    multiply subject = NULL;
    if (strcmp(argv[5], "openblas")) {
        lib = dlopen(argv[5], RTLD_LOCAL | RTLD_NOW);
        if (!lib) {
            fprintf(stderr, "%s\n", dlerror());
            return 3;
        }
        subject = (multiply)dlsym(lib, "cob_sgemm_rowmajor");
        if (!subject)
            return 4;
    }
    int la = k + 7, lb = n + 11, lc = n + 13;
    float *ap = array((size_t)m * la + 7, 4), *bp = array((size_t)k * lb + 9, 4),
          *cp = array((size_t)m * lc + 13, 4);
    float *a = ap + 3, *b = bp + 5, *c = cp + 7;
    double *da = array((size_t)m * k, 8), *db = array((size_t)k * n, 8),
           *dc = array((size_t)m * n, 8);
    puts("m,n,k,seed,fixture,cells,nonfinite,over_abs_0_002,over_rel_1pct,wrong_sign,max_abs,max_"
         "rel,rms_abs,ref_at_max_abs,got_at_max_abs,row,col");
    for (int f = 0; f <= TILTED; f++) {
        uint32_t gen = seed ^ ((uint32_t)f + 7) * 234749U;
        for (int i = 0; i < m; i++)
            for (int j = 0; j < k; j++)
                da[(size_t)i * k + j] = a[(size_t)i * la + j] = value(f, 0, i, j, seed, &gen);
        for (int i = 0; i < k; i++)
            for (int j = 0; j < n; j++)
                db[(size_t)i * n + j] = b[(size_t)i * lb + j] = value(f, 1, i, j, seed, &gen);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1., da, k, db, n, 0., dc,
                    n);
        for (int spot = 0; spot < 31; spot++) {
            int row = step(&gen) % (uint32_t)m, col = step(&gen) % (uint32_t)n;
            long double v = 0;
            for (int p = 0; p < k; p++)
                v += (long double)a[(size_t)row * la + p] * b[(size_t)p * lb + col];
            if (fabsl(v - dc[(size_t)row * n + col]) > 1e-9)
                return 5;
        }
        for (int i = 0; i < m; i++)
            for (int j = 0; j < lc; j++)
                c[(size_t)i * lc + j] = j < n ? NAN : -773.5f;
        if (subject)
            subject(m, n, k, a, la, b, lb, c, lc);
        else
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.f, a, la, b, lb, 0.f,
                        c, lc);
        int bad = 0, over = 0, rel_over = 0, sign = 0, wi = 0, wj = 0;
        double worst = 0, max_rel = 0, squares = 0;
        for (int i = 0; i < m; i++)
            for (int j = 0; j < lc; j++) {
                double got = c[(size_t)i * lc + j];
                if (j >= n) {
                    if (got != -773.5f)
                        return 6;
                    continue;
                }
                double ref = dc[(size_t)i * n + j], err = fabs(got - ref);
                if (!isfinite(ref))
                    return 7;
                if (!isfinite(got)) {
                    bad++;
                    err = INFINITY;
                }
                double rel = ref == 0 ? (err == 0 ? 0 : INFINITY) : err / fabs(ref);
                over += err > .002;
                rel_over += rel > .01;
                sign += ref * got < 0;
                if (err > worst) {
                    worst = err;
                    wi = i;
                    wj = j;
                }
                max_rel = fmax(max_rel, rel);
                squares += err * err;
            }
        for (int i = 0; i < m; i++)
            for (int j = 0; j < k; j++)
                if (a[(size_t)i * la + j] != da[(size_t)i * k + j])
                    return 8;
        for (int i = 0; i < k; i++)
            for (int j = 0; j < n; j++)
                if (b[(size_t)i * lb + j] != db[(size_t)i * n + j])
                    return 9;
        printf("%d,%d,%d,%u,%s,%d,%d,%d,%d,%d,%.12g,%.12g,%.12g,%.12g,%.12g,%d,%d\n", m, n, k, seed,
               name[f], m * n, bad, over, rel_over, sign, worst, max_rel,
               sqrt(squares / ((double)m * n)), dc[(size_t)wi * n + wj],
               (double)c[(size_t)wi * lc + wj], wi, wj);
        fflush(stdout);
    }
    free(ap);
    free(bp);
    free(cp);
    free(da);
    free(db);
    free(dc);
    if (lib)
        dlclose(lib);
    return 0;
}
