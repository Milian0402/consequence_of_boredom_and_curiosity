#define _POSIX_C_SOURCE 200809L
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
typedef void (*multiply)(int, int, int, const float *, int, const float *, int, float *, int);
struct shape {
    int m, n, k;
};
static const struct shape shapes[] = {
    {2024, 3081, 3005}, {2153, 3007, 3159}, {1949, 3227, 3023}, {1920, 2880, 2880}};
static uint32_t next(uint32_t *s) {
    *s ^= *s << 13;
    *s ^= *s >> 17;
    *s ^= *s << 5;
    return *s;
}
static float draw(uint32_t *s) {
    return (float)((int)(next(s) >> 9) - 4194304) * (1.f / 8388608.f);
}
static double clock_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}
int main(int argc, char **argv) {
    if (argc != 3)
        return 1;
    void *old = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL),
         *modified = dlopen(argv[2], RTLD_NOW | RTLD_LOCAL);
    if (!old || !modified) {
        fprintf(stderr, "%s\n", dlerror());
        return 2;
    }
    multiply a = (multiply)dlsym(old, "cob_sgemm_rowmajor"),
             g = (multiply)dlsym(modified, "cob_sgemm_rowmajor");
    if (!a || !g || a == g)
        return 3;
    multiply f[] = {a, a, g};
    const int orders[6][3] = {{0, 1, 2}, {2, 0, 1}, {1, 2, 0}, {2, 1, 0}, {0, 2, 1}, {1, 0, 2}};
    printf("id,m,n,k,seed,block,position,method,seconds,verified_cells,reference_spots\n");
    for (int s = 0; s < 2; s++)
        for (int t = 0; t < 4; t++) {
            int x = s ? 3 - t : t, m = shapes[x].m, n = shapes[x].n, k = shapes[x].k, la = k + 5,
                lb = n + 9, lc = n + 7;
            size_t ca = (size_t)m * la, cb = (size_t)k * lb, cc = (size_t)m * lc;
            float *ap = calloc(ca + 4, 4), *bp = calloc(cb + 9, 4), *cp = calloc(cc + 12, 4),
                  *ref = malloc(cc * 4);
            if (!ap || !bp || !cp || !ref)
                return 4;
            float *A = ap + 3, *B = bp + 7, *C = cp + 5;
            uint32_t seed = (s ? 362004289u : 37774301u) + x, gen = seed;
            for (int i = 0; i < m; i++)
                for (int p = 0; p < k; p++)
                    A[(size_t)i * la + p] = draw(&gen);
            for (int p = 0; p < k; p++)
                for (int j = 0; j < n; j++)
                    B[(size_t)p * lb + j] = draw(&gen);
            for (size_t q = 0; q < cc; q++)
                C[q] = 17.25f;
            for (int h = 0; h < 3; h++) {
                f[h](m, n, k, A, la, B, lb, C, lc);
                if (h == 0)
                    memcpy(ref, C, cc * 4);
                else if (memcmp(ref, C, cc * 4)) {
                    fprintf(stderr, "normal-input bits differ, shape %d\n", x);
                    return 5;
                }
            }
            for (int p = 0; p < 96; p++) {
                int i = next(&gen) % (unsigned)m, j = next(&gen) % (unsigned)n;
                double val = 0;
                for (int l = 0; l < k; l++)
                    val += (double)A[(size_t)i * la + l] * B[(size_t)l * lb + j];
                if (!isfinite(C[(size_t)i * lc + j]) || fabs(C[(size_t)i * lc + j] - val) > .002)
                    return 6;
            }
            for (int b = 0; b < 6; b++)
                for (int z = 0; z < 3; z++) {
                    int method = orders[(b + x + 3 * s) % 6][z];
                    double start = clock_s();
                    f[method](m, n, k, A, la, B, lb, C, lc);
                    double elapsed = clock_s() - start;
                    if (memcmp(C, ref, cc * 4))
                        return 7;
                    printf("%d,%d,%d,%d,%u,%d,%d,%c,%.12g,%d,96\n", x, m, n, k, seed, b, z,
                           "ABG"[method], elapsed, m * n);
                }
            free(ref);
            free(cp);
            free(bp);
            free(ap);
            fflush(stdout);
        }
    dlclose(old);
    dlclose(modified);
    return 0;
}
