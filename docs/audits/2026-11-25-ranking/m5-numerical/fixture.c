#include "cob_gemm.h"

#include <math.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>

enum { M = 64, N = 25, K = 29, LD = 32, LDC = 31 };
static float input[M][LD], output[M][LDC];

static float sample(uint32_t *state)
{
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return ((float)(x % 34001) - 17000.0f) / 17057.0f;
}

int main(int argc, char **argv)
{
    if (argc != 2 || (strcmp(argv[1],"base") && strcmp(argv[1],"candidate"))) return 2;
    uint32_t state = 40096813;
    for (int i = 0; i < M; ++i)
        for (int p = 0; p < LD; ++p)
            input[i][p] = p < K ? ldexpf(sample(&state), p % 13 - 6 + i % 5 - 2) : NAN;

    cob_sgemm_rowmajor(M, N, K, &input[0][0], LD, &input[0][0], LD, &output[0][0], LDC);
    int above = 0, nearest_mismatches = 0, worst_row = -1, worst_col = -1;
    double maximum = 0.0, worst_reference = 0.0;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double reference = 0.0;
            for (int p = 0; p < K; ++p)
                reference += (double)input[i][p] * input[p][j];
            double error = fabs(reference - output[i][j]);
            if (!isfinite(error)) return 2;
            if (error > 0.002) ++above;
            if (output[i][j] != (float)reference) ++nearest_mismatches;
            if (error > maximum) {
                maximum = error;
                worst_reference = reference;
                worst_row = i;
                worst_col = j;
            }
        }
    }
    printf("%s outputs=%d above_limit=%d unequal_to_nearest=%d row=%d col=%d "
           "reference=%.12g max_abs=%.12g\n", argv[1], M*N, above, nearest_mismatches,
           worst_row, worst_col, worst_reference, maximum);
    int baseline = strcmp(argv[1], "base") == 0;
    double expected = baseline ? 0.00283442336513 : 0.00088129836513;
    return !(above == baseline && nearest_mismatches == (baseline ? 1002 : 0) &&
             worst_row == 44 && worst_col == 12 && fabs(maximum-expected) <= 1e-12);
}
