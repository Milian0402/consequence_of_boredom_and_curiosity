#include "cob_gemm.h"

#include <math.h>
#include <stdint.h>
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
    if (argc != 2) return 2;
    uint32_t state = 40096813;
    for (int i = 0; i < M; ++i)
        for (int p = 0; p < LD; ++p)
            input[i][p] = p < K ? ldexpf(sample(&state), p % 13 - 6 + i % 5 - 2) : NAN;

    cob_sgemm_rowmajor(M, N, K, &input[0][0], LD, &input[0][0], LD, &output[0][0], LDC);
    int above = 0, worst_row = -1, worst_col = -1;
    double maximum = 0.0, worst_reference = 0.0;
    FILE *raw = fopen(argv[1], "wb");
    if (!raw) return 2;
    for (int i = 0; i < M; ++i) {
        if (fwrite(output[i], sizeof(float), N, raw) != N) return 2;
        for (int j = 0; j < N; ++j) {
            double reference = 0.0;
            for (int p = 0; p < K; ++p)
                reference += (double)input[i][p] * input[p][j];
            double error = fabs(reference - output[i][j]);
            if (!isfinite(error)) return 2;
            if (error > 0.002) ++above;
            if (error > maximum) {
                maximum = error;
                worst_reference = reference;
                worst_row = i;
                worst_col = j;
            }
        }
    }
    if (fclose(raw)) return 2;
    printf("outputs=%d above_limit=%d row=%d col=%d reference=%.12g max_abs=%.12g\n",
           M*N, above, worst_row, worst_col, worst_reference, maximum);
    if (above == 1 && worst_row == 44 && worst_col == 12 &&
        fabs(maximum - 0.00283442336513) <= 1e-12) return 1;
    return above ? 2 : 0;
}
