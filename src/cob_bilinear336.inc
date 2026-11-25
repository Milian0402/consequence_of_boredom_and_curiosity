/* Single-level 3 by 3 by 6 experiment using the exact rank-40 Tichavsky-Kovac
   coefficients in arbenson/fast-matmul at 132358be7a6ab57c83dd5ce70ece460bd1ec5dfa.
   Derived coefficient expressions copyright (c) 2014 Sandia Corporation.
   See LICENSE.fast-matmul. The upstream coefficients are all 0 or +/- 1/2.
   The common 1/8 scale is applied once on output. Each form is summed in
   balanced pairs, and all internal products are classical native GEMMs. */
static void cob_336_a(int form, int rows, int cols, int ld,
    const float* const* blocks, float* dst, int dst_ld)
{
    const float* p[9];
    for (int row = 0; row < rows; ++row) {
        for (int index = 0; index < 9; ++index)
            p[index] = blocks[index] + (size_t)row * ld;
        float* const out = dst + (size_t)row * dst_ld;
        switch (form) {
        case 0:
            for (int j = 0; j < cols; ++j) out[j] = (((p[0][j] + p[2][j]) + (p[4][j] + (p[5][j] + p[7][j]))) - p[6][j]);
            break;
        case 1:
            for (int j = 0; j < cols; ++j) out[j] = ((p[0][j] + (p[4][j] + p[7][j])) - (p[2][j] + (p[5][j] + p[6][j])));
            break;
        case 2:
            for (int j = 0; j < cols; ++j) out[j] = ((p[4][j] + (p[6][j] + p[7][j])) - (p[0][j] + (p[2][j] + p[5][j])));
            break;
        case 3:
            for (int j = 0; j < cols; ++j) out[j] = ((p[2][j] + (p[6][j] + p[7][j])) - (p[0][j] + (p[4][j] + p[5][j])));
            break;
        case 4:
            for (int j = 0; j < cols; ++j) out[j] = ((p[0][j] + (p[2][j] + p[4][j])) - (p[5][j] + (p[6][j] + p[7][j])));
            break;
        case 5:
            for (int j = 0; j < cols; ++j) out[j] = ((p[2][j] + (p[4][j] + p[7][j])) - (p[0][j] + (p[5][j] + p[6][j])));
            break;
        case 6:
            for (int j = 0; j < cols; ++j) out[j] = ((p[4][j] + (p[5][j] + p[7][j])) - (p[0][j] + (p[2][j] + p[6][j])));
            break;
        case 7:
            for (int j = 0; j < cols; ++j) out[j] = (((p[0][j] + p[2][j]) + (p[4][j] + (p[6][j] + p[7][j]))) - p[5][j]);
            break;
        case 8:
            for (int j = 0; j < cols; ++j) out[j] = ((p[2][j] + (p[4][j] + p[6][j])) - (p[0][j] + (p[5][j] + p[7][j])));
            break;
        case 9:
            for (int j = 0; j < cols; ++j) out[j] = ((p[4][j] + (p[5][j] + p[6][j])) - (p[0][j] + (p[2][j] + p[7][j])));
            break;
        case 10:
            for (int j = 0; j < cols; ++j) out[j] = (((p[0][j] + p[2][j]) + (p[5][j] + (p[6][j] + p[7][j]))) - p[4][j]);
            break;
        case 11:
            for (int j = 0; j < cols; ++j) out[j] = (((p[0][j] + p[2][j]) + (p[4][j] + (p[5][j] + p[6][j]))) - p[7][j]);
            break;
        case 12:
            for (int j = 0; j < cols; ++j) out[j] = ((p[2][j] + (p[5][j] + p[7][j])) - (p[0][j] + (p[4][j] + p[6][j])));
            break;
        case 13:
            for (int j = 0; j < cols; ++j) out[j] = ((p[2][j] + (p[4][j] + p[5][j])) - (p[0][j] + (p[6][j] + p[7][j])));
            break;
        case 14:
            for (int j = 0; j < cols; ++j) out[j] = (((p[2][j] + p[4][j]) + (p[5][j] + (p[6][j] + p[7][j]))) - p[0][j]);
            break;
        case 15:
            for (int j = 0; j < cols; ++j) out[j] = (((p[0][j] + p[4][j]) + (p[5][j] + (p[6][j] + p[7][j]))) - p[2][j]);
            break;
        case 16:
            for (int j = 0; j < cols; ++j) out[j] = -((p[3][j] + p[4][j]) + (p[6][j] + p[7][j]));
            break;
        case 17:
            for (int j = 0; j < cols; ++j) out[j] = ((p[0][j] + p[1][j]) - (p[6][j] + p[7][j]));
            break;
        case 18:
            for (int j = 0; j < cols; ++j) out[j] = ((p[4][j] + p[7][j]) - (p[3][j] + p[6][j]));
            break;
        case 19:
            for (int j = 0; j < cols; ++j) out[j] = ((p[1][j] + p[2][j]) + (p[4][j] + p[5][j]));
            break;
        case 20:
            for (int j = 0; j < cols; ++j) out[j] = ((p[2][j] + p[3][j]) - (p[0][j] + p[5][j]));
            break;
        case 21:
            for (int j = 0; j < cols; ++j) out[j] = -((p[0][j] + p[2][j]) + (p[3][j] + p[5][j]));
            break;
        case 22:
            for (int j = 0; j < cols; ++j) out[j] = ((p[5][j] + p[8][j]) - (p[4][j] + p[7][j]));
            break;
        case 23:
            for (int j = 0; j < cols; ++j) out[j] = -((p[4][j] + p[5][j]) + (p[7][j] + p[8][j]));
            break;
        case 24:
            for (int j = 0; j < cols; ++j) out[j] = ((p[1][j] + p[6][j]) - (p[0][j] + p[7][j]));
            break;
        case 25:
            for (int j = 0; j < cols; ++j) out[j] = ((p[2][j] + p[4][j]) - (p[1][j] + p[5][j]));
            break;
        case 26:
            for (int j = 0; j < cols; ++j) out[j] = ((p[0][j] + p[2][j]) - (p[6][j] + p[8][j]));
            break;
        case 27:
            for (int j = 0; j < cols; ++j) out[j] = -((p[0][j] + p[2][j]) + (p[6][j] + p[8][j]));
            break;
        case 28:
            for (int j = 0; j < cols; ++j) out[j] = ((p[0][j] + p[6][j]) - (p[2][j] + p[8][j]));
            break;
        case 29:
            for (int j = 0; j < cols; ++j) out[j] = ((p[4][j] + p[8][j]) - (p[5][j] + p[7][j]));
            break;
        case 30:
            for (int j = 0; j < cols; ++j) out[j] = ((p[7][j] + p[8][j]) - (p[4][j] + p[5][j]));
            break;
        case 31:
            for (int j = 0; j < cols; ++j) out[j] = ((p[0][j] + p[8][j]) - (p[2][j] + p[6][j]));
            break;
        case 32:
            for (int j = 0; j < cols; ++j) out[j] = ((p[0][j] + p[6][j]) - (p[1][j] + p[7][j]));
            break;
        case 33:
            for (int j = 0; j < cols; ++j) out[j] = ((p[0][j] + p[3][j]) - (p[2][j] + p[5][j]));
            break;
        case 34:
            for (int j = 0; j < cols; ++j) out[j] = ((p[1][j] + p[4][j]) - (p[2][j] + p[5][j]));
            break;
        case 35:
            for (int j = 0; j < cols; ++j) out[j] = ((p[0][j] + p[1][j]) + (p[6][j] + p[7][j]));
            break;
        case 36:
            for (int j = 0; j < cols; ++j) out[j] = ((p[4][j] + p[6][j]) - (p[3][j] + p[7][j]));
            break;
        case 37:
            for (int j = 0; j < cols; ++j) out[j] = ((p[6][j] + p[7][j]) - (p[3][j] + p[4][j]));
            break;
        case 38:
            for (int j = 0; j < cols; ++j) out[j] = ((p[4][j] + p[5][j]) - (p[1][j] + p[2][j]));
            break;
        case 39:
            for (int j = 0; j < cols; ++j) out[j] = ((p[3][j] + p[5][j]) - (p[0][j] + p[2][j]));
            break;
        }
    }
}

static void cob_336_b(int form, int rows, int cols, int ld,
    const float* const* blocks, float* dst, int dst_ld)
{
    const float* p[18];
    for (int row = 0; row < rows; ++row) {
        for (int index = 0; index < 18; ++index)
            p[index] = blocks[index] + (size_t)row * ld;
        float* const out = dst + (size_t)row * dst_ld;
        switch (form) {
        case 0:
            for (int j = 0; j < cols; ++j) out[j] = (((p[4][j] + (p[5][j] + p[8][j])) + (p[9][j] + (p[14][j] + p[17][j]))) - (p[1][j] + (p[10][j] + p[12][j])));
            break;
        case 1:
            for (int j = 0; j < cols; ++j) out[j] = ((p[7][j] + (p[10][j] + p[15][j])) - ((p[2][j] + (p[3][j] + p[4][j])) + (p[11][j] + (p[12][j] + p[13][j]))));
            break;
        case 2:
            for (int j = 0; j < cols; ++j) out[j] = (((p[2][j] + p[7][j]) + (p[9][j] + p[16][j])) - ((p[0][j] + p[5][j]) + (p[6][j] + (p[13][j] + p[17][j]))));
            break;
        case 3:
            for (int j = 0; j < cols; ++j) out[j] = ((p[9][j] + p[17][j]) - ((p[1][j] + (p[4][j] + p[5][j])) + ((p[8][j] + p[10][j]) + (p[12][j] + p[14][j]))));
            break;
        case 4:
            for (int j = 0; j < cols; ++j) out[j] = (((p[3][j] + p[4][j]) + (p[10][j] + (p[13][j] + p[15][j]))) - ((p[2][j] + p[7][j]) + (p[11][j] + p[12][j])));
            break;
        case 5:
            for (int j = 0; j < cols; ++j) out[j] = (((p[3][j] + p[6][j]) + (p[8][j] + p[11][j])) - ((p[0][j] + p[1][j]) + (p[14][j] + (p[15][j] + p[16][j]))));
            break;
        case 6:
            for (int j = 0; j < cols; ++j) out[j] = (((p[2][j] + (p[5][j] + p[6][j])) + (p[9][j] + (p[16][j] + p[17][j]))) - (p[0][j] + (p[7][j] + p[13][j])));
            break;
        case 7:
            for (int j = 0; j < cols; ++j) out[j] = (((p[1][j] + p[5][j]) + (p[8][j] + (p[12][j] + p[17][j]))) - ((p[4][j] + p[9][j]) + (p[10][j] + p[14][j])));
            break;
        case 8:
            for (int j = 0; j < cols; ++j) out[j] = ((p[6][j] + (p[7][j] + p[17][j])) - ((p[0][j] + (p[2][j] + p[5][j])) + (p[9][j] + (p[13][j] + p[16][j]))));
            break;
        case 9:
            for (int j = 0; j < cols; ++j) out[j] = ((p[1][j] + (p[6][j] + p[16][j])) - ((p[0][j] + (p[3][j] + p[8][j])) + (p[11][j] + (p[14][j] + p[15][j]))));
            break;
        case 10:
            for (int j = 0; j < cols; ++j) out[j] = (((p[1][j] + p[3][j]) + (p[8][j] + (p[15][j] + p[16][j]))) - ((p[0][j] + p[6][j]) + (p[11][j] + p[14][j])));
            break;
        case 11:
            for (int j = 0; j < cols; ++j) out[j] = ((p[4][j] + (p[7][j] + p[13][j])) - ((p[2][j] + (p[3][j] + p[10][j])) + (p[11][j] + (p[12][j] + p[15][j]))));
            break;
        case 12:
            for (int j = 0; j < cols; ++j) out[j] = (((p[5][j] + p[8][j]) + (p[9][j] + p[10][j])) - ((p[1][j] + p[4][j]) + (p[12][j] + (p[14][j] + p[17][j]))));
            break;
        case 13:
            for (int j = 0; j < cols; ++j) out[j] = ((((p[0][j] + p[2][j]) + (p[6][j] + p[7][j])) + ((p[9][j] + p[13][j]) + (p[16][j] + p[17][j]))) - p[5][j]);
            break;
        case 14:
            for (int j = 0; j < cols; ++j) out[j] = (((p[0][j] + (p[1][j] + p[3][j])) + ((p[6][j] + p[8][j]) + (p[14][j] + p[16][j]))) - (p[11][j] + p[15][j]));
            break;
        case 15:
            for (int j = 0; j < cols; ++j) out[j] = ((((p[2][j] + p[4][j]) + (p[7][j] + p[10][j])) + ((p[11][j] + p[12][j]) + (p[13][j] + p[15][j]))) - p[3][j]);
            break;
        case 16:
            for (int j = 0; j < cols; ++j) out[j] = (((p[2][j] + p[4][j]) + (p[10][j] + p[15][j])) - ((p[0][j] + (p[1][j] + p[3][j])) + (p[5][j] + (p[6][j] + p[17][j]))));
            break;
        case 17:
            for (int j = 0; j < cols; ++j) out[j] = ((p[7][j] + (p[9][j] + p[11][j])) - ((p[0][j] + (p[4][j] + p[6][j])) + ((p[8][j] + p[10][j]) + (p[13][j] + p[14][j]))));
            break;
        case 18:
            for (int j = 0; j < cols; ++j) out[j] = (((p[2][j] + (p[3][j] + p[4][j])) + (p[5][j] + (p[6][j] + p[17][j]))) - ((p[0][j] + p[1][j]) + (p[10][j] + p[15][j])));
            break;
        case 19:
            for (int j = 0; j < cols; ++j) out[j] = ((((p[0][j] + p[4][j]) + (p[6][j] + p[7][j])) + ((p[8][j] + p[9][j]) + (p[13][j] + p[14][j]))) - (p[10][j] + p[11][j]));
            break;
        case 20:
            for (int j = 0; j < cols; ++j) out[j] = (((p[0][j] + (p[1][j] + p[2][j])) + ((p[4][j] + p[5][j]) + (p[10][j] + p[15][j]))) - (p[3][j] + (p[6][j] + p[17][j])));
            break;
        case 21:
            for (int j = 0; j < cols; ++j) out[j] = (((p[0][j] + p[4][j]) + (p[5][j] + (p[6][j] + p[17][j]))) - ((p[1][j] + p[2][j]) + (p[3][j] + (p[10][j] + p[15][j]))));
            break;
        case 22:
            for (int j = 0; j < cols; ++j) out[j] = (((p[3][j] + (p[5][j] + p[8][j])) + (p[12][j] + (p[13][j] + p[17][j]))) - ((p[7][j] + p[14][j]) + (p[15][j] + p[16][j])));
            break;
        case 23:
            for (int j = 0; j < cols; ++j) out[j] = (((p[3][j] + (p[5][j] + p[8][j])) + (p[14][j] + (p[16][j] + p[17][j]))) - ((p[7][j] + p[12][j]) + (p[13][j] + p[15][j])));
            break;
        case 24:
            for (int j = 0; j < cols; ++j) out[j] = (((p[0][j] + (p[4][j] + p[8][j])) + ((p[9][j] + p[11][j]) + (p[13][j] + p[14][j]))) - (p[6][j] + (p[7][j] + p[10][j])));
            break;
        case 25:
            for (int j = 0; j < cols; ++j) out[j] = (((p[6][j] + p[7][j]) + (p[8][j] + p[11][j])) - ((p[0][j] + (p[4][j] + p[9][j])) + (p[10][j] + (p[13][j] + p[14][j]))));
            break;
        case 26:
            for (int j = 0; j < cols; ++j) out[j] = (((p[3][j] + (p[5][j] + p[8][j])) + ((p[13][j] + p[14][j]) + (p[15][j] + p[17][j]))) - (p[7][j] + (p[12][j] + p[16][j])));
            break;
        case 27:
            for (int j = 0; j < cols; ++j) out[j] = (((p[3][j] + (p[5][j] + p[8][j])) + ((p[12][j] + p[15][j]) + (p[16][j] + p[17][j]))) - (p[7][j] + (p[13][j] + p[14][j])));
            break;
        case 28:
            for (int j = 0; j < cols; ++j) out[j] = ((p[3][j] + (p[5][j] + p[8][j])) - ((p[7][j] + (p[12][j] + p[13][j])) + ((p[14][j] + p[15][j]) + (p[16][j] + p[17][j]))));
            break;
        case 29:
            for (int j = 0; j < cols; ++j) out[j] = (((p[3][j] + (p[5][j] + p[8][j])) + (p[13][j] + (p[15][j] + p[16][j]))) - ((p[7][j] + p[12][j]) + (p[14][j] + p[17][j])));
            break;
        case 30:
            for (int j = 0; j < cols; ++j) out[j] = (((p[3][j] + (p[5][j] + p[8][j])) + (p[12][j] + (p[14][j] + p[15][j]))) - ((p[7][j] + p[13][j]) + (p[16][j] + p[17][j])));
            break;
        case 31:
            for (int j = 0; j < cols; ++j) out[j] = (((p[3][j] + (p[5][j] + p[8][j])) + ((p[12][j] + p[13][j]) + (p[14][j] + p[16][j]))) - (p[7][j] + (p[15][j] + p[17][j])));
            break;
        case 32:
            for (int j = 0; j < cols; ++j) out[j] = (((p[6][j] + p[8][j]) + (p[9][j] + (p[10][j] + p[11][j]))) - ((p[0][j] + p[4][j]) + (p[7][j] + (p[13][j] + p[14][j]))));
            break;
        case 33:
            for (int j = 0; j < cols; ++j) out[j] = (((p[0][j] + (p[1][j] + p[2][j])) + ((p[3][j] + p[4][j]) + (p[6][j] + p[17][j]))) - (p[5][j] + (p[10][j] + p[15][j])));
            break;
        case 34:
            for (int j = 0; j < cols; ++j) out[j] = (((p[7][j] + p[8][j]) + (p[9][j] + p[10][j])) - ((p[0][j] + (p[4][j] + p[6][j])) + (p[11][j] + (p[13][j] + p[14][j]))));
            break;
        case 35:
            for (int j = 0; j < cols; ++j) out[j] = (p[8][j] - (((p[0][j] + p[4][j]) + (p[6][j] + p[7][j])) + ((p[9][j] + p[10][j]) + (p[11][j] + (p[13][j] + p[14][j])))));
            break;
        case 36:
            for (int j = 0; j < cols; ++j) out[j] = (((p[1][j] + p[4][j]) + (p[6][j] + p[17][j])) - ((p[0][j] + (p[2][j] + p[3][j])) + (p[5][j] + (p[10][j] + p[15][j]))));
            break;
        case 37:
            for (int j = 0; j < cols; ++j) out[j] = (((p[0][j] + p[2][j]) + (p[6][j] + p[17][j])) - ((p[1][j] + (p[3][j] + p[4][j])) + (p[5][j] + (p[10][j] + p[15][j]))));
            break;
        case 38:
            for (int j = 0; j < cols; ++j) out[j] = ((((p[0][j] + p[4][j]) + (p[7][j] + p[8][j])) + ((p[10][j] + p[11][j]) + (p[13][j] + p[14][j]))) - (p[6][j] + p[9][j]));
            break;
        case 39:
            for (int j = 0; j < cols; ++j) out[j] = (((p[1][j] + p[2][j]) + (p[5][j] + (p[6][j] + p[17][j]))) - ((p[0][j] + p[3][j]) + (p[4][j] + (p[10][j] + p[15][j]))));
            break;
        }
    }
}

static void cob_336_c(int form, int rows, int cols, int ld,
    const float* const* blocks, float* dst, int dst_ld)
{
    const float* p[40];
    for (int row = 0; row < rows; ++row) {
        for (int index = 0; index < 40; ++index)
            p[index] = blocks[index] + (size_t)row * ld;
        float* const out = dst + (size_t)row * dst_ld;
        switch (form) {
        case 0:
            for (int j = 0; j < cols; ++j) out[j] = 0.125f * ((((p[1][j] + p[7][j]) + (p[18][j] + p[19][j])) + ((p[33][j] + p[36][j]) + (p[37][j] + (p[38][j] + p[39][j])))) - (((p[0][j] + (p[3][j] + p[4][j])) + ((p[11][j] + p[12][j]) + (p[15][j] + p[16][j]))) + (((p[17][j] + p[20][j]) + (p[21][j] + p[24][j])) + ((p[25][j] + p[32][j]) + (p[34][j] + p[35][j])))));
            break;
        case 1:
            for (int j = 0; j < cols; ++j) out[j] = 0.125f * ((((p[3][j] + (p[5][j] + p[7][j])) + ((p[10][j] + p[12][j]) + (p[17][j] + p[19][j]))) + (((p[22][j] + p[26][j]) + (p[27][j] + p[28][j])) + ((p[29][j] + p[30][j]) + (p[32][j] + p[34][j])))) - (((p[0][j] + p[9][j]) + (p[14][j] + p[23][j])) + ((p[24][j] + p[25][j]) + (p[31][j] + (p[35][j] + p[38][j])))));
            break;
        case 2:
            for (int j = 0; j < cols; ++j) out[j] = 0.125f * ((((p[8][j] + (p[15][j] + p[19][j])) + (p[22][j] + (p[24][j] + p[26][j]))) + ((p[27][j] + (p[28][j] + p[29][j])) + (p[30][j] + (p[34][j] + p[35][j])))) - (((p[1][j] + (p[2][j] + p[4][j])) + (p[6][j] + (p[11][j] + p[13][j]))) + ((p[17][j] + (p[23][j] + p[25][j])) + (p[31][j] + (p[32][j] + p[38][j])))));
            break;
        case 3:
            for (int j = 0; j < cols; ++j) out[j] = 0.125f * ((((p[4][j] + p[9][j]) + (p[10][j] + p[17][j])) + ((p[19][j] + p[24][j]) + (p[25][j] + (p[34][j] + p[38][j])))) - ((p[1][j] + (p[5][j] + p[11][j])) + ((p[14][j] + p[15][j]) + (p[32][j] + p[35][j]))));
            break;
        case 4:
            for (int j = 0; j < cols; ++j) out[j] = 0.125f * ((((p[10][j] + p[13][j]) + (p[14][j] + (p[18][j] + p[25][j]))) + ((p[33][j] + p[34][j]) + (p[36][j] + (p[37][j] + p[39][j])))) - (((p[2][j] + (p[5][j] + p[6][j])) + ((p[8][j] + p[9][j]) + (p[16][j] + p[17][j]))) + ((p[19][j] + (p[20][j] + p[21][j])) + ((p[24][j] + p[32][j]) + (p[35][j] + p[38][j])))));
            break;
        case 5:
            for (int j = 0; j < cols; ++j) out[j] = 0.125f * ((((p[0][j] + p[2][j]) + (p[3][j] + p[7][j])) + ((p[8][j] + p[13][j]) + (p[17][j] + p[24][j]))) - (((p[6][j] + p[12][j]) + (p[19][j] + p[25][j])) + ((p[32][j] + p[34][j]) + (p[35][j] + p[38][j]))));
            break;
        case 6:
            for (int j = 0; j < cols; ++j) out[j] = 0.125f * ((((p[1][j] + (p[3][j] + p[4][j])) + (p[15][j] + (p[16][j] + p[17][j]))) + ((p[18][j] + (p[19][j] + p[20][j])) + ((p[25][j] + p[33][j]) + (p[35][j] + p[36][j])))) - (((p[0][j] + p[7][j]) + (p[11][j] + (p[12][j] + p[21][j]))) + ((p[24][j] + (p[32][j] + p[34][j])) + (p[37][j] + (p[38][j] + p[39][j])))));
            break;
        case 7:
            for (int j = 0; j < cols; ++j) out[j] = 0.125f * ((((p[1][j] + (p[2][j] + p[8][j])) + (p[11][j] + (p[13][j] + p[15][j]))) + ((p[16][j] + (p[18][j] + p[20][j])) + ((p[21][j] + p[33][j]) + (p[37][j] + p[39][j])))) - (p[4][j] + (p[6][j] + p[36][j])));
            break;
        case 8:
            for (int j = 0; j < cols; ++j) out[j] = 0.125f * ((((p[0][j] + p[3][j]) + (p[5][j] + (p[7][j] + p[14][j]))) + ((p[20][j] + p[21][j]) + (p[33][j] + (p[36][j] + p[39][j])))) - ((p[9][j] + (p[10][j] + p[12][j])) + (p[16][j] + (p[18][j] + p[37][j]))));
            break;
        case 9:
            for (int j = 0; j < cols; ++j) out[j] = 0.125f * ((((p[0][j] + p[2][j]) + (p[6][j] + (p[13][j] + p[16][j]))) + ((p[21][j] + (p[23][j] + p[31][j])) + (p[33][j] + (p[36][j] + p[37][j])))) - (((p[3][j] + (p[7][j] + p[8][j])) + (p[12][j] + (p[18][j] + p[20][j]))) + ((p[22][j] + (p[26][j] + p[27][j])) + ((p[28][j] + p[29][j]) + (p[30][j] + p[39][j])))));
            break;
        case 10:
            for (int j = 0; j < cols; ++j) out[j] = 0.125f * ((((p[5][j] + (p[6][j] + p[8][j])) + ((p[9][j] + p[10][j]) + (p[13][j] + p[14][j]))) + ((p[20][j] + (p[24][j] + p[32][j])) + ((p[33][j] + p[34][j]) + (p[37][j] + p[38][j])))) - (((p[2][j] + p[16][j]) + (p[17][j] + (p[18][j] + p[19][j]))) + ((p[21][j] + p[25][j]) + (p[35][j] + (p[36][j] + p[39][j])))));
            break;
        case 11:
            for (int j = 0; j < cols; ++j) out[j] = 0.125f * ((((p[5][j] + (p[10][j] + p[15][j])) + ((p[16][j] + p[20][j]) + (p[22][j] + p[26][j]))) + ((p[27][j] + (p[28][j] + p[29][j])) + ((p[30][j] + p[36][j]) + (p[37][j] + p[39][j])))) - (((p[1][j] + p[4][j]) + (p[9][j] + (p[11][j] + p[14][j]))) + ((p[18][j] + p[21][j]) + (p[23][j] + (p[31][j] + p[33][j])))));
            break;
        case 12:
            for (int j = 0; j < cols; ++j) out[j] = 0.125f * ((((p[5][j] + p[6][j]) + (p[14][j] + p[22][j])) + ((p[23][j] + p[26][j]) + (p[28][j] + (p[30][j] + p[31][j])))) - ((p[2][j] + (p[8][j] + p[9][j])) + ((p[10][j] + p[13][j]) + (p[27][j] + p[29][j]))));
            break;
        case 13:
            for (int j = 0; j < cols; ++j) out[j] = 0.125f * (((((p[0][j] + p[5][j]) + (p[7][j] + p[9][j])) + ((p[10][j] + p[12][j]) + (p[14][j] + p[22][j]))) + (((p[23][j] + p[24][j]) + (p[27][j] + p[28][j])) + ((p[29][j] + p[31][j]) + (p[32][j] + (p[34][j] + p[38][j]))))) - ((p[3][j] + (p[17][j] + p[19][j])) + ((p[25][j] + p[26][j]) + (p[30][j] + p[35][j]))));
            break;
        case 14:
            for (int j = 0; j < cols; ++j) out[j] = 0.125f * ((((p[1][j] + (p[2][j] + p[4][j])) + (p[15][j] + (p[17][j] + p[19][j]))) + ((p[25][j] + (p[27][j] + p[28][j])) + (p[30][j] + (p[31][j] + p[35][j])))) - (((p[6][j] + (p[8][j] + p[11][j])) + (p[13][j] + (p[22][j] + p[23][j]))) + ((p[24][j] + (p[26][j] + p[29][j])) + (p[32][j] + (p[34][j] + p[38][j])))));
            break;
        case 15:
            for (int j = 0; j < cols; ++j) out[j] = 0.125f * ((((p[0][j] + (p[2][j] + p[3][j])) + (p[6][j] + (p[8][j] + p[12][j]))) + ((p[16][j] + (p[20][j] + p[21][j])) + ((p[23][j] + p[28][j]) + (p[29][j] + p[30][j])))) - (((p[7][j] + p[13][j]) + (p[18][j] + (p[22][j] + p[26][j]))) + ((p[27][j] + (p[31][j] + p[33][j])) + (p[36][j] + (p[37][j] + p[39][j])))));
            break;
        case 16:
            for (int j = 0; j < cols; ++j) out[j] = 0.125f * ((((p[1][j] + p[11][j]) + (p[12][j] + p[15][j])) + ((p[26][j] + p[28][j]) + (p[29][j] + p[31][j]))) - (((p[0][j] + p[3][j]) + (p[4][j] + p[7][j])) + ((p[22][j] + p[23][j]) + (p[27][j] + p[30][j]))));
            break;
        case 17:
            for (int j = 0; j < cols; ++j) out[j] = 0.125f * ((((p[4][j] + p[5][j]) + (p[9][j] + (p[11][j] + p[15][j]))) + ((p[16][j] + p[20][j]) + (p[21][j] + (p[22][j] + p[28][j])))) - (((p[1][j] + (p[10][j] + p[14][j])) + ((p[18][j] + p[23][j]) + (p[26][j] + p[27][j]))) + ((p[29][j] + (p[30][j] + p[31][j])) + ((p[33][j] + p[36][j]) + (p[37][j] + p[39][j])))));
            break;
        }
    }
}

#ifndef COB_336_MIN
#if defined(COB_USE_APPLE_AMX)
#define COB_336_MIN 4608
#else
#define COB_336_MIN 2304
#endif
#endif

static int cob_sgemm_336(int m, int n, int k, const float* a, int lda,
    const float* b, int ldb, float* c, int ldc)
{
    if (m < COB_336_MIN || m > 5184 || k != m || n != 2 * m || m % 192 != 0 ||
        !a || !b || !c || lda < k || ldb < n || ldc < n)
        return 0;
    const int dim = m / 3;
    const size_t tile = (size_t)dim * dim;
    const size_t bytes = 42 * tile * sizeof(float);
    if (bytes > ((size_t)512 << 20)) return 0;
    float* workspace = cob_aligned_alloc(128, bytes);
    if (!workspace) return 0;
    const float* left[9];
    const float* right[18];
    const float* products[40];
    for (int q = 0; q < 9; ++q)
        left[q] = a + (size_t)(q / 3) * dim * lda + (q % 3) * dim;
    for (int q = 0; q < 18; ++q)
        right[q] = b + (size_t)(q / 6) * dim * ldb + (q % 6) * dim;
    float* x = workspace;
    float* y = x + tile;
    for (int r = 0; r < 40; ++r) {
        float* product = y + (size_t)(r + 1) * tile;
        products[r] = product;
        cob_336_a(r, dim, dim, lda, left, x, dim);
        cob_336_b(r, dim, dim, ldb, right, y, dim);
#if defined(COB_USE_APPLE_AMX)
        int ok = 0;
#if defined(COB_WITH_STRASSEN_SME_LEAF) && defined(COB_USE_APPLE_SME)
        if (cob_apple_sme2p1_available())
            ok = cob_sgemm_strassen_sme_leaf(dim,dim,dim,x,dim,y,dim,product,dim);
#endif
        if (!ok) ok = cob_sgemm_rowmajor_amx(dim,dim,dim,x,dim,y,dim,product,dim);
        if (!ok) { free(workspace); return 0; }
#else
        cob_sgemm_rowmajor_fallback(dim,dim,dim,x,dim,y,dim,product,dim);
#endif
    }
    for (int q = 0; q < 18; ++q)
        cob_336_c(q, dim, dim, dim, products,
            c + (size_t)(q / 6) * dim * ldc + (q % 6) * dim, ldc);
    free(workspace);
    return 1;
}
