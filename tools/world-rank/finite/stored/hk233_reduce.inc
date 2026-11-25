/* Generated from Sandia's BSD-2-licensed Hopcroft-Kerr table.
 * Copyright (c) 2014, Sandia Corporation. See LICENSE.fast-matmul. */
static void hk_reduce(int mh, int nh, int ldc, size_t pc,
                      const float *restrict product,
                      float *restrict c0, float *restrict c1, float *restrict c2,
                      float *restrict c3, float *restrict c4, float *restrict c5)
{
    for (int i = 0; i < mh; ++i) {
        const size_t ci = (size_t)i * ldc, pi = (size_t)i * nh;
        for (int j = 0; j < nh; ++j) {
            const float p1 = c3[ci + j];
            const float p2 = c0[ci + j];
            const float p3 = c1[ci + j];
            const float p4 = product[0 * pc + pi + j];
            const float p5 = product[1 * pc + pi + j];
            const float p6 = -product[2 * pc + pi + j];
            const float p7 = c2[ci + j];
            const float p8 = product[3 * pc + pi + j];
            const float p9 = c4[ci + j];
            const float p10 = product[7 * pc + pi + j];
            const float p11 = -product[8 * pc + pi + j];
            const float p12 = c5[ci + j];
            const float p13 = -product[6 * pc + pi + j];
            const float p14 = product[5 * pc + pi + j];
            const float p15 = product[4 * pc + pi + j];
            const float d = p8 - p6;
            const float e = p11 + p14;
            const float f = p1 + d;
            const float g = e + p13;
            const float h = p2 + p4;
            const float i0 = h + p5;
            const float z0 = f + i0;
            const float j1 = p3 + p4;
            const float z1 = j1 - g;
            const float j2 = p6 + p7;
            const float z2 = j2 + g;
            const float j3 = p10 - p15;
            const float z3 = f + j3;
            const float j4 = p3 + p9;
            const float k4 = p10 - p5;
            const float l4 = j4 + k4;
            const float z4 = l4 - e;
            const float j5 = p11 + p12;
            const float k5 = p15 - d;
            const float z5 = j5 + k5;
            c0[ci + j] = z0;
            c1[ci + j] = z1;
            c2[ci + j] = z2;
            c3[ci + j] = z3;
            c4[ci + j] = z4;
            c5[ci + j] = z5;
        }
    }
}
