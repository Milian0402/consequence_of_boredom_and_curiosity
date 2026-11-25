    for (int j = 0; j < n; j += 256) {
        const int remaining = n-j < 256 ? n-j : 256;
        const int panels = (remaining+63)/64;
        for (int p = 0; p < k; ++p) {
            const float *right = b+(size_t)p*ldb+j;
            for (int q=0; q<panels; ++q) {
                const int w=remaining-64*q;
                svcount_t cols=w>=64?svptrue_c32():svwhilelt_c32_u64(0,(unsigned)w,4);
                svfloat32x4_t bv=svld1_f32_x4(cols,right+64*q);
                svst1_f32_x4(svptrue_c32(),bp+((size_t)q*k+p)*64,bv);
            }
        }
        for (int i = 0; i < m; i += 16) {
            const int h = m-i < 16 ? m-i : 16;
            for (int q=0; q<panels; ++q) {
                int w = remaining-q*64 < 64 ? remaining-q*64 : 64;
                float *dst=c+(size_t)i*ldc+j+q*64;
                svzero_za();
                micro_kernel_1x4(ap+i*depth,bp+(size_t)q*k*64,k);
                if (w==64 && h==16) store_za(dst,ldc);
                else cob_sme_transfer_short_c(h,w,dst,ldc,1);
            }
        }
    }
