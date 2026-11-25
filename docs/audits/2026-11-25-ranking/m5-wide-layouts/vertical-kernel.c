__attribute__((always_inline)) static inline void
compact_strip(int tiles,int m,int k,size_t depth,const float *ap,const float *b,int ldb,
              float *c,int ldc,svbool_t cols) __arm_streaming __arm_inout("za")
{
    const svbool_t full=svptrue_b32();
    const svcount_t four=svptrue_c32();
    svzero_za();
    int p=0;
#define STRIP_STEP(q) do {                                                                            \
    svfloat32_t bv=svld1_f32(cols,b+(size_t)(p+q)*ldb);                                              \
    svmopa_za32_f32_m(0,full,full,svget4_f32(a0,q),bv);                                              \
    svmopa_za32_f32_m(1,full,full,svget4_f32(a1,q),bv);                                              \
    if (tiles>=3) svmopa_za32_f32_m(2,full,full,svget4_f32(a2,q),bv);                                \
    if (tiles>=4) svmopa_za32_f32_m(3,full,full,svget4_f32(a3,q),bv);                                \
} while(0)
    for (;p+4<=k;p+=4) {
        const float *left=ap+(size_t)p*16;
        svfloat32x4_t a0=svld1_f32_x4(four,left);
        svfloat32x4_t a1=svld1_f32_x4(four,left+16*depth);
        svfloat32x4_t a2=tiles>=3?svld1_f32_x4(four,left+32*depth):a0;
        svfloat32x4_t a3=tiles>=4?svld1_f32_x4(four,left+48*depth):a0;
        STRIP_STEP(0); STRIP_STEP(1); STRIP_STEP(2); STRIP_STEP(3);
    }
#undef STRIP_STEP
    for (;p<k;++p) {
        const float *left=ap+(size_t)p*16;
        svfloat32_t bv=svld1_f32(cols,b+(size_t)p*ldb);
        svmopa_za32_f32_m(0,full,full,svld1_f32(full,left),bv);
        svmopa_za32_f32_m(1,full,full,svld1_f32(full,left+16*depth),bv);
        if (tiles>=3) svmopa_za32_f32_m(2,full,full,svld1_f32(full,left+32*depth),bv);
        if (tiles>=4) svmopa_za32_f32_m(3,full,full,svld1_f32(full,left+48*depth),bv);
    }
#define STORE_STRIP(tile) do {                                                                         \
    const int rows=m-(tile)*16<16?m-(tile)*16:16;                                                      \
    for (int i=0;i<rows;++i) svst1_hor_za32(tile,i,cols,c+(size_t)((tile)*16+i)*ldc);                \
} while(0)
    STORE_STRIP(0);
    STORE_STRIP(1);
    if (tiles>=3) STORE_STRIP(2);
    if (tiles>=4) STORE_STRIP(3);
#undef STORE_STRIP
}
