/* Fourteen rows keep 28 independent dot-product vectors live. Two vectors
   hold the current 32 columns of B and the last two broadcast two rows of A.
   The entire reduction stays in those registers across pairs of K steps. */
#define COB_T14_ROWS(OP) \
    OP(0,1) OP(2,3) OP(4,5) OP(6,7) OP(8,9) OP(10,11) OP(12,13) \
    OP(14,15) OP(16,17) OP(18,19) OP(20,21) OP(22,23) OP(24,25) OP(26,27)
#define COB_T14_ZERO(l,h) \
    "vpxord %%zmm" #l ", %%zmm" #l ", %%zmm" #l "\n\t" \
    "vpxord %%zmm" #h ", %%zmm" #h ", %%zmm" #h "\n\t"
#define COB_T14_LOAD(l,h) \
    "vmovups (%[row]), %%zmm" #l "\n\t" \
    "vmovups 64(%[row]), %%zmm" #h "\n\t" \
    "addq %[stride], %[row]\n\t"
#define COB_T14_STORE(l,h) \
    "vmovups %%zmm" #l ", (%[row])\n\t" \
    "vmovups %%zmm" #h ", 64(%[row])\n\t" \
    "addq %[stride], %[row]\n\t"
#define COB_T14_PAIR(off,r,s,l0,h0,l1,h1) \
    "vbroadcastss " #off "+" #r "*4(%[a]), %%zmm30\n\t" \
    "vbroadcastss " #off "+" #s "*4(%[a]), %%zmm31\n\t" \
    "vfmadd231ps %%zmm28, %%zmm30, %%zmm" #l0 "\n\t" \
    "vfmadd231ps %%zmm29, %%zmm30, %%zmm" #h0 "\n\t" \
    "vfmadd231ps %%zmm28, %%zmm31, %%zmm" #l1 "\n\t" \
    "vfmadd231ps %%zmm29, %%zmm31, %%zmm" #h1 "\n\t"
#define COB_T14_STEP(ao,bl,bh) \
    "vmovaps " #bl "(%[b]), %%zmm28\n\t" \
    "vmovaps " #bh "(%[b]), %%zmm29\n\t" \
    COB_T14_PAIR(ao,0,1,0,1,2,3) COB_T14_PAIR(ao,2,3,4,5,6,7) \
    COB_T14_PAIR(ao,4,5,8,9,10,11) COB_T14_PAIR(ao,6,7,12,13,14,15) \
    COB_T14_PAIR(ao,8,9,16,17,18,19) COB_T14_PAIR(ao,10,11,20,21,22,23) \
    COB_T14_PAIR(ao,12,13,24,25,26,27)
__attribute__((target("avx512f,fma")))
static void cob_x86_tile_14x32(int k, const float* a, const float* b,
    float* c, int ldc, int resume)
{
    float* row;
    const size_t stride = sizeof(float) * (size_t)ldc;
    __asm__ volatile (
        "testl %[resume], %[resume]\n\t"
        "jz 1f\n\t"
        "movq %[c], %[row]\n\t"
        COB_T14_ROWS(COB_T14_LOAD)
        "jmp 2f\n\t"
        "1:\n\t"
        COB_T14_ROWS(COB_T14_ZERO)
        "2:\n\t"
        "cmpl $2, %[k]\n\t"
        "jl 4f\n\t"
        ".p2align 5\n\t"
        "3:\n\t"
        COB_T14_STEP(0,0,64)
        COB_T14_STEP(56,128,192)
        "addq $112, %[a]\n\t"
        "addq $256, %[b]\n\t"
        "subl $2, %[k]\n\t"
        "cmpl $2, %[k]\n\t"
        "jge 3b\n\t"
        "4:\n\t"
        "testl %[k], %[k]\n\t"
        "jz 5f\n\t"
        COB_T14_STEP(0,0,64)
        "5:\n\t"
        "movq %[c], %[row]\n\t"
        COB_T14_ROWS(COB_T14_STORE)
        : [a] "+&r"(a), [b] "+&r"(b), [k] "+&r"(k), [row] "=&r"(row)
        : [c] "r"(c), [stride] "r"(stride), [resume] "r"(resume)
        : "memory","cc","zmm0","zmm1","zmm2","zmm3","zmm4","zmm5","zmm6","zmm7",
          "zmm8","zmm9","zmm10","zmm11","zmm12","zmm13","zmm14","zmm15","zmm16",
          "zmm17","zmm18","zmm19","zmm20","zmm21","zmm22","zmm23","zmm24","zmm25",
          "zmm26","zmm27","zmm28","zmm29","zmm30","zmm31");
}
#undef COB_T14_STEP
#undef COB_T14_PAIR
#undef COB_T14_STORE
#undef COB_T14_LOAD
#undef COB_T14_ZERO
#undef COB_T14_ROWS

/* Reserve the larger partial tile only in the selected 14-row kernel. */
static void cob_x86_edge_14x32(int k, const float* a, const float* b, float* c,
    int ldc, int rows, int cols, int resume)
{
    _Alignas(64) float edge[14 * 32] = {0};
    if (resume)
        for (int i = 0; i < rows; ++i)
            memcpy(edge + i*32, c + (size_t)i*ldc, (size_t)cols*sizeof(float));
    cob_x86_tile_14x32(k, a, b, edge, 32, resume);
    for (int i = 0; i < rows; ++i)
        memcpy(c + (size_t)i*ldc, edge + i*32, (size_t)cols*sizeof(float));
}

static void cob_x86_apply_14x32(int m, int n, int k, const float* a,
    const float* b, float* c, int ldc, int resume)
{
    for (int j = 0; j < n; j += 32) {
        const float* bp = b + (size_t)j*k;
        const int cols = cob_min_i32(32, n-j);
        for (int i = 0; i < m; i += 14) {
            const float* ap = a + (size_t)i*k;
            float* cp = c + (size_t)i*ldc + j;
            const int rows = cob_min_i32(14, m-i);
            if (rows == 14 && cols == 32) cob_x86_tile_14x32(k, ap, bp, cp, ldc, resume);
            else cob_x86_edge_14x32(k, ap, bp, cp, ldc, rows, cols, resume);
        }
    }
}
