/* A four-depth microkernel fixes accumulator and input registers. Each full tile
   warms its next packed A row panel while consuming the current one. AVX2 uses
   L1 hints for the smaller panel; AVX-512 uses L2 to leave room for packed B. */
#define COB_HK_16_STEP(Q) \
        "vmovaps 0+64*(" Q ")(%[b]), %%ymm12\n\t" \
        "vmovaps 32+64*(" Q ")(%[b]), %%ymm13\n\t" \
        "vbroadcastss 0+24*(" Q ")(%[a]), %%ymm14\n\t" \
        "vbroadcastss 4+24*(" Q ")(%[a]), %%ymm15\n\t" \
        "vfmadd231ps %%ymm12, %%ymm14, %%ymm0\n\t" \
        "vfmadd231ps %%ymm13, %%ymm14, %%ymm1\n\t" \
        "vfmadd231ps %%ymm12, %%ymm15, %%ymm2\n\t" \
        "vfmadd231ps %%ymm13, %%ymm15, %%ymm3\n\t" \
        "vbroadcastss 8+24*(" Q ")(%[a]), %%ymm14\n\t" \
        "vbroadcastss 12+24*(" Q ")(%[a]), %%ymm15\n\t" \
        "vfmadd231ps %%ymm12, %%ymm14, %%ymm4\n\t" \
        "vfmadd231ps %%ymm13, %%ymm14, %%ymm5\n\t" \
        "vfmadd231ps %%ymm12, %%ymm15, %%ymm6\n\t" \
        "vfmadd231ps %%ymm13, %%ymm15, %%ymm7\n\t" \
        "vbroadcastss 16+24*(" Q ")(%[a]), %%ymm14\n\t" \
        "vbroadcastss 20+24*(" Q ")(%[a]), %%ymm15\n\t" \
        "vfmadd231ps %%ymm12, %%ymm14, %%ymm8\n\t" \
        "vfmadd231ps %%ymm13, %%ymm14, %%ymm9\n\t" \
        "vfmadd231ps %%ymm12, %%ymm15, %%ymm10\n\t" \
        "vfmadd231ps %%ymm13, %%ymm15, %%ymm11\n\t"

__attribute__((target("avx2,fma")))
static void cob_hk_reg_6x16(int k, const float* a, const float* b, float* c,
    int ldc, int resume, const float* next_a)
{
    float* row;
    const size_t stride = (size_t)ldc*sizeof(float);
    __asm__ volatile (
        "testl %[resume], %[resume]\n\t"
        "jz 1f\n\t"
        "movq %[c], %[row]\n\t"
        "vmovups (%[row]), %%ymm0\n\t"
        "vmovups 32(%[row]), %%ymm1\n\t"
        "addq %[stride], %[row]\n\t"
        "vmovups (%[row]), %%ymm2\n\t"
        "vmovups 32(%[row]), %%ymm3\n\t"
        "addq %[stride], %[row]\n\t"
        "vmovups (%[row]), %%ymm4\n\t"
        "vmovups 32(%[row]), %%ymm5\n\t"
        "addq %[stride], %[row]\n\t"
        "vmovups (%[row]), %%ymm6\n\t"
        "vmovups 32(%[row]), %%ymm7\n\t"
        "addq %[stride], %[row]\n\t"
        "vmovups (%[row]), %%ymm8\n\t"
        "vmovups 32(%[row]), %%ymm9\n\t"
        "addq %[stride], %[row]\n\t"
        "vmovups (%[row]), %%ymm10\n\t"
        "vmovups 32(%[row]), %%ymm11\n\t"
        "addq %[stride], %[row]\n\t"
        "jmp 2f\n\t"
        "1:\n\t"
        "vpxor %%ymm0, %%ymm0, %%ymm0\n\t"
        "vpxor %%ymm1, %%ymm1, %%ymm1\n\t"
        "vpxor %%ymm2, %%ymm2, %%ymm2\n\t"
        "vpxor %%ymm3, %%ymm3, %%ymm3\n\t"
        "vpxor %%ymm4, %%ymm4, %%ymm4\n\t"
        "vpxor %%ymm5, %%ymm5, %%ymm5\n\t"
        "vpxor %%ymm6, %%ymm6, %%ymm6\n\t"
        "vpxor %%ymm7, %%ymm7, %%ymm7\n\t"
        "vpxor %%ymm8, %%ymm8, %%ymm8\n\t"
        "vpxor %%ymm9, %%ymm9, %%ymm9\n\t"
        "vpxor %%ymm10, %%ymm10, %%ymm10\n\t"
        "vpxor %%ymm11, %%ymm11, %%ymm11\n\t"
        "2:\n\t"
        "cmpl $4, %[k]\n\t"
        "jl 4f\n\t"
        ".p2align 5\n\t"
        "3:\n\t"
        "prefetcht0 (%[next_a])\n\t"
        COB_HK_16_STEP("0")
        "prefetcht0 64(%[next_a])\n\t"
        COB_HK_16_STEP("1")
        COB_HK_16_STEP("2")
        COB_HK_16_STEP("3")
        "addq $96, %[a]\n\t"
        "addq $256, %[b]\n\t"
        "addq $96, %[next_a]\n\t"
        "subl $4, %[k]\n\t"
        "cmpl $4, %[k]\n\t"
        "jge 3b\n\t"
        "4:\n\t"
        "testl %[k], %[k]\n\t"
        "jz 6f\n\t"
        "5:\n\t"
        COB_HK_16_STEP("0")
        "addq $24, %[a]\n\t"
        "addq $64, %[b]\n\t"
        "decl %[k]\n\t"
        "jnz 5b\n\t"
        "6:\n\t"
        "movq %[c], %[row]\n\t"
        "vmovups %%ymm0, (%[row])\n\t"
        "vmovups %%ymm1, 32(%[row])\n\t"
        "addq %[stride], %[row]\n\t"
        "vmovups %%ymm2, (%[row])\n\t"
        "vmovups %%ymm3, 32(%[row])\n\t"
        "addq %[stride], %[row]\n\t"
        "vmovups %%ymm4, (%[row])\n\t"
        "vmovups %%ymm5, 32(%[row])\n\t"
        "addq %[stride], %[row]\n\t"
        "vmovups %%ymm6, (%[row])\n\t"
        "vmovups %%ymm7, 32(%[row])\n\t"
        "addq %[stride], %[row]\n\t"
        "vmovups %%ymm8, (%[row])\n\t"
        "vmovups %%ymm9, 32(%[row])\n\t"
        "addq %[stride], %[row]\n\t"
        "vmovups %%ymm10, (%[row])\n\t"
        "vmovups %%ymm11, 32(%[row])\n\t"
        "addq %[stride], %[row]\n\t"
        : [a] "+&r"(a), [b] "+&r"(b), [k] "+&r"(k), [row] "=&r"(row), [next_a] "+&r"(next_a)
        : [c] "r"(c), [stride] "r"(stride), [resume] "r"(resume)
        : "cc", "memory", "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6", "ymm7", "ymm8", "ymm9", "ymm10", "ymm11", "ymm12", "ymm13", "ymm14", "ymm15");
}
#undef COB_HK_16_STEP

#define COB_HK_32_STEP(Q) \
        "vmovaps 0+128*(" Q ")(%[b]), %%zmm24\n\t" \
        "vmovaps 64+128*(" Q ")(%[b]), %%zmm25\n\t" \
        "vbroadcastss 0+48*(" Q ")(%[a]), %%zmm26\n\t" \
        "vbroadcastss 4+48*(" Q ")(%[a]), %%zmm27\n\t" \
        "vfmadd231ps %%zmm24, %%zmm26, %%zmm0\n\t" \
        "vfmadd231ps %%zmm25, %%zmm26, %%zmm1\n\t" \
        "vfmadd231ps %%zmm24, %%zmm27, %%zmm2\n\t" \
        "vfmadd231ps %%zmm25, %%zmm27, %%zmm3\n\t" \
        "vbroadcastss 8+48*(" Q ")(%[a]), %%zmm26\n\t" \
        "vbroadcastss 12+48*(" Q ")(%[a]), %%zmm27\n\t" \
        "vfmadd231ps %%zmm24, %%zmm26, %%zmm4\n\t" \
        "vfmadd231ps %%zmm25, %%zmm26, %%zmm5\n\t" \
        "vfmadd231ps %%zmm24, %%zmm27, %%zmm6\n\t" \
        "vfmadd231ps %%zmm25, %%zmm27, %%zmm7\n\t" \
        "vbroadcastss 16+48*(" Q ")(%[a]), %%zmm26\n\t" \
        "vbroadcastss 20+48*(" Q ")(%[a]), %%zmm27\n\t" \
        "vfmadd231ps %%zmm24, %%zmm26, %%zmm8\n\t" \
        "vfmadd231ps %%zmm25, %%zmm26, %%zmm9\n\t" \
        "vfmadd231ps %%zmm24, %%zmm27, %%zmm10\n\t" \
        "vfmadd231ps %%zmm25, %%zmm27, %%zmm11\n\t" \
        "vbroadcastss 24+48*(" Q ")(%[a]), %%zmm26\n\t" \
        "vbroadcastss 28+48*(" Q ")(%[a]), %%zmm27\n\t" \
        "vfmadd231ps %%zmm24, %%zmm26, %%zmm12\n\t" \
        "vfmadd231ps %%zmm25, %%zmm26, %%zmm13\n\t" \
        "vfmadd231ps %%zmm24, %%zmm27, %%zmm14\n\t" \
        "vfmadd231ps %%zmm25, %%zmm27, %%zmm15\n\t" \
        "vbroadcastss 32+48*(" Q ")(%[a]), %%zmm26\n\t" \
        "vbroadcastss 36+48*(" Q ")(%[a]), %%zmm27\n\t" \
        "vfmadd231ps %%zmm24, %%zmm26, %%zmm16\n\t" \
        "vfmadd231ps %%zmm25, %%zmm26, %%zmm17\n\t" \
        "vfmadd231ps %%zmm24, %%zmm27, %%zmm18\n\t" \
        "vfmadd231ps %%zmm25, %%zmm27, %%zmm19\n\t" \
        "vbroadcastss 40+48*(" Q ")(%[a]), %%zmm26\n\t" \
        "vbroadcastss 44+48*(" Q ")(%[a]), %%zmm27\n\t" \
        "vfmadd231ps %%zmm24, %%zmm26, %%zmm20\n\t" \
        "vfmadd231ps %%zmm25, %%zmm26, %%zmm21\n\t" \
        "vfmadd231ps %%zmm24, %%zmm27, %%zmm22\n\t" \
        "vfmadd231ps %%zmm25, %%zmm27, %%zmm23\n\t"

__attribute__((target("avx512f,fma")))
static void cob_hk_reg_12x32(int k, const float* a, const float* b, float* c,
    int ldc, int resume, const float* next_a)
{
    float* row;
    const size_t stride = (size_t)ldc*sizeof(float);
    __asm__ volatile (
        "testl %[resume], %[resume]\n\t"
        "jz 1f\n\t"
        "movq %[c], %[row]\n\t"
        "vmovups (%[row]), %%zmm0\n\t"
        "vmovups 64(%[row]), %%zmm1\n\t"
        "addq %[stride], %[row]\n\t"
        "vmovups (%[row]), %%zmm2\n\t"
        "vmovups 64(%[row]), %%zmm3\n\t"
        "addq %[stride], %[row]\n\t"
        "vmovups (%[row]), %%zmm4\n\t"
        "vmovups 64(%[row]), %%zmm5\n\t"
        "addq %[stride], %[row]\n\t"
        "vmovups (%[row]), %%zmm6\n\t"
        "vmovups 64(%[row]), %%zmm7\n\t"
        "addq %[stride], %[row]\n\t"
        "vmovups (%[row]), %%zmm8\n\t"
        "vmovups 64(%[row]), %%zmm9\n\t"
        "addq %[stride], %[row]\n\t"
        "vmovups (%[row]), %%zmm10\n\t"
        "vmovups 64(%[row]), %%zmm11\n\t"
        "addq %[stride], %[row]\n\t"
        "vmovups (%[row]), %%zmm12\n\t"
        "vmovups 64(%[row]), %%zmm13\n\t"
        "addq %[stride], %[row]\n\t"
        "vmovups (%[row]), %%zmm14\n\t"
        "vmovups 64(%[row]), %%zmm15\n\t"
        "addq %[stride], %[row]\n\t"
        "vmovups (%[row]), %%zmm16\n\t"
        "vmovups 64(%[row]), %%zmm17\n\t"
        "addq %[stride], %[row]\n\t"
        "vmovups (%[row]), %%zmm18\n\t"
        "vmovups 64(%[row]), %%zmm19\n\t"
        "addq %[stride], %[row]\n\t"
        "vmovups (%[row]), %%zmm20\n\t"
        "vmovups 64(%[row]), %%zmm21\n\t"
        "addq %[stride], %[row]\n\t"
        "vmovups (%[row]), %%zmm22\n\t"
        "vmovups 64(%[row]), %%zmm23\n\t"
        "addq %[stride], %[row]\n\t"
        "jmp 2f\n\t"
        "1:\n\t"
        "vpxord %%zmm0, %%zmm0, %%zmm0\n\t"
        "vpxord %%zmm1, %%zmm1, %%zmm1\n\t"
        "vpxord %%zmm2, %%zmm2, %%zmm2\n\t"
        "vpxord %%zmm3, %%zmm3, %%zmm3\n\t"
        "vpxord %%zmm4, %%zmm4, %%zmm4\n\t"
        "vpxord %%zmm5, %%zmm5, %%zmm5\n\t"
        "vpxord %%zmm6, %%zmm6, %%zmm6\n\t"
        "vpxord %%zmm7, %%zmm7, %%zmm7\n\t"
        "vpxord %%zmm8, %%zmm8, %%zmm8\n\t"
        "vpxord %%zmm9, %%zmm9, %%zmm9\n\t"
        "vpxord %%zmm10, %%zmm10, %%zmm10\n\t"
        "vpxord %%zmm11, %%zmm11, %%zmm11\n\t"
        "vpxord %%zmm12, %%zmm12, %%zmm12\n\t"
        "vpxord %%zmm13, %%zmm13, %%zmm13\n\t"
        "vpxord %%zmm14, %%zmm14, %%zmm14\n\t"
        "vpxord %%zmm15, %%zmm15, %%zmm15\n\t"
        "vpxord %%zmm16, %%zmm16, %%zmm16\n\t"
        "vpxord %%zmm17, %%zmm17, %%zmm17\n\t"
        "vpxord %%zmm18, %%zmm18, %%zmm18\n\t"
        "vpxord %%zmm19, %%zmm19, %%zmm19\n\t"
        "vpxord %%zmm20, %%zmm20, %%zmm20\n\t"
        "vpxord %%zmm21, %%zmm21, %%zmm21\n\t"
        "vpxord %%zmm22, %%zmm22, %%zmm22\n\t"
        "vpxord %%zmm23, %%zmm23, %%zmm23\n\t"
        "2:\n\t"
        "cmpl $4, %[k]\n\t"
        "jl 4f\n\t"
        ".p2align 5\n\t"
        "3:\n\t"
        "prefetcht1 (%[next_a])\n\t"
        COB_HK_32_STEP("0")
        "prefetcht1 64(%[next_a])\n\t"
        COB_HK_32_STEP("1")
        "prefetcht1 128(%[next_a])\n\t"
        COB_HK_32_STEP("2")
        COB_HK_32_STEP("3")
        "addq $192, %[a]\n\t"
        "addq $512, %[b]\n\t"
        "addq $192, %[next_a]\n\t"
        "subl $4, %[k]\n\t"
        "cmpl $4, %[k]\n\t"
        "jge 3b\n\t"
        "4:\n\t"
        "testl %[k], %[k]\n\t"
        "jz 6f\n\t"
        "5:\n\t"
        COB_HK_32_STEP("0")
        "addq $48, %[a]\n\t"
        "addq $128, %[b]\n\t"
        "decl %[k]\n\t"
        "jnz 5b\n\t"
        "6:\n\t"
        "movq %[c], %[row]\n\t"
        "vmovups %%zmm0, (%[row])\n\t"
        "vmovups %%zmm1, 64(%[row])\n\t"
        "addq %[stride], %[row]\n\t"
        "vmovups %%zmm2, (%[row])\n\t"
        "vmovups %%zmm3, 64(%[row])\n\t"
        "addq %[stride], %[row]\n\t"
        "vmovups %%zmm4, (%[row])\n\t"
        "vmovups %%zmm5, 64(%[row])\n\t"
        "addq %[stride], %[row]\n\t"
        "vmovups %%zmm6, (%[row])\n\t"
        "vmovups %%zmm7, 64(%[row])\n\t"
        "addq %[stride], %[row]\n\t"
        "vmovups %%zmm8, (%[row])\n\t"
        "vmovups %%zmm9, 64(%[row])\n\t"
        "addq %[stride], %[row]\n\t"
        "vmovups %%zmm10, (%[row])\n\t"
        "vmovups %%zmm11, 64(%[row])\n\t"
        "addq %[stride], %[row]\n\t"
        "vmovups %%zmm12, (%[row])\n\t"
        "vmovups %%zmm13, 64(%[row])\n\t"
        "addq %[stride], %[row]\n\t"
        "vmovups %%zmm14, (%[row])\n\t"
        "vmovups %%zmm15, 64(%[row])\n\t"
        "addq %[stride], %[row]\n\t"
        "vmovups %%zmm16, (%[row])\n\t"
        "vmovups %%zmm17, 64(%[row])\n\t"
        "addq %[stride], %[row]\n\t"
        "vmovups %%zmm18, (%[row])\n\t"
        "vmovups %%zmm19, 64(%[row])\n\t"
        "addq %[stride], %[row]\n\t"
        "vmovups %%zmm20, (%[row])\n\t"
        "vmovups %%zmm21, 64(%[row])\n\t"
        "addq %[stride], %[row]\n\t"
        "vmovups %%zmm22, (%[row])\n\t"
        "vmovups %%zmm23, 64(%[row])\n\t"
        "addq %[stride], %[row]\n\t"
        : [a] "+&r"(a), [b] "+&r"(b), [k] "+&r"(k), [row] "=&r"(row), [next_a] "+&r"(next_a)
        : [c] "r"(c), [stride] "r"(stride), [resume] "r"(resume)
        : "cc", "memory", "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13", "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27");
}
#undef COB_HK_32_STEP

static void cob_hk_reg_apply(cob_blocked_kernel kernel, int m, int n, int k,
    const float* packed_a, const float* packed_b, float* c, int ldc, int resume)
{
    const int mr = kernel.mr, nr = kernel.nr;
    for (int j = 0; j < n; j += nr) {
        const int cols = cob_min_i32(nr, n-j);
        const float* b = packed_b + (size_t)j*k;
        for (int i = 0; i < m; i += mr) {
            const int rows = cob_min_i32(mr, m-i);
            const float* a = packed_a + (size_t)i*k;
            float* dst = c + (size_t)i*ldc+j;
            if (rows != mr || cols != nr) {
                cob_blocked_edge(kernel.tile, k, a, b, dst, ldc, rows, cols, nr, resume);
                continue;
            }
            /* A packed column pass restarts at row zero for the next B panel. */
            const float* next = i+mr < m ? a + (size_t)mr*k : packed_a;
            if (nr == 16) cob_hk_reg_6x16(k, a, b, dst, ldc, resume, next);
            else cob_hk_reg_12x32(k, a, b, dst, ldc, resume, next);
        }
    }
}
