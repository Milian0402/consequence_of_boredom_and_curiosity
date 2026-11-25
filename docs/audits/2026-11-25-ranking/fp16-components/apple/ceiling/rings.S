/* Both entries have an ordinary, non-streaming C interface. Each chip resets
   ZA, accumulates 256 iterations, and spans eight interleaved tile quartets. */
.text
.p2align 5
.macro kernel name, typ
.globl _\name
_\name:
    stp d8, d9, [sp, #-64]!
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]
    smstart
    ptrue p0.b
    fdup z0.\typ, #1.0
    fdup z1.\typ, #2.0
    fdup z2.\typ, #0.5
    fdup z3.\typ, #-0.5
    cntw x3
1:
    zero {za}
    mov w4, #256
2:
    .rept 8
        fmopa za0.s, p0/m, p0/m, z0.\typ, z0.\typ
        fmopa za1.s, p0/m, p0/m, z0.\typ, z1.\typ
        fmopa za2.s, p0/m, p0/m, z2.\typ, z0.\typ
        fmopa za3.s, p0/m, p0/m, z3.\typ, z1.\typ
    .endr
    subs w4, w4, #1
    b.ne 2b
    subs w0, w0, #1
    b.ne 1b
    mov w12, #0
3:
    mova z4.s, p0/m, za0h.s[w12, 0]
    st1w {z4.s}, p0, [x1]
    mova z4.s, p0/m, za1h.s[w12, 0]
    st1w {z4.s}, p0, [x1,#1,mul vl]
    mova z4.s, p0/m, za2h.s[w12, 0]
    st1w {z4.s}, p0, [x1,#2,mul vl]
    mova z4.s, p0/m, za3h.s[w12, 0]
    st1w {z4.s}, p0, [x1,#3,mul vl]
    addvl x1, x1, #4
    add w12, w12, #1
    cmp w12, w3
    b.ne 3b
    smstop
    ldp d10, d11, [sp, #16]
    ldp d12, d13, [sp, #32]
    ldp d14, d15, [sp, #48]
    ldp d8, d9, [sp], #64
    mov w0, w3
    ret
.endm
kernel ring32,s
.p2align 5
kernel ring16,h
