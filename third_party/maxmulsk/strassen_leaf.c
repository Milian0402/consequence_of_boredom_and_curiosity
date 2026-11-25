// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ulaş Sertan KEMEÇ
// Adapted in November 2026 from MaxMulSK v5.1,
// https://github.com/UlasSertan/MaxMulSK/blob/7473af234eeaa79061bcc693acca77ad8a4e9c75/sme/v5/sme-1x4-fusedpack-v51.cpp
// COB changes: C11 entry point for strided Strassen half-products, fixed
// blocking and full-width B panels. See LICENSE and NOTICE in this directory.

#include <arm_sme.h>
#include <arm_sve.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#define RESTRICT restrict
#define MAXMULSK_V51_FILL 0
#define MAXMULSK_V51_FUSE 1
#define MAXMULSK_V51_PFA 1
#define MAXMULSK_V51_PFB 0

__attribute__((always_inline)) static inline size_t min_size(size_t a, size_t b)
    __arm_streaming_compatible { return a < b ? a : b; }

// Output group g of the vertical drain is four consecutive vertical slices of
// one ZA tile: tile g/4, first column 4*(g%4). Both must be compile-time
// constants -- the tile operand of svread_ver_za32 is an immediate.
#define FK_T(g) ((g) / 4)
#define FK_C(g) (4 * ((g) % 4))
#define FK_GROUP(g)                                                            \
svcreate4_f32(svread_ver_za32_f32_m(zu, pg, FK_T(g), FK_C(g) + 0),         \
              svread_ver_za32_f32_m(zu, pg, FK_T(g), FK_C(g) + 1),         \
              svread_ver_za32_f32_m(zu, pg, FK_T(g), FK_C(g) + 2),         \
              svread_ver_za32_f32_m(zu, pg, FK_T(g), FK_C(g) + 3))

// One steady step: store the group a Z set already holds, then refill that same
// Z set with the group two ahead. Store operand is consumed before the refill,
// so the two x4 store streams never wait on each other.
#define FK_DRAIN(g)                                                            \
svst1_f32_x4(p4, out + (g) * 64, s0);                                      \
s0 = FK_GROUP((g) + 2);                                                    \
svst1_f32_x4(p4, out + ((g) + 1) * 64, s1);                                \
s1 = FK_GROUP((g) + 3);

#define FK_PACK_A_KP_CHUNK(SRC, STRIDE)                                           \
do {                                                               \
    /* ---- HORIZONTAL FILL: memory -> Z -> ZA ------------- */    \
    /* No svzero_za here: every ZA element that is read back */    \
    /* below is written first -- all 16 slices of all four */      \
    /* tiles. Merging predication cannot expose stale ZA. */       \
    /* One x4 load brings the four consecutive 16-float */         \
    /* vectors of ONE source row, which land in tiles 0..3 at */   \
    /* the same slice. q0 and q1 are two independent four-Z */     \
    /* groups so a load is always in flight while ZA is */         \
    /* written. */                                                 \
    svfloat32x4_t q0 = svld1_f32_x4(p4, (SRC) + 0 * (STRIDE));             \
    svfloat32x4_t q1 = svld1_f32_x4(p4, (SRC) + 1 * (STRIDE));             \
    for (uint32_t r = 0; r < 14; r += 2) {                         \
        svwrite_hor_za32_f32_m(0, r, pg, svget4_f32(q0, 0));       \
        svwrite_hor_za32_f32_m(1, r, pg, svget4_f32(q0, 1));       \
        svwrite_hor_za32_f32_m(2, r, pg, svget4_f32(q0, 2));       \
        svwrite_hor_za32_f32_m(3, r, pg, svget4_f32(q0, 3));       \
        /* every component of q0 is consumed; refill it */         \
        q0 = svld1_f32_x4(p4, (SRC) + (r + 2) * (STRIDE));                 \
        svwrite_hor_za32_f32_m(0, r + 1, pg, svget4_f32(q1, 0));   \
        svwrite_hor_za32_f32_m(1, r + 1, pg, svget4_f32(q1, 1));   \
        svwrite_hor_za32_f32_m(2, r + 1, pg, svget4_f32(q1, 2));   \
        svwrite_hor_za32_f32_m(3, r + 1, pg, svget4_f32(q1, 3));   \
        q1 = svld1_f32_x4(p4, (SRC) + (r + 3) * (STRIDE));                 \
    }                                                              \
    /* DRAIN: rows 14 and 15 are already in q0/q1. No load of */   \
    /* rows 16/17 -- they do not exist in this 16-row panel. */    \
    svwrite_hor_za32_f32_m(0, 14, pg, svget4_f32(q0, 0));          \
    svwrite_hor_za32_f32_m(1, 14, pg, svget4_f32(q0, 1));          \
    svwrite_hor_za32_f32_m(2, 14, pg, svget4_f32(q0, 2));          \
    svwrite_hor_za32_f32_m(3, 14, pg, svget4_f32(q0, 3));          \
    svwrite_hor_za32_f32_m(0, 15, pg, svget4_f32(q1, 0));          \
    svwrite_hor_za32_f32_m(1, 15, pg, svget4_f32(q1, 1));          \
    svwrite_hor_za32_f32_m(2, 15, pg, svget4_f32(q1, 2));          \
    svwrite_hor_za32_f32_m(3, 15, pg, svget4_f32(q1, 3));          \
    /* ---- VERTICAL DRAIN: ZA -> Z -> packed_A ------------ */    \
    /* Reading tile t vertically is the transpose for free: */     \
    /* vertical slice c of tile t is A[m+0..15][kk+kp+16t+c], */   \
    /* which is exactly packed_A[(kp+16t+c)*16 + 0..15]. */        \
    /* Output group g = four vertical slices = 64 floats. */       \
    /* Groups 0-3 come from ZA0, 4-7 ZA1, 8-11 ZA2, 12-15 ZA3; */  \
    /* the pipeline is NOT restarted at a tile boundary. */        \
    float* const out = pA + kp * 16;                         \
    svfloat32x4_t s0 = FK_GROUP(0);                                \
    svfloat32x4_t s1 = FK_GROUP(1);                                \
    /* Static unroll: the tile operand of svread_ver_za32 must */  \
    /* be an immediate, so g cannot be a runtime variable. */      \
    FK_DRAIN(0)  FK_DRAIN(2)  FK_DRAIN(4)  FK_DRAIN(6)             \
    FK_DRAIN(8)  FK_DRAIN(10) FK_DRAIN(12)                         \
    /* DRAIN: s0 holds group 14 (ZA3 vertical 8..11), */           \
    /* s1 holds group 15 (ZA3 vertical 12..15). No further */      \
    /* extraction -- the two stores are back to back. */           \
    svst1_f32_x4(p4, out + 14 * 64, s0);                           \
    svst1_f32_x4(p4, out + 15 * 64, s1);                           \
} while (0)



// MICRO KERNEL 1x4-sym: SVL rows x 4*SVL cols using ZA accumulator.
// ZA_i = A x B_i  -- single A vec, 4 different B panels per k-step.
// K-unrolled by 4, software-pipelined as the A<->B mirror of 4x1.
//
// Per group of 4 k-steps:
//   1 A x4 load   (covers 4 k-steps' SVL-wide A vecs)
//   4 B x4 loads  (one per k-step, each = 4 B panels)
//   16 svmopa
//
// Schedule per 4 k-steps:
//   PROLOGUE: load A x4 (k+0..k+3), load B x4 (k+0)
//   group k+0: ZA0(a0,b0), ZA1(a0,b1), [load B(k+1)], ZA2(a0,b2), ZA3(a0,b3)
//   group k+1: ZA0(a1,b0), ZA1(a1,b1), [load B(k+2)], ZA2(a1,b2), ZA3(a1,b3)
//   group k+2: ZA0(a2,b0), ZA1(a2,b1), [load B(k+3)], ZA2(a2,b2), ZA3(a2,b3)
//   group k+3: ZA0(a3,b0), ZA1(a3,b1), [load next A x4], ZA2(a3,b2), ZA3(a3,b3)
//   end-of-iter: load next iter's first B x4
// Same-tile write gap: 4 svmopa slots + 1 load (matches 4x1).
// =========================================================================
__attribute__((noinline))
static void micro_kernel_1x4(float* RESTRICT packed_A, float* RESTRICT packed_B,
                      size_t K_curr) __arm_inout("za") __arm_streaming {
    const size_t SVL = (size_t)svcntsw();
    const svbool_t pg = svptrue_b32();
    const svcount_t pg4 = svptrue_c32();
    const size_t GS_B = 4 * SVL; // B group stride per k-step

    const float* pA = packed_A;
    const float* pB = packed_B;

    const size_t K_main = (K_curr / 4) * 4;

    if (K_main >= 4) {
        // PROLOGUE: load first A x4 (covers k+0..k+3) and first B x4 (k+0)
        svfloat32x4_t a_x4 = svld1_f32_x4(pg4, pA); pA += 4 * SVL;
        svfloat32_t a0 = svget4_f32(a_x4, 0);
        svfloat32_t a1 = svget4_f32(a_x4, 1);
        svfloat32_t a2 = svget4_f32(a_x4, 2);
        svfloat32_t a3 = svget4_f32(a_x4, 3);

        svfloat32x4_t b_x4 = svld1_f32_x4(pg4, pB); pB += GS_B;

        for (size_t k = 0; k < K_main - 4; k += 4) {
            // Extract current B panels (k-step k+0)
            svfloat32_t b0 = svget4_f32(b_x4, 0);
            svfloat32_t b1 = svget4_f32(b_x4, 1);
            svfloat32_t b2 = svget4_f32(b_x4, 2);
            svfloat32_t b3 = svget4_f32(b_x4, 3);

            // group k+0: ZA0, ZA1, [load B(k+1)], ZA2, ZA3
            svmopa_za32_f32_m(0, pg, pg, a0, b0);
            svmopa_za32_f32_m(1, pg, pg, a0, b1);
            b_x4 = svld1_f32_x4(pg4, pB); pB += GS_B;
            svmopa_za32_f32_m(2, pg, pg, a0, b2);
            svmopa_za32_f32_m(3, pg, pg, a0, b3);

            b0 = svget4_f32(b_x4, 0);
            b1 = svget4_f32(b_x4, 1);
            b2 = svget4_f32(b_x4, 2);
            b3 = svget4_f32(b_x4, 3);

            // group k+1: ZA0, ZA1, [load B(k+2)], ZA2, ZA3
            svmopa_za32_f32_m(0, pg, pg, a1, b0);
            svmopa_za32_f32_m(1, pg, pg, a1, b1);
            b_x4 = svld1_f32_x4(pg4, pB); pB += GS_B;
            svmopa_za32_f32_m(2, pg, pg, a1, b2);
            svmopa_za32_f32_m(3, pg, pg, a1, b3);

            b0 = svget4_f32(b_x4, 0);
            b1 = svget4_f32(b_x4, 1);
            b2 = svget4_f32(b_x4, 2);
            b3 = svget4_f32(b_x4, 3);

            // group k+2: ZA0, ZA1, [load B(k+3)], ZA2, ZA3
            svmopa_za32_f32_m(0, pg, pg, a2, b0);
            svmopa_za32_f32_m(1, pg, pg, a2, b1);
            b_x4 = svld1_f32_x4(pg4, pB); pB += GS_B;
            svmopa_za32_f32_m(2, pg, pg, a2, b2);
            svmopa_za32_f32_m(3, pg, pg, a2, b3);

            b0 = svget4_f32(b_x4, 0);
            b1 = svget4_f32(b_x4, 1);
            b2 = svget4_f32(b_x4, 2);
            b3 = svget4_f32(b_x4, 3);

            // group k+3: ZA0, ZA1, [load next A x4], ZA2, ZA3
            svmopa_za32_f32_m(0, pg, pg, a3, b0);
            svmopa_za32_f32_m(1, pg, pg, a3, b1);
            a_x4 = svld1_f32_x4(pg4, pA); pA += 4 * SVL;
            svmopa_za32_f32_m(2, pg, pg, a3, b2);
            svmopa_za32_f32_m(3, pg, pg, a3, b3);

            a0 = svget4_f32(a_x4, 0);
            a1 = svget4_f32(a_x4, 1);
            a2 = svget4_f32(a_x4, 2);
            a3 = svget4_f32(a_x4, 3);
            b_x4 = svld1_f32_x4(pg4, pB); pB += GS_B; // prefetch next iter's first B
        }

        // EPILOGUE: last group of 4 (no next prefetch)
        {
            svfloat32_t b0 = svget4_f32(b_x4, 0);
            svfloat32_t b1 = svget4_f32(b_x4, 1);
            svfloat32_t b2 = svget4_f32(b_x4, 2);
            svfloat32_t b3 = svget4_f32(b_x4, 3);

            svmopa_za32_f32_m(0, pg, pg, a0, b0);
            svmopa_za32_f32_m(1, pg, pg, a0, b1);
            svmopa_za32_f32_m(2, pg, pg, a0, b2);
            svmopa_za32_f32_m(3, pg, pg, a0, b3);

            b_x4 = svld1_f32_x4(pg4, pB); pB += GS_B;
            b0 = svget4_f32(b_x4, 0); b1 = svget4_f32(b_x4, 1);
            b2 = svget4_f32(b_x4, 2); b3 = svget4_f32(b_x4, 3);

            svmopa_za32_f32_m(0, pg, pg, a1, b0);
            svmopa_za32_f32_m(1, pg, pg, a1, b1);
            svmopa_za32_f32_m(2, pg, pg, a1, b2);
            svmopa_za32_f32_m(3, pg, pg, a1, b3);

            b_x4 = svld1_f32_x4(pg4, pB); pB += GS_B;
            b0 = svget4_f32(b_x4, 0); b1 = svget4_f32(b_x4, 1);
            b2 = svget4_f32(b_x4, 2); b3 = svget4_f32(b_x4, 3);

            svmopa_za32_f32_m(0, pg, pg, a2, b0);
            svmopa_za32_f32_m(1, pg, pg, a2, b1);
            svmopa_za32_f32_m(2, pg, pg, a2, b2);
            svmopa_za32_f32_m(3, pg, pg, a2, b3);

            b_x4 = svld1_f32_x4(pg4, pB); pB += GS_B;
            b0 = svget4_f32(b_x4, 0); b1 = svget4_f32(b_x4, 1);
            b2 = svget4_f32(b_x4, 2); b3 = svget4_f32(b_x4, 3);

            svmopa_za32_f32_m(0, pg, pg, a3, b0);
            svmopa_za32_f32_m(1, pg, pg, a3, b1);
            svmopa_za32_f32_m(2, pg, pg, a3, b2);
            svmopa_za32_f32_m(3, pg, pg, a3, b3);
        }
    }

    // K TAIL: remaining 0-3 iterations, one at a time.
    // Mirror of 4x1 K-tail: 1 A x1 load + 1 B x4 load + 4 svmopa.
    for (size_t k = K_main; k < K_curr; k++) {
        svfloat32_t a = svld1_f32(pg, pA); pA += SVL;
        svfloat32x4_t b_x4 = svld1_f32_x4(pg4, pB); pB += GS_B;
        svmopa_za32_f32_m(0, pg, pg, a, svget4_f32(b_x4, 0));
        svmopa_za32_f32_m(1, pg, pg, a, svget4_f32(b_x4, 1));
        svmopa_za32_f32_m(2, pg, pg, a, svget4_f32(b_x4, 2));
        svmopa_za32_f32_m(3, pg, pg, a, svget4_f32(b_x4, 3));
    }

}

// ZA -> C writeback, lifted out of the micro-kernel.
//
// This OVERWRITES C rather than accumulating into it. That is sound here and
// not in 1x4-sym: with K innermost, each C tile is stored exactly once with
// the complete K sum, so there is nothing in C to preserve. Dropping the
// read-modify-write removes 64 loads and 64 adds per tile (4 KiB of reads).
//
// CONSEQUENCE: run_multiplication computes C = A*B, not C += A*B. The caller
// does NOT need to zero C first. This differs from the other MaxMulSK SME
// kernels, which all accumulate.
// Tiles stack horizontally: ZA_i -> C[0..SVL, i*SVL..(i+1)*SVL]
// =========================================================================
// v5 (2026-09-27): each ZA slice is stored straight to memory with ST1W
// (svst1_hor_za32), 4 x 64 B per C row, instead of 4 MOVAs into Z registers
// and one 256 B x4 store. Measured: writeback alone 110-118 ns -> 25-36
// ns per tile; zero + micro-kernel + writeback -34% at K=64, -14% at K=256,
// -2.7% at K=1024. Output is bit-identical (it is the same data, moved by a
// different instruction).
// Invariant: ZA tile t slice i holds C row i, columns 16t..16t+15 (SVL = 16
// fp32 lanes, checked by the driver); all 16 rows x 64 floats are writable.
__attribute__((noinline))
static void store_za(float* RESTRICT C, size_t wide_of_C) __arm_in("za") __arm_streaming {
    const svbool_t pg = svptrue_b32();
    for (uint32_t i = 0; i < 16; i++) {
        float* const row = C + i * wide_of_C;
        svst1_hor_za32(0, i, pg, row + 0);
        svst1_hor_za32(1, i, pg, row + 16);
        svst1_hor_za32(2, i, pg, row + 32);
        svst1_hor_za32(3, i, pg, row + 48);
    }
}

// Later K panels: C += ZA without MOVA (fixed v5 behaviour since 2026-09-27).
// svadd_za32_f32_vg1x4(s, z) adds z0..z3 to the ZA
// array vectors of the 4-vector group at index s. With SVL = 16 fp32 lanes the
// ZA array has 64 vectors and a 32-bit tile t slice i is array vector 4*i + t,
// so the group at s = 4*i + t is slices i, i+4, i+8, i+12 of tile t. ZA holds
// this K panel's product, so the result is C + (p0 + p1 + ...) as in v4 and C
// stays bit-identical to v4 (checked on every measured shape).
// Invariant: C points at a full 16x64 tile (row stride wide_of_C) and ZA holds
// this panel's finished product; afterwards ZA holds the tile's running sum.
__attribute__((noinline))
static void add_c_into_za(const float* RESTRICT C, size_t wide_of_C) __arm_inout("za") __arm_streaming {
    const svbool_t pg = svptrue_b32();
    for (uint32_t t = 0; t < 4; t++)
        for (uint32_t i = 0; i < 4; i++) {
            const float* const c = C + t * 16;
            const svfloat32x4_t z = svcreate4_f32(
                svld1_f32(pg, c + (i + 0) * wide_of_C), svld1_f32(pg, c + (i + 4) * wide_of_C),
                svld1_f32(pg, c + (i + 8) * wide_of_C), svld1_f32(pg, c + (i + 12) * wide_of_C));
            svadd_za32_f32_vg1x4(4 * i + t, z);
        }
}

// Pulls the next full C tile's 16 rows (256 B each, two 128 B lines) towards L2
// while this tile computes, so add_c_into_za does not wait on memory. Without it
// reading C at the tile's end cost 3-5% on multi-panel shapes.
__attribute__((always_inline)) static inline void prefetch_c_tile(const float* C, size_t wide_of_C) __arm_streaming_compatible {
    for (size_t i = 0; i < 16; i++) {
        const float* row = C + i * wide_of_C;
        __asm__ volatile("prfm pldl2keep, [%0]\n\tprfm pldl2keep, [%0, #128]" :: "r"(row));
    }
}

// Edge tiles only: drain the full 16x64 accumulator into a dense scratch
// (row stride 64), from which the driver copies just the valid rows and
// columns. Same ST1W body as store_za.
__attribute__((noinline))
static void store_za_tile(float* RESTRICT scratch) __arm_in("za") __arm_streaming {
    store_za(scratch, 64);
}

#define FK_BSTEP()                                                             \
do {                                                                       \
    svst1_f32_x4(p4, bd_prev, bp);                                         \
    bp = svld1_f32_x4(p4, bs);                                             \
    bd_prev = bd; bleft--;                                                 \
    if (bleft) FK_BADVANCE();                                                         \
} while (0)

// Moves the B cursor (bs source, bd destination) to the next 64-float piece, in
// row order: panel index bi runs 0..npanels-1 across row k (source +64,
// destination +Kc*64 to the next panel slot), then wraps to row k+1 (source +N
// from the row start, destination +64 from the row's panel-0 slot). Each B row
// is therefore read as one contiguous Nc-float run (column order
// cost 16-26% once the B row stride reached the 16 KiB page).
#define FK_BADVANCE()                                                          \
do {                                                                       \
    if (++bi == npanels) { bi = 0; bsrc_row += N; bs = bsrc_row; bd_row += 64; bd = bd_row; } \
    else { bs += 64; bd += Kc * 64; }                                      \
    if (MAXMULSK_V51_PFB) __asm__ volatile("prfm pldl2keep, [%0]" :: "r"(bs + MAXMULSK_V51_PFB * N)); \
} while (0)

// Prefetch of the A rows' next chunk(s), issued next to the FILL loads.
#define FK_PFA_ROW(ROWPTR)                                                     \
do { if (MAXMULSK_V51_PFA) __asm__ volatile("prfm pldl2keep, [%0]" :: "r"((ROWPTR) + MAXMULSK_V51_PFA * 64)); } while (0)

#define FK_SLOT(i)                                                             \
do { for (uint32_t q_ = quota[i]; q_ && bleft; q_--) FK_BSTEP(); } while (0)

#if MAXMULSK_V51_FILL
#define FK_FILL_SLOT(i) FK_SLOT(i)
#else
#define FK_FILL_SLOT(i) ((void)0)
#endif

// FK_PACK_A_KP_CHUNK with B slots: 0-6 after each FILL step, 7 after the FILL
// tail, 8-14 after each DRAIN step, 15 after the final stores. The A work is
// the same instruction sequence as the original macro. ZA is scratch only:
// every slice read in the DRAIN was written in this chunk's FILL, so nothing is
// live across chunks and no zeroing is needed.
#define FK_PACK_A_KP_CHUNK_FUSED(SRC, STRIDE)                                  \
do {                                                                       \
    svfloat32x4_t q0 = svld1_f32_x4(p4, (SRC) + 0 * (STRIDE));             \
    svfloat32x4_t q1 = svld1_f32_x4(p4, (SRC) + 1 * (STRIDE));             \
    for (uint32_t r = 0; r < 14; r += 2) {                                 \
        svwrite_hor_za32_f32_m(0, r, pg, svget4_f32(q0, 0));               \
        svwrite_hor_za32_f32_m(1, r, pg, svget4_f32(q0, 1));               \
        svwrite_hor_za32_f32_m(2, r, pg, svget4_f32(q0, 2));               \
        svwrite_hor_za32_f32_m(3, r, pg, svget4_f32(q0, 3));               \
        q0 = svld1_f32_x4(p4, (SRC) + (r + 2) * (STRIDE));                 \
        svwrite_hor_za32_f32_m(0, r + 1, pg, svget4_f32(q1, 0));           \
        svwrite_hor_za32_f32_m(1, r + 1, pg, svget4_f32(q1, 1));           \
        svwrite_hor_za32_f32_m(2, r + 1, pg, svget4_f32(q1, 2));           \
        svwrite_hor_za32_f32_m(3, r + 1, pg, svget4_f32(q1, 3));           \
        q1 = svld1_f32_x4(p4, (SRC) + (r + 3) * (STRIDE));                 \
        FK_PFA_ROW((SRC) + r * (STRIDE));                                  \
        FK_PFA_ROW((SRC) + (r + 1) * (STRIDE));                            \
        FK_FILL_SLOT(r / 2);                                               \
    }                                                                      \
    FK_PFA_ROW((SRC) + 14 * (STRIDE));                                     \
    FK_PFA_ROW((SRC) + 15 * (STRIDE));                                     \
    svwrite_hor_za32_f32_m(0, 14, pg, svget4_f32(q0, 0));                  \
    svwrite_hor_za32_f32_m(1, 14, pg, svget4_f32(q0, 1));                  \
    svwrite_hor_za32_f32_m(2, 14, pg, svget4_f32(q0, 2));                  \
    svwrite_hor_za32_f32_m(3, 14, pg, svget4_f32(q0, 3));                  \
    svwrite_hor_za32_f32_m(0, 15, pg, svget4_f32(q1, 0));                  \
    svwrite_hor_za32_f32_m(1, 15, pg, svget4_f32(q1, 1));                  \
    svwrite_hor_za32_f32_m(2, 15, pg, svget4_f32(q1, 2));                  \
    svwrite_hor_za32_f32_m(3, 15, pg, svget4_f32(q1, 3));                  \
    FK_FILL_SLOT(7);                                                       \
    float* const out = pA + kp * 16;                                       \
    svfloat32x4_t s0 = FK_GROUP(0);                                        \
    svfloat32x4_t s1 = FK_GROUP(1);                                        \
    FK_DRAIN(0)  FK_SLOT(8);  FK_DRAIN(2)  FK_SLOT(9);                     \
    FK_DRAIN(4)  FK_SLOT(10); FK_DRAIN(6)  FK_SLOT(11);                    \
    FK_DRAIN(8)  FK_SLOT(12); FK_DRAIN(10) FK_SLOT(13);                    \
    FK_DRAIN(12) FK_SLOT(14);                                              \
    svst1_f32_x4(p4, out + 14 * 64, s0);                                   \
    svst1_f32_x4(p4, out + 15 * 64, s1);                                   \
    FK_SLOT(15);                                                           \
} while (0)

// Edge A chunk (rows < 16 or kr < 64), verbatim from the original driver: a
// dense zero-filled 16x64 staging block packed with the same macro at stride 64.
// Zeroing uses predicated SVE stores, never memset, which would lower to
// __arm_sc_memset inside streaming code.
#define FK_PACK_A_EDGE_CHUNK()                                                 \
do {                                                                       \
    for (size_t r = 0; r < 16; r++) {                                      \
        const size_t have = (r < rows) ? kr : 0;                           \
        const float* srow = arow_base + (r < rows ? r : 0) * K + kp;       \
        float* drow = stage + r * 64;                                      \
        for (size_t j = 0; j < 64; j += SVL) {                             \
            const svbool_t pv = svwhilelt_b32_u64(j, have);                \
            svst1_f32(pg, drow + j, svld1_f32(pv, srow + (j < have ? j : 0)));              \
        }                                                                  \
    }                                                                      \
    FK_PACK_A_KP_CHUNK(stage, 64);                                         \
} while (0)

// Packs the Mc block starting at row mm into packed_A, one 16-row slot per
// (ir/16) at stride Kc_pad*16 -- the slot layout of the original driver.
// ZA is used as transpose scratch only (see the chunk macros).
__attribute__((always_inline)) static inline
void pack_A_block(const float* A, size_t K, size_t mm, size_t mcl,
                  size_t kk, size_t kcl, size_t Kc_pad,
                  float* packed_A, float* stage) __arm_streaming __arm_inout("za") {
    const svbool_t  pg = svptrue_b32();
    const svcount_t p4 = svptrue_c32();
    const svfloat32_t zu = svundef_f32();
    const size_t SVL = (size_t)svcntsw();
    for (size_t ir = 0; ir < mcl; ir += 16) {
        float* const pA = packed_A + (ir / 16) * (Kc_pad * 16);
        const size_t rows = min_size(16, mcl - ir);
        const float* const arow_base = A + (mm + ir) * K + kk;
        for (size_t kp = 0; kp < kcl; kp += 64) {
            const size_t kr = min_size(64, kcl - kp);
            if (rows == 16 && kr == 64) {
                const float* const arow = arow_base + kp;
                FK_PACK_A_KP_CHUNK(arow, K);
            } else {
                FK_PACK_A_EDGE_CHUNK();
            }
        }
    }
}

// Packs the whole (kcl x ncl) B block at columns nn into packed_B AND the Mc
// block at row 0 into packed_A, with B rows spread over the slots of every
// FULL A chunk. Rows left over when the full chunks run out (or all of B,
// when there is no full chunk) are copied after the A work.
__attribute__((always_inline)) static inline
void fuse_pack(const float* A, const float* B, size_t K, size_t N,
               size_t mcl, size_t nn, size_t ncl, size_t kk, size_t kcl,
               size_t Kc, size_t Kc_pad,
               float* packed_A, float* packed_B, float* stage) __arm_streaming __arm_inout("za") {
    const svbool_t  pg = svptrue_b32();
    const svcount_t p4 = svptrue_c32();
    const svfloat32_t zu = svundef_f32();
    const size_t SVL = (size_t)svcntsw();

    const size_t npanels = (ncl + 63) / 64;
    const size_t full_chunks = (mcl / 16) * (kcl / 64);
    const size_t brows = npanels * kcl;

    uint32_t quota[16] = {0};
    if (full_chunks && MAXMULSK_V51_FUSE) {
        const uint32_t R = (uint32_t)((brows + full_chunks - 1) / full_chunks);
        const uint32_t first = MAXMULSK_V51_FILL ? 0 : 8;
        const uint32_t nslots = 16 - first;
        for (uint32_t i = 0; i < nslots; i++)
            quota[first + i] = R / nslots + (i < R % nslots ? 1 : 0);
    }

    const float* bsrc_row = B + kk * N + nn;
    const float* bs = bsrc_row;
    float* bd_row = packed_B;
    float* bd = bd_row;
    float* bd_prev = bd;
    size_t bi = 0;
    size_t bleft = brows;
    // Prime the pipeline: load the first piece without storing anything.
    svfloat32x4_t bp = svld1_f32_x4(p4, bs);
    bd_prev = bd; bleft--;
    if (bleft) FK_BADVANCE();

    for (size_t ir = 0; ir < mcl; ir += 16) {
        float* const pA = packed_A + (ir / 16) * (Kc_pad * 16);
        const size_t rows = min_size(16, mcl - ir);
        const float* const arow_base = A + ir * K + kk;
        for (size_t kp = 0; kp < kcl; kp += 64) {
            const size_t kr = min_size(64, kcl - kp);
            if (rows == 16 && kr == 64) {
                const float* const arow = arow_base + kp;
                FK_PACK_A_KP_CHUNK_FUSED(arow, K);
            } else {
                FK_PACK_A_EDGE_CHUNK();
            }
        }
    }

    while (bleft) FK_BSTEP();
    svst1_f32_x4(p4, bd_prev, bp);
}

// A run of source rows to prefetch, walked in 128-byte lines. Invariant: the rows
// [base, base + rows*ld) with lines_per_row lines each lie inside the caller's A or B
// matrix (built from the same bounds the next pack will read); prfm never faults, but
// no address outside the matrix is ever formed.
typedef struct { const float* base; size_t ld, lines_per_row, total, next, per_tile; } PfCursor;
__attribute__((always_inline)) static inline
void pf_set(PfCursor* c, const float* base, size_t ld, size_t rows, size_t row_floats, size_t tiles) __arm_streaming_compatible {
    c->base = base; c->ld = ld; c->lines_per_row = (row_floats + 31) / 32; c->total = rows * c->lines_per_row; c->next = 0;
    c->per_tile = tiles ? (c->total + tiles - 1) / tiles : c->total;
}
__attribute__((always_inline)) static inline void pf_issue(PfCursor* c) __arm_streaming_compatible {
    for (size_t i = 0; i < c->per_tile && c->next < c->total; i++, c->next++) {
        const float* a = c->base + (c->next / c->lines_per_row) * c->ld + (c->next % c->lines_per_row) * 32;
        __asm__ volatile("prfm pldl2keep, [%0]" :: "r"(a));
    }
}

// Source of the A pack that follows the Mc block at row mm of (kk, nn): the next Mc block
// of the same (kk, nn), else the first A block of the next (kk, nn); nothing after the
// last. Rows / columns are clamped exactly as the packs clamp them.
__attribute__((always_inline)) static inline
void pf_next_a(PfCursor* c, const float* A, size_t lda, size_t M, size_t K, size_t N, size_t Mc, size_t Kc, size_t Nc,
               size_t kk, size_t nn, size_t mm, size_t tiles) __arm_streaming_compatible {
    size_t r0, rows, k0;
    if (mm + Mc < M) { r0 = mm + Mc; rows = min_size(Mc, M - r0); k0 = kk; }
    else if (nn + Nc < N) { r0 = 0; rows = min_size(Mc, M); k0 = kk; }
    else if (kk + Kc < K) { r0 = 0; rows = min_size(Mc, M); k0 = kk + Kc; }
    else { c->total = 0; c->next = 0; return; }
    pf_set(c, A + r0 * lda + k0, lda, rows, min_size(Kc, K - k0), tiles);
}

// Computes the Mc block at row mm against the packed Nc block, in the same
// tile order as the original driver (64-column panel outer, 16-row panel
// inner) with the same ZA zeroing and the same full/edge writeback, so C is
// bit-identical. packed_A / packed_B must already hold this block.
__attribute__((always_inline)) static inline
void compute_block(float* C, size_t N, size_t mm, size_t mcl,
                   size_t nn, size_t ncl, size_t kcl, bool first_k,
                   size_t Kc, size_t Kc_pad,
                   float* packed_A, float* packed_B, float* scratch,
                   PfCursor* pfa) __arm_streaming __arm_inout("za") {
    const size_t SVL = (size_t)svcntsw();
    for (size_t nr = 0; nr < ncl; nr += 64) {
        float* const pB = packed_B + (nr / 64) * (Kc * 64);
        for (size_t ir = 0; ir < mcl; ir += 16) {
            float* const pA = packed_A + (ir / 16) * (Kc_pad * 16);
            const size_t rows = min_size(16, mcl - ir);
            float* const dstC = C + (mm + ir) * N + (nn + nr);
            const size_t cols = min_size(64, ncl - nr);
            const bool full = rows == 16 && cols == 64;
            svzero_za();
            // Later K panels read the next tile's C at its end; start pulling it now.
            if (!first_k) {
                const size_t nir = ir + 16 < mcl ? ir + 16 : 0, nnr = ir + 16 < mcl ? nr : nr + 64;
                if (nnr < ncl && nir + 16 <= mcl && nnr + 64 <= ncl)
                    prefetch_c_tile(C + (mm + nir) * N + (nn + nnr), N);
            }
            pf_issue(pfa);
            micro_kernel_1x4(pA, pB, kcl);

            if (full) {
                if (!first_k) add_c_into_za(dstC, N);
                store_za(dstC, N);
            } else {
                store_za_tile(scratch);
                for (size_t r = 0; r < rows; r++) {
                    const float* sr = scratch + r * 64;
                    float* cr = dstC + r * N;
                    for (size_t j = 0; j < cols; j += SVL) {
                        const svbool_t pv = svwhilelt_b32_u64(j, cols);
                        svfloat32_t v = svld1_f32(pv, sr + j);
                        if (!first_k) v = svadd_f32_m(pv, v, svld1_f32(pv, cr + j));
                        svst1_f32(pv, cr + j, v);
                    }
                }
            }
        }
    }
}

// One streaming region for the whole GEMM. The mm = 0 round is peeled out of
// the Mc loop so the fused pack needs no branch: it runs exactly once per
// (kk, nn), where both a fresh B block and a fresh A block are needed.
__arm_locally_streaming __arm_new("za")
__attribute__((noinline))
static bool run_streaming(const float* A, const float* B, float* C,
                          size_t M, size_t K, size_t N, size_t lda, size_t ldb, size_t ldc,
                          float* packed_A, float* packed_B,
                          float* stage, float* scratch,
                          size_t Kc, size_t Nc, size_t Mc) {
    if ((size_t)svcntsw() != 16) return false;
    const size_t Kc_pad = ((Kc + 63) / 64) * 64;

    PfCursor pfa = {0};

    for (size_t kk = 0; kk < K; kk += Kc) {
        const size_t kcl = min_size(Kc, K - kk);
        const bool first_k = (kk == 0);
        for (size_t nn = 0; nn < N; nn += Nc) {
            const size_t ncl = min_size(Nc, N - nn);

            const size_t mcl0 = min_size(Mc, M);

            fuse_pack(A, B, lda, ldb, mcl0, nn, ncl, kk, kcl, Kc, Kc_pad, packed_A, packed_B, stage);
            pf_next_a(&pfa, A, lda, M, K, N, Mc, Kc, Nc, kk, nn, 0, ((mcl0 + 15) / 16) * ((ncl + 63) / 64));
            compute_block(C, ldc, 0, mcl0, nn, ncl, kcl, first_k, Kc, Kc_pad, packed_A, packed_B, scratch, &pfa);

            for (size_t mm = Mc; mm < M; mm += Mc) {
                const size_t mcl = min_size(Mc, M - mm);
                pack_A_block(A, lda, mm, mcl, kk, kcl, Kc_pad, packed_A, stage);
                pf_next_a(&pfa, A, lda, M, K, N, Mc, Kc, Nc, kk, nn, mm, ((mcl + 15) / 16) * ((ncl + 63) / 64));
                compute_block(C, ldc, mm, mcl, nn, ncl, kcl, first_k, Kc, Kc_pad, packed_A, packed_B, scratch, &pfa);
            }
        }
    }
    return true;
}


/* The dispatcher verifies SME availability and a 16-lane streaming vector
   length. Full 64-column B panels avoid reading outside strided quadrants. */
int cob_sgemm_strassen_sme_leaf(int m, int n, int k, const float* a, int lda,
    const float* b, int ldb, float* c, int ldc)
{
    const size_t mc = min_size(64, ((size_t)m + 15) / 16 * 16);
    const size_t nc = min_size(1024, (size_t)n);
    const size_t kc = min_size(1024, (size_t)k);
    const size_t packed_a_count = mc * ((kc + 63) / 64 * 64);
    const size_t packed_b_count = nc * kc;
    const size_t tile_count = 16 * 64;
    float* packed_a = aligned_alloc(64,
        (packed_a_count + packed_b_count + 2 * tile_count) * sizeof(float));
    if (!packed_a) return 0;
    float* packed_b = packed_a + packed_a_count;
    float* stage = packed_b + packed_b_count;
    float* scratch = stage + tile_count;
    const int ok = run_streaming(a, b, c, m, k, n, lda, ldb, ldc,
        packed_a, packed_b, stage, scratch, kc, nc, mc);
    free(packed_a);
    return ok;
}
