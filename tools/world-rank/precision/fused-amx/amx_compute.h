#ifndef COB_AMX_SIX_COMPUTE_H
#define COB_AMX_SIX_COMPUTE_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int cob_amx_six_compiled(void);
/* Cumulative per-thread tile, depth-block, and dot counts (indices 0, 1, 2).
   Ordinary and unsupported builds return zero. */
uint64_t cob_amx_six_trace(int index);

/* Call only after AMX-BF16 and the calling thread's OS tile permission are verified.
   All partial input tiles are zero-padded. A's 3 planes use (M tile, K tile,
   row, k). B's 3 planes use (N tile, K tile, k/2, column, k%2).
   The tile shape is M=16, N=16, K=32. C contains only live row-major outputs.
   Returns 0 on completion or 1 for unsupported builds/invalid inputs. */
int cob_amx_six_tiles(int m, int n, int k, const uint16_t *packed_a,
                      const uint16_t *packed_b, float *c, int ldc);
#ifdef __cplusplus
}
#endif
#endif
