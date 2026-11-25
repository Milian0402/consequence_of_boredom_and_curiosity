#ifndef HK_PHASE_CLOCK_H
#define HK_PHASE_CLOCK_H
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { HK_FORMS, HK_CHILD, HK_OUTPUT, FUSED_BODY, FUSED_PACK_A, FUSED_PACK_B,
       FUSED_APPLY_SHORT, FUSED_APPLY_LONG, FUSED_MERGE, FUSED_EDGES,
       BASE_BODY, BASE_PACK_A, BASE_PACK_B, BASE_APPLY_SHORT, BASE_APPLY_LONG, HK_PHASES };
static struct { uint64_t ns[HK_PHASES], count[HK_PHASES]; } hk_phase;

static inline uint64_t hk_phase_clock(void)
{
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t)) abort();
    return (uint64_t)t.tv_sec * UINT64_C(1000000000) + (uint64_t)t.tv_nsec;
}
static inline void hk_phase_end(int id, uint64_t start)
{
    hk_phase.ns[id] += hk_phase_clock() - start;
    ++hk_phase.count[id];
}
#define HK_TIMED(id, expression) do { \
    uint64_t hk_begin = hk_phase_clock(); \
    expression; \
    hk_phase_end((id), hk_begin); \
} while (0)

void hk_profile_reset(void) { memset(&hk_phase, 0, sizeof(hk_phase)); }
void hk_profile_read(uint64_t* ns, uint64_t* counts)
{
    memcpy(ns, hk_phase.ns, sizeof(hk_phase.ns));
    memcpy(counts, hk_phase.count, sizeof(hk_phase.count));
}
#endif
