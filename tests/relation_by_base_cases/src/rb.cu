// k_gdn-shaped controls for the relation-by-base rule (docs/design/relation-by-base.md).
// Block s reads state at window src_tab[s] and writes state at window dst_tab[s]; a window is
// 256 * entry_units bytes, as Coral's recurrent.cu (u = (int32_t)(tab * entry_units), u * 256 + off).
// Variants (one PTX each, nvcc -O3 -arch=sm_90a --ptx -DVARIANT=n):
//   0 same map: thread t reads float t of its source window and writes float t of its destination
//   1 shifted write: thread t writes float t + 1 (a(m) + 4): a cross-thread pair of R
//   2 same table word, two threads write one float (t / 2): an ordinary race
//   3 per-thread table word: thread t reads src_tab[t] (the atom varies with the thread)
//   4 unequal widths: thread t reads 8 bytes at 8t, writes 4 bytes at 4t
//   5 per-head offset, as k_gdn's rec_off: block (s, h) reads and writes float 64 h + t of its windows
//   6 k_gdn's destination row: block s writes at window dst_tab[2 s] (next[s * positions], positions 2)
//   7 as 0, and thread 0 of block s also stores to dst_tab[s] (the table is written by the launch)
//   8 two read rows: block s writes float t at window src_tab[s] (both words are rows of `read`)
//   9 row atom varying from row 2: thread t reads window src_tab[(t + 2) & 63]; thread 62 reads row 0
#include <stdint.h>
#ifndef VARIANT
#define VARIANT 0
#endif
extern "C" __global__ void __launch_bounds__(128) k_rb(const uint64_t* src_tab, const uint64_t* dst_tab, char* state,
                                                       int64_t entry_units) {
  const int s = blockIdx.x, t = threadIdx.x;
#if VARIANT == 3
  const int32_t src_u = (int32_t)((int64_t)src_tab[t] * entry_units);
#elif VARIANT == 9
  const int32_t src_u = (int32_t)((int64_t)src_tab[(t + 2) & 63] * entry_units);
#else
  const int32_t src_u = (int32_t)((int64_t)src_tab[s] * entry_units);
#endif
#if VARIANT == 2 || VARIANT == 8
  const int32_t dst_u = (int32_t)((int64_t)src_tab[s] * entry_units);
#elif VARIANT == 6
  const int32_t dst_u = (int32_t)((int64_t)dst_tab[s * 2] * entry_units);
#else
  const int32_t dst_u = (int32_t)((int64_t)dst_tab[s] * entry_units);
#endif
#if VARIANT == 5
  const size_t rec_off = (size_t)blockIdx.y * 64 * 4;
#else
  const size_t rec_off = 0;
#endif
  const char* src = state + (size_t)src_u * 256 + rec_off;
  float* dst = (float*)(state + (size_t)dst_u * 256 + rec_off);
#if VARIANT == 8
  const float v = (float)t;
#elif VARIANT == 4
  const float2 v2 = ((const float2*)src)[t];
  const float v = v2.x + v2.y;
#else
  const float v = ((const float*)src)[t];
#endif
#if VARIANT == 1
  dst[t + 1] = 2.f * v;
#elif VARIANT == 2
  dst[t / 2] = 2.f * v;
#else
  dst[t] = 2.f * v;
#endif
#if VARIANT == 7
  if (t == 0) ((uint64_t*)dst_tab)[s] = (uint64_t)src_u;
#endif
}
