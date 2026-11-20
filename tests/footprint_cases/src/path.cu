// Stage A cases (docs/design/loop-summaries.md, relational path conditions). One warp, every lane
// the same access (the footprint is the check; races are not);
// hdr words are loaded data with no premise unless the input states one. out has 8 u32 elements.
#include <stdint.h>
extern "C" {
// A2 eq: w == 5 forced taken bounds w exactly: out[w] contained.
__global__ void k_eq(const uint32_t* hdr, uint32_t* out) {
  const uint32_t w = hdr[0];
  if (w != 5) return;
  out[w] = 1;
}
// Control: w == 8 reaches out[8], one past the end.
__global__ void k_eq_past(const uint32_t* hdr, uint32_t* out) {
  const uint32_t w = hdr[0];
  if (w != 8) return;
  out[w] = 1;
}
// A2 two unknowns: a < b && b <= 8 bounds a < 8 only relationally.
__global__ void k_two(const uint32_t* hdr, uint32_t* out) {
  const uint32_t a = hdr[0], b = hdr[1];
  if (b > 8) return;
  if (a >= b) return;
  out[a] = 1;
}
// Control: a <= b && b <= 8 reaches out[8].
__global__ void k_two_past(const uint32_t* hdr, uint32_t* out) {
  const uint32_t a = hdr[0], b = hdr[1];
  if (b > 8) return;
  if (a > b) return;
  out[a] = 1;
}
// A2 or.pred (k_sample's shape): (s >= 8 || hdr[1] != 0) forced not taken bounds s < 8.
__global__ void k_or(const uint32_t* hdr, uint32_t* out) {
  const uint32_t s = hdr[0];
  const bool last = hdr[1] == 0;
  if (s >= 8 || !last) return;
  out[s] = 1;
}
// Control: (s >= 8 && hdr[1] != 0) forced not taken bounds neither.
__global__ void k_and(const uint32_t* hdr, uint32_t* out) {
  const uint32_t s = hdr[0];
  const bool last = hdr[1] == 0;
  if (s >= 8 && !last) return;
  out[s] = 1;
}
// Mask: n <= 15 (premise) gives n & -8 <= 8, inside 16 elements (interval rule for -2^m masks).
__global__ void k_mask(const uint32_t* hdr, uint32_t* out16) {
  const uint32_t n = hdr[0];
  out16[n & 0xFFFFFFF8u] = 1;
}
// Control: with n <= 63, n & ~6 reaches 57, past the 16 elements.
__global__ void k_mask_odd(const uint32_t* hdr, uint32_t* out16) {
  const uint32_t n = hdr[0];
  out16[n & 0xFFFFFFF9u] = 1;
}
// A2 or.pred with a known operand (k_sample's split-1 shape): z is a launch argument (0), so
// (s >= 8 || z != 0) is s >= 8; forced not taken bounds s < 8.
__global__ void k_or_known(const uint32_t* hdr, uint32_t* out, uint32_t z) {
  const uint32_t s = hdr[0];
  if (s >= 8 || z != 0) return;
  out[s] = 1;
}
// Control: s >= 9 reaches out[8].
__global__ void k_or_known_past(const uint32_t* hdr, uint32_t* out, uint32_t z) {
  const uint32_t s = hdr[0];
  if (s >= 9 || z != 0) return;
  out[s] = 1;
}
}
