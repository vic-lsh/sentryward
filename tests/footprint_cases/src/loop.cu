// Loop-summary cases (docs/design/loop-summaries.md). Built with nvcc 12.9 -O3 -std=c++17 --ptx
// -arch=sm_121a (tests/footprint_cases/src/build.sh). One thread each; n = hdr[0] is loaded data.
#include <stdint.h>
extern "C" {
// Test 1: every iteration i < n, n bounded by a premise: contained.
__global__ void k_copy_n(const uint32_t* hdr, const float* in, float* out) {
  const uint32_t n = hdr[0];
  for (uint32_t i = 0; i < n; i++) out[i] = in[i];
}
// Test 2 (control): i <= n reads and writes one element past the extent at i = n.
__global__ void k_copy_le(const uint32_t* hdr, const float* in, float* out) {
  const uint32_t n = hdr[0];
  for (uint32_t i = 0; i <= n; i++) out[i] = in[i];
}
// Test 4 (control): the read index is a carried value loaded from data (havocked): not contained.
__global__ void k_chase(const uint32_t* hdr, const uint32_t* idx, const float* in, float* out) {
  const uint32_t n = hdr[0];
  uint32_t p = 0;
  for (uint32_t i = 0; i < n; i++) {
    out[i] = in[p];
    p = idx[i];
  }
}
// Test 1b: k_conv's shape, a row range [lo, hi) from a table of ends.
__global__ void k_rows(const uint32_t* ends, const float* in, float* out) {
  const uint32_t s = blockIdx.x;
  const int lo = s == 0 ? 0 : (int)ends[s - 1], n = (int)ends[s] - lo;
  for (int i = 0; i < n; i++) out[lo + i] = in[lo + i];
}
}
