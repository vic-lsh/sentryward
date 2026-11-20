// Structural-mode role and producer-binding cases (tests/test_structural_native.py).
// One CTA of 256 threads; thread i owns element i of each f32 object.
#include <math.h>
extern "C" {

// C(P(a), a): p = P(a) stored, y = p * a. Term: cons(prod(a), a), avoids_transcendental on cons's input a.
__global__ void shared_pos(const float* a, float* p, float* y) {
  const int i = threadIdx.x;
  const float t = a[i] * a[i] + 1.f;
  p[i] = t;
  y[i] = t * a[i];
}
// P = sigmoid: a reaches y through a transcendental inside the producer. Must be refused (role).
__global__ void shared_neg_sigmoid(const float* a, float* p, float* y) {
  const int i = threadIdx.x;
  const float t = 1.f / (1.f + expf(-a[i]));
  p[i] = t;
  y[i] = t * a[i];
}

// An unstored producer q(a) used by two adds: y1 = q(a) + b, y2 = q(a) + c (one q value).
__global__ void unstored_pos(const float* a, const float* b, const float* c, float* y1, float* y2) {
  const int i = threadIdx.x;
  const float t = a[i] * a[i] + 1.f;
  y1[i] = t + b[i];
  y2[i] = t + c[i];
}
// Two different values with q's leaves at the two uses. Must be refused (producer_binding).
__global__ void unstored_neg_two(const float* a, const float* b, const float* c, float* y1, float* y2) {
  const int i = threadIdx.x;
  y1[i] = (a[i] * a[i] + 1.f) + b[i];
  y2[i] = (a[i] * a[i] + 2.f) + c[i];
}
// add(mul(a, b), c) with mul unstored, computed as (a - b) + c. Must be refused
// (unstored_primitive_producer): an unstored primitive's operation is not checked.
__global__ void unstored_neg_sub(const float* a, const float* b, const float* c, float* y) {
  const int i = threadIdx.x;
  y[i] = (a[i] - b[i]) + c[i];
}

// scale(a, b), multiplicand role on a: y = a * b.
__global__ void mult_pos(const float* a, const float* b, float* y) {
  const int i = threadIdx.x;
  y[i] = a[i] * b[i];
}
// y = a * b + a: one path from a bypasses the multiply. Must be refused (role, multiplicand).
__global__ void mult_neg_bypass(const float* a, const float* b, float* y) {
  const int i = threadIdx.x;
  y[i] = a[i] * b[i] + a[i];
}
}
