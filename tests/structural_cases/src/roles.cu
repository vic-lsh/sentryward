// Structural-mode role and producer-binding cases (tests/test_structural_native.py).
// One CTA of 256 threads; thread i owns element i of each f32 object.
#include <math.h>
#include <cuda_fp16.h>
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
// C(P(a), Q(P(a))): p = P(a) and q = Q(p) stored, y = p * q; avoids_transcendental on C's P
// argument. Q = 2p: no transcendental on any path from p.
__global__ void chain_pos(const float* a, float* p, float* q, float* y) {
  const int i = threadIdx.x;
  const float t = a[i] * a[i] + 1.f;
  p[i] = t;
  const float u = 2.f * t;
  q[i] = u;
  y[i] = t * u;
}
// Q = exp(p): p reaches y through Q's exp (behind q's cut). Must be refused (role).
__global__ void chain_neg_exp(const float* a, float* p, float* q, float* y) {
  const int i = threadIdx.x;
  const float t = a[i] * a[i] + 1.f;
  p[i] = t;
  const float u = expf(t);
  q[i] = u;
  y[i] = t * u;
}

// The unstored producer bound as t at one use and f32(f16(t)) at the other: two values with
// one (node, result, instance). Must be refused (producer_binding).
__global__ void unstored_neg_conv(const float* a, const float* b, const float* c, float* y1, float* y2) {
  const int i = threadIdx.x;
  const float t = a[i] * a[i] + 1.f;
  y1[i] = t + b[i];
  y2[i] = __half2float(__float2half(t)) + c[i];
}
// scale(P(a), b) with a multiplicand role on the body argument: refused (role_boundary), the
// leaf-cone walk would cross into the producer.
__global__ void body_mult(const float* a, const float* b, float* p, float* y) {
  const int i = threadIdx.x;
  const float t = a[i] * a[i] + 1.f;
  p[i] = t;
  y[i] = t * b[i];
}
}
