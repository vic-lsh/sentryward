// Controls for strided access families in canonical address mode (tests/test_strided_native.py).
// Build (NGC nvcr.io/nvidia/pytorch:25.04-py3, nvcc 12.9): nvcc -O3 -std=c++17 --ptx -arch=sm_90a src/strided.cu -o ptx/strided.ptx
#include <stdint.h>
#define UNIT 4096  // bytes per table unit
#define VECS 1024  // uint4 copied per segment (16 KiB)

extern "C" {
// (1) A grid-stride copy whose source base is loaded from a table the kernel never writes:
// every thread's base is the same read-only word, so each copy instruction is one family.
__global__ void k_copy_ro(const uint64_t* tab, const char* state, uint4* staging) {
  const uint4* src = (const uint4*)(state + tab[blockIdx.y] * UNIT);
  uint4* dst = staging + (size_t)blockIdx.y * VECS;
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < VECS; i += (int64_t)gridDim.x * blockDim.x)
    dst[i] = src[i];
}

// (2) The same copy, but the kernel also writes a word of `tab` (outside the words it reads):
// the base load may observe a write, so it keeps its per-thread occurrence (no merge).
__global__ void k_copy_written_base(uint64_t* tab, const char* state, uint4* staging) {
  const uint4* src = (const uint4*)(state + tab[blockIdx.y] * UNIT);
  uint4* dst = staging + (size_t)blockIdx.y * VECS;
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < VECS; i += (int64_t)gridDim.x * blockDim.x)
    dst[i] = src[i];
  if (blockIdx.x == 0 && threadIdx.x == 0) tab[gridDim.y + blockIdx.y] = 0;
}

// (3) A per-thread data base: thread t writes 16 bytes at out + tab[t] * 16. The table is
// read-only but each thread loads its own word (different atoms, no merge); two threads whose
// words are equal write the same bytes, so a race must be reported.
__global__ void k_table_scatter(const uint64_t* tab, const uint4* in, uint4* out) {
  const int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  out[tab[t]] = in[t];
}

// (4) A copy whose stride breaks at one member: thread 37 writes element 38, which thread 38
// also writes. The family splits at the break, and the race is still reported.
__global__ void k_copy_broken_stride(const uint64_t* tab, const char* state, uint4* staging) {
  const uint4* src = (const uint4*)(state + tab[blockIdx.y] * UNIT);
  uint4* dst = staging + (size_t)blockIdx.y * VECS;
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = i == 37 ? i + 1 : i;
  dst[j] = src[i];
}
}

extern "C" {
// (4') The unbroken twin of (4): one family per instruction, race free.
__global__ void k_copy_once(const uint64_t* tab, const char* state, uint4* staging) {
  const uint4* src = (const uint4*)(state + tab[blockIdx.y] * UNIT);
  uint4* dst = staging + (size_t)blockIdx.y * VECS;
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  dst[i] = src[i];
}
}
