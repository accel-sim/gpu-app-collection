// =============================================================================
// wgmma_verify.cu  (v2)
//
// Functional-correctness and timing microbenchmark for wgmma.mma_async (SM90a).
//
// Type / shape combinations tested (all m64n16kK):
//   f16  k16  D=f32   swizzle modes 0, 1(128B), 2(64B), 3(32B)
//   bf16 k16  D=f32
//   tf32 k8   D=f32
//   e4m3 k32  D=f32
//   e5m2 k32  D=f32
//   s8   k32  D=s32
//   u8   k32  D=s32
//   b1   k256 D=s32   (GPGPU-Sim functional sim only;
//                      real H100 needs bit-packed smem, not 1 byte/bit)
//
// Smem layout (K-major, no swizzle unless SW!=0):
//   T = 16 / E  (elements per 128-bit column)
//   off = (stride%T + leading%8 * T)*E  +  (stride/T)*SBO  +  (leading/8)*LBO
//   SBO = 128 always.   LBO_A = M*E*8,   LBO_B = N*E*8.
//   All shapes satisfy  M*K*E = 2048  and  K*N*E = 512.
//
// Two test cases per type:
//   all-ones   – A=B=1  →  D[m][n] = K  (trivially verifiable)
//   sequential – small values compared against a host double-precision reference
//
// Usage:
//   ./wgmma_verify                       – real H100
//   PTX_SIM_MODE_FUNC=1 ./wgmma_verify  – GPGPU-Sim functional sim
// =============================================================================

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <climits>
#include <limits>
#include <vector>
#include <string>
#include <cstring>
#include <functional>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static constexpr int M       = 64;
static constexpr int N       = 16;
static constexpr int WGSIZE  = 128;
static constexpr int D_ELEMS = N / 2;    // 8 D-registers per thread

// For all standard types: M*K*E = 2048,  K*N*E = 512  (fixed by WGMMA design)
static constexpr int SMEM_A = 2048;
static constexpr int SMEM_B = 512;
static constexpr int SBO    = 128;   // stride byte offset (always 128 for K-major)

// b1 (sim convention): 1 byte per bit, K=256 "elements"
static constexpr int B1_K      = 256;
static constexpr int B1_SMEM_A = M * B1_K;   // 16384
static constexpr int B1_SMEM_B = B1_K * N;   // 4096

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint32_t smem_addr(const void* ptr) {
  uint32_t a;
  asm volatile(
    "{ .reg .u64 _p; cvta.to.shared.u64 _p, %1; cvt.u32.u64 %0, _p; }\n"
    : "=r"(a) : "l"(ptr));
  return a;
}

__device__ __forceinline__ uint64_t make_gmma_desc(
    uint32_t base, uint32_t lbo, uint32_t sbo, uint32_t sw = 0) {
  uint64_t d = 0;
  d |= (uint64_t)(base >> 4) & 0x3FFFu;
  d |= ((uint64_t)((lbo >> 4) & 0x3FFFu)) << 16;
  d |= ((uint64_t)((sbo >> 4) & 0x3FFFu)) << 32;
  d |= ((uint64_t)(sw & 0x3u)) << 62;
  return d;
}

// K-major smem byte offset (before swizzle).
// For A (M×K): leading=k, stride=m.  For B (K×N): leading=k, stride=n.
__host__ __device__ __forceinline__
int smem_off(int leading, int stride, int E, int LBO) {
  int T = 16 / E;
  return (stride % T + (leading % 8) * T) * E
       + (stride / T) * SBO
       + (leading / 8) * LBO;
}

// XOR-swizzle (self-inverse) applied to a byte offset.
__host__ __device__ __forceinline__
int apply_sw(int off, int sw) {
  if (!sw) return off;
  int s = (sw == 1) ? 128 : (sw == 2) ? 64 : 32;
  int b = off / s, i = off % s;
  return b * s + (i ^ ((b & 1) ? (s >> 1) : 0));
}

__host__ __device__ __forceinline__
int smem_off_sw(int leading, int stride, int E, int LBO, int sw) {
  return apply_sw(smem_off(leading, stride, E, LBO), sw);
}

// ---------------------------------------------------------------------------
// WGMMA protocol macros
// ---------------------------------------------------------------------------
#define WGMMA_FENCE  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory")
#define WGMMA_COMMIT asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory")
#define WGMMA_WAIT   asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory")
#define READ_CLK(c)  asm volatile("mov.u32 %0,%%clock;\n" : "=r"(c))

// ===========================================================================
// Kernels
// ===========================================================================

// ---------------------------------------------------------------------------
// F16  m64n16k16  D=f32   template on swizzle mode SW (0–3)
// ---------------------------------------------------------------------------
template<int SW>
__global__ void kernel_f16(const half* A_g, const half* B_g,
                            float* D_g, uint32_t* clk_g) {
  __shared__ __align__(128) char smA[SMEM_A], smB[SMEM_B];
  const int tid = threadIdx.x;
  constexpr int K = 16, E = 2, LBO_A = M*E*8, LBO_B = N*E*8;  // 1024, 256

  if (!tid) { uint32_t c; READ_CLK(c); clk_g[0] = c; }
  for (int i = tid; i < M*K; i += WGSIZE) {
    int m = i/K, k = i%K;
    *(half*)(smA + smem_off_sw(k, m, E, LBO_A, SW)) = A_g[i];
  }
  for (int i = tid; i < K*N; i += WGSIZE) {
    int k = i/N, n = i%N;
    *(half*)(smB + smem_off_sw(k, n, E, LBO_B, SW)) = B_g[i];
  }
  __syncthreads();
  if (!tid) { uint32_t c; READ_CLK(c); clk_g[1] = c; }

  uint64_t da = make_gmma_desc(smem_addr(smA), LBO_A, SBO, SW);
  uint64_t db = make_gmma_desc(smem_addr(smB), LBO_B, SBO, SW);
  float d0=0,d1=0,d2=0,d3=0,d4=0,d5=0,d6=0,d7=0;
  WGMMA_FENCE;
  asm volatile(
    "{ .reg .pred p; setp.ne.b32 p,1,0;\n"
    "wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 "
    "{%0,%1,%2,%3,%4,%5,%6,%7},%8,%9,p,1,1,0,0; }\n"
    : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3),"+f"(d4),"+f"(d5),"+f"(d6),"+f"(d7)
    : "l"(da),"l"(db) : "memory");
  WGMMA_COMMIT; WGMMA_WAIT; WGMMA_FENCE;
  if (!tid) { uint32_t c; READ_CLK(c); clk_g[2] = c; }

  int b = tid * D_ELEMS;
  D_g[b]=d0; D_g[b+1]=d1; D_g[b+2]=d2; D_g[b+3]=d3;
  D_g[b+4]=d4; D_g[b+5]=d5; D_g[b+6]=d6; D_g[b+7]=d7;
}

// ---------------------------------------------------------------------------
// BF16  m64n16k16  D=f32
// ---------------------------------------------------------------------------
__global__ void kernel_bf16(const __nv_bfloat16* A_g, const __nv_bfloat16* B_g,
                             float* D_g, uint32_t* clk_g) {
  __shared__ __align__(128) char smA[SMEM_A], smB[SMEM_B];
  const int tid = threadIdx.x;
  constexpr int K = 16, E = 2, LBO_A = M*E*8, LBO_B = N*E*8;

  if (!tid) { uint32_t c; READ_CLK(c); clk_g[0] = c; }
  for (int i = tid; i < M*K; i += WGSIZE) {
    int m = i/K, k = i%K;
    *(__nv_bfloat16*)(smA + smem_off(k, m, E, LBO_A)) = A_g[i];
  }
  for (int i = tid; i < K*N; i += WGSIZE) {
    int k = i/N, n = i%N;
    *(__nv_bfloat16*)(smB + smem_off(k, n, E, LBO_B)) = B_g[i];
  }
  __syncthreads();
  if (!tid) { uint32_t c; READ_CLK(c); clk_g[1] = c; }

  uint64_t da = make_gmma_desc(smem_addr(smA), LBO_A, SBO, 0);
  uint64_t db = make_gmma_desc(smem_addr(smB), LBO_B, SBO, 0);
  float d0=0,d1=0,d2=0,d3=0,d4=0,d5=0,d6=0,d7=0;
  WGMMA_FENCE;
  asm volatile(
    "{ .reg .pred p; setp.ne.b32 p,1,0;\n"
    "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "
    "{%0,%1,%2,%3,%4,%5,%6,%7},%8,%9,p,1,1,0,0; }\n"
    : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3),"+f"(d4),"+f"(d5),"+f"(d6),"+f"(d7)
    : "l"(da),"l"(db) : "memory");
  WGMMA_COMMIT; WGMMA_WAIT; WGMMA_FENCE;
  if (!tid) { uint32_t c; READ_CLK(c); clk_g[2] = c; }

  int b = tid * D_ELEMS;
  D_g[b]=d0; D_g[b+1]=d1; D_g[b+2]=d2; D_g[b+3]=d3;
  D_g[b+4]=d4; D_g[b+5]=d5; D_g[b+6]=d6; D_g[b+7]=d7;
}

// ---------------------------------------------------------------------------
// TF32  m64n16k8  D=f32   (no trans parameters)
// ---------------------------------------------------------------------------
__global__ void kernel_tf32(const float* A_g, const float* B_g,
                             float* D_g, uint32_t* clk_g) {
  __shared__ __align__(128) char smA[SMEM_A], smB[SMEM_B];
  const int tid = threadIdx.x;
  constexpr int K = 8, E = 4, LBO_A = M*E*8, LBO_B = N*E*8;  // 2048, 512

  if (!tid) { uint32_t c; READ_CLK(c); clk_g[0] = c; }
  for (int i = tid; i < M*K; i += WGSIZE) {
    int m = i/K, k = i%K;
    *(float*)(smA + smem_off(k, m, E, LBO_A)) = A_g[i];
  }
  for (int i = tid; i < K*N; i += WGSIZE) {
    int k = i/N, n = i%N;
    *(float*)(smB + smem_off(k, n, E, LBO_B)) = B_g[i];
  }
  __syncthreads();
  if (!tid) { uint32_t c; READ_CLK(c); clk_g[1] = c; }

  uint64_t da = make_gmma_desc(smem_addr(smA), LBO_A, SBO, 0);
  uint64_t db = make_gmma_desc(smem_addr(smB), LBO_B, SBO, 0);
  float d0=0,d1=0,d2=0,d3=0,d4=0,d5=0,d6=0,d7=0;
  WGMMA_FENCE;
  asm volatile(
    "{ .reg .pred p; setp.ne.b32 p,1,0;\n"
    "wgmma.mma_async.sync.aligned.m64n16k8.f32.tf32.tf32 "
    "{%0,%1,%2,%3,%4,%5,%6,%7},%8,%9,p,1,1; }\n"
    : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3),"+f"(d4),"+f"(d5),"+f"(d6),"+f"(d7)
    : "l"(da),"l"(db) : "memory");
  WGMMA_COMMIT; WGMMA_WAIT; WGMMA_FENCE;
  if (!tid) { uint32_t c; READ_CLK(c); clk_g[2] = c; }

  int b = tid * D_ELEMS;
  D_g[b]=d0; D_g[b+1]=d1; D_g[b+2]=d2; D_g[b+3]=d3;
  D_g[b+4]=d4; D_g[b+5]=d5; D_g[b+6]=d6; D_g[b+7]=d7;
}

// ---------------------------------------------------------------------------
// FP8 E4M3  m64n16k32  D=f32   (no trans)
// A_g / B_g carry raw fp8 bytes (uint8_t).
// ---------------------------------------------------------------------------
__global__ void kernel_e4m3(const uint8_t* A_g, const uint8_t* B_g,
                             float* D_g, uint32_t* clk_g) {
  __shared__ __align__(128) char smA[SMEM_A], smB[SMEM_B];
  const int tid = threadIdx.x;
  constexpr int K = 32, E = 1, LBO_A = M*E*8, LBO_B = N*E*8;  // 512, 128

  if (!tid) { uint32_t c; READ_CLK(c); clk_g[0] = c; }
  for (int i = tid; i < M*K; i += WGSIZE) {
    int m = i/K, k = i%K;
    *(uint8_t*)(smA + smem_off(k, m, E, LBO_A)) = A_g[i];
  }
  for (int i = tid; i < K*N; i += WGSIZE) {
    int k = i/N, n = i%N;
    *(uint8_t*)(smB + smem_off(k, n, E, LBO_B)) = B_g[i];
  }
  __syncthreads();
  if (!tid) { uint32_t c; READ_CLK(c); clk_g[1] = c; }

  uint64_t da = make_gmma_desc(smem_addr(smA), LBO_A, SBO, 0);
  uint64_t db = make_gmma_desc(smem_addr(smB), LBO_B, SBO, 0);
  float d0=0,d1=0,d2=0,d3=0,d4=0,d5=0,d6=0,d7=0;
  WGMMA_FENCE;
  asm volatile(
    "{ .reg .pred p; setp.ne.b32 p,1,0;\n"
    "wgmma.mma_async.sync.aligned.m64n16k32.f32.e4m3.e4m3 "
    "{%0,%1,%2,%3,%4,%5,%6,%7},%8,%9,p,1,1; }\n"
    : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3),"+f"(d4),"+f"(d5),"+f"(d6),"+f"(d7)
    : "l"(da),"l"(db) : "memory");
  WGMMA_COMMIT; WGMMA_WAIT; WGMMA_FENCE;
  if (!tid) { uint32_t c; READ_CLK(c); clk_g[2] = c; }

  int b = tid * D_ELEMS;
  D_g[b]=d0; D_g[b+1]=d1; D_g[b+2]=d2; D_g[b+3]=d3;
  D_g[b+4]=d4; D_g[b+5]=d5; D_g[b+6]=d6; D_g[b+7]=d7;
}

// ---------------------------------------------------------------------------
// FP8 E5M2  m64n16k32  D=f32   (no trans)
// ---------------------------------------------------------------------------
__global__ void kernel_e5m2(const uint8_t* A_g, const uint8_t* B_g,
                             float* D_g, uint32_t* clk_g) {
  __shared__ __align__(128) char smA[SMEM_A], smB[SMEM_B];
  const int tid = threadIdx.x;
  constexpr int K = 32, E = 1, LBO_A = M*E*8, LBO_B = N*E*8;

  if (!tid) { uint32_t c; READ_CLK(c); clk_g[0] = c; }
  for (int i = tid; i < M*K; i += WGSIZE) {
    int m = i/K, k = i%K;
    *(uint8_t*)(smA + smem_off(k, m, E, LBO_A)) = A_g[i];
  }
  for (int i = tid; i < K*N; i += WGSIZE) {
    int k = i/N, n = i%N;
    *(uint8_t*)(smB + smem_off(k, n, E, LBO_B)) = B_g[i];
  }
  __syncthreads();
  if (!tid) { uint32_t c; READ_CLK(c); clk_g[1] = c; }

  uint64_t da = make_gmma_desc(smem_addr(smA), LBO_A, SBO, 0);
  uint64_t db = make_gmma_desc(smem_addr(smB), LBO_B, SBO, 0);
  float d0=0,d1=0,d2=0,d3=0,d4=0,d5=0,d6=0,d7=0;
  WGMMA_FENCE;
  asm volatile(
    "{ .reg .pred p; setp.ne.b32 p,1,0;\n"
    "wgmma.mma_async.sync.aligned.m64n16k32.f32.e5m2.e5m2 "
    "{%0,%1,%2,%3,%4,%5,%6,%7},%8,%9,p,1,1; }\n"
    : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3),"+f"(d4),"+f"(d5),"+f"(d6),"+f"(d7)
    : "l"(da),"l"(db) : "memory");
  WGMMA_COMMIT; WGMMA_WAIT; WGMMA_FENCE;
  if (!tid) { uint32_t c; READ_CLK(c); clk_g[2] = c; }

  int b = tid * D_ELEMS;
  D_g[b]=d0; D_g[b+1]=d1; D_g[b+2]=d2; D_g[b+3]=d3;
  D_g[b+4]=d4; D_g[b+5]=d5; D_g[b+6]=d6; D_g[b+7]=d7;
}

// ---------------------------------------------------------------------------
// S8  m64n16k32  D=s32   (no scale, no trans)
// ---------------------------------------------------------------------------
__global__ void kernel_s8(const int8_t* A_g, const int8_t* B_g,
                           int32_t* D_g, uint32_t* clk_g) {
  __shared__ __align__(128) char smA[SMEM_A], smB[SMEM_B];
  const int tid = threadIdx.x;
  constexpr int K = 32, E = 1, LBO_A = M*E*8, LBO_B = N*E*8;

  if (!tid) { uint32_t c; READ_CLK(c); clk_g[0] = c; }
  for (int i = tid; i < M*K; i += WGSIZE) {
    int m = i/K, k = i%K;
    *(int8_t*)(smA + smem_off(k, m, E, LBO_A)) = A_g[i];
  }
  for (int i = tid; i < K*N; i += WGSIZE) {
    int k = i/N, n = i%N;
    *(int8_t*)(smB + smem_off(k, n, E, LBO_B)) = B_g[i];
  }
  __syncthreads();
  if (!tid) { uint32_t c; READ_CLK(c); clk_g[1] = c; }

  uint64_t da = make_gmma_desc(smem_addr(smA), LBO_A, SBO, 0);
  uint64_t db = make_gmma_desc(smem_addr(smB), LBO_B, SBO, 0);
  int32_t d0=0,d1=0,d2=0,d3=0,d4=0,d5=0,d6=0,d7=0;
  WGMMA_FENCE;
  asm volatile(
    "{ .reg .pred p; setp.ne.b32 p,1,0;\n"
    "wgmma.mma_async.sync.aligned.m64n16k32.s32.s8.s8 "
    "{%0,%1,%2,%3,%4,%5,%6,%7},%8,%9,p; }\n"
    : "+r"(d0),"+r"(d1),"+r"(d2),"+r"(d3),"+r"(d4),"+r"(d5),"+r"(d6),"+r"(d7)
    : "l"(da),"l"(db) : "memory");
  WGMMA_COMMIT; WGMMA_WAIT; WGMMA_FENCE;
  if (!tid) { uint32_t c; READ_CLK(c); clk_g[2] = c; }

  int b = tid * D_ELEMS;
  D_g[b]=d0; D_g[b+1]=d1; D_g[b+2]=d2; D_g[b+3]=d3;
  D_g[b+4]=d4; D_g[b+5]=d5; D_g[b+6]=d6; D_g[b+7]=d7;
}

// ---------------------------------------------------------------------------
// U8  m64n16k32  D=s32
// ---------------------------------------------------------------------------
__global__ void kernel_u8(const uint8_t* A_g, const uint8_t* B_g,
                           int32_t* D_g, uint32_t* clk_g) {
  __shared__ __align__(128) char smA[SMEM_A], smB[SMEM_B];
  const int tid = threadIdx.x;
  constexpr int K = 32, E = 1, LBO_A = M*E*8, LBO_B = N*E*8;

  if (!tid) { uint32_t c; READ_CLK(c); clk_g[0] = c; }
  for (int i = tid; i < M*K; i += WGSIZE) {
    int m = i/K, k = i%K;
    *(uint8_t*)(smA + smem_off(k, m, E, LBO_A)) = A_g[i];
  }
  for (int i = tid; i < K*N; i += WGSIZE) {
    int k = i/N, n = i%N;
    *(uint8_t*)(smB + smem_off(k, n, E, LBO_B)) = B_g[i];
  }
  __syncthreads();
  if (!tid) { uint32_t c; READ_CLK(c); clk_g[1] = c; }

  uint64_t da = make_gmma_desc(smem_addr(smA), LBO_A, SBO, 0);
  uint64_t db = make_gmma_desc(smem_addr(smB), LBO_B, SBO, 0);
  int32_t d0=0,d1=0,d2=0,d3=0,d4=0,d5=0,d6=0,d7=0;
  WGMMA_FENCE;
  asm volatile(
    "{ .reg .pred p; setp.ne.b32 p,1,0;\n"
    "wgmma.mma_async.sync.aligned.m64n16k32.s32.u8.u8 "
    "{%0,%1,%2,%3,%4,%5,%6,%7},%8,%9,p; }\n"
    : "+r"(d0),"+r"(d1),"+r"(d2),"+r"(d3),"+r"(d4),"+r"(d5),"+r"(d6),"+r"(d7)
    : "l"(da),"l"(db) : "memory");
  WGMMA_COMMIT; WGMMA_WAIT; WGMMA_FENCE;
  if (!tid) { uint32_t c; READ_CLK(c); clk_g[2] = c; }

  int b = tid * D_ELEMS;
  D_g[b]=d0; D_g[b+1]=d1; D_g[b+2]=d2; D_g[b+3]=d3;
  D_g[b+4]=d4; D_g[b+5]=d5; D_g[b+6]=d6; D_g[b+7]=d7;
}

// ---------------------------------------------------------------------------
// B1  m64n16k256  D=s32
//
// WARNING — GPGPU-Sim functional sim only.
// The sim reads each "bit" as a separate byte (1 byte per bit element, using
// only the LSB).  Real H100 hardware expects bits packed 8-per-byte (256 bits
// = 32 bytes per row of A).  This kernel uses 1 byte/bit, so the smem
// buffers are 256× larger than on real hardware.
// ---------------------------------------------------------------------------
__global__ void kernel_b1(const uint8_t* A_g, const uint8_t* B_g,
                           int32_t* D_g, uint32_t* clk_g) {
  __shared__ __align__(128) uint8_t smA[B1_SMEM_A];   // 16384 bytes
  __shared__ __align__(128) uint8_t smB[B1_SMEM_B];   // 4096 bytes
  const int tid = threadIdx.x;
  constexpr int K = B1_K, E = 1, LBO_A = M*E*8, LBO_B = N*E*8;  // 512, 128

  if (!tid) { uint32_t c; READ_CLK(c); clk_g[0] = c; }
  for (int i = tid; i < M*K; i += WGSIZE) {
    int m = i/K, k = i%K;
    smA[smem_off(k, m, E, LBO_A)] = A_g[i];
  }
  for (int i = tid; i < K*N; i += WGSIZE) {
    int k = i/N, n = i%N;
    smB[smem_off(k, n, E, LBO_B)] = B_g[i];
  }
  __syncthreads();
  if (!tid) { uint32_t c; READ_CLK(c); clk_g[1] = c; }

  uint64_t da = make_gmma_desc(smem_addr(smA), LBO_A, SBO, 0);
  uint64_t db = make_gmma_desc(smem_addr(smB), LBO_B, SBO, 0);
  int32_t d0=0,d1=0,d2=0,d3=0,d4=0,d5=0,d6=0,d7=0;
  WGMMA_FENCE;
  asm volatile(
    "{ .reg .pred p; setp.ne.b32 p,1,0;\n"
    "wgmma.mma_async.sync.aligned.m64n16k256.s32.b1.b1.and.popc "
    "{%0,%1,%2,%3,%4,%5,%6,%7},%8,%9,p; }\n"
    : "+r"(d0),"+r"(d1),"+r"(d2),"+r"(d3),"+r"(d4),"+r"(d5),"+r"(d6),"+r"(d7)
    : "l"(da),"l"(db) : "memory");
  WGMMA_COMMIT; WGMMA_WAIT; WGMMA_FENCE;
  if (!tid) { uint32_t c; READ_CLK(c); clk_g[2] = c; }

  int b = tid * D_ELEMS;
  D_g[b]=d0; D_g[b+1]=d1; D_g[b+2]=d2; D_g[b+3]=d3;
  D_g[b+4]=d4; D_g[b+5]=d5; D_g[b+6]=d6; D_g[b+7]=d7;
}

// ===========================================================================
// Host: fragment mapping and matrix reassembly
// ===========================================================================

static void d_frag_pos(int T, int e, int* row, int* col) {
  int warp = T/32, lane = T%32, s = e/4, k = e%4;
  *row = (lane/4)*2 + k/2 + warp*16;
  *col = (lane%4)*2 + k%2 + s*8;
}

static void reassemble_D_f32(const float* flat, float D[M][N]) {
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) D[m][n] = 0.f;
  for (int T = 0; T < WGSIZE; ++T)
    for (int e = 0; e < D_ELEMS; ++e) {
      int row, col; d_frag_pos(T, e, &row, &col);
      if (row < M && col < N) D[row][col] = flat[T*D_ELEMS + e];
    }
}

static void reassemble_D_s32(const int32_t* flat, int32_t D[M][N]) {
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) D[m][n] = 0;
  for (int T = 0; T < WGSIZE; ++T)
    for (int e = 0; e < D_ELEMS; ++e) {
      int row, col; d_frag_pos(T, e, &row, &col);
      if (row < M && col < N) D[row][col] = flat[T*D_ELEMS + e];
    }
}

// ===========================================================================
// Host: result checking and printing
// ===========================================================================

static bool check_f32(const char* name,
                      float D_sim[M][N], const double* D_ref_flat,
                      uint32_t c0, uint32_t c1, uint32_t c2) {
  double max_err = 0; int max_m = 0, max_n = 0;
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      double e = fabs((double)D_sim[m][n] - D_ref_flat[m*N+n]);
      if (e > max_err) { max_err = e; max_m = m; max_n = n; }
    }
  double ref_mag = fabs(D_ref_flat[max_m*N+max_n]);
  double tol = (ref_mag < 1.0) ? 1e-3 : ref_mag * 1e-2;
  bool pass = (max_err <= tol);
  printf("[%-40s] %s  max_err=%.4e  fill=%u  wgmma=%u cycles\n",
         name, pass ? "PASS" : "FAIL", max_err, c1-c0, c2-c1);
  if (!pass)
    printf("  Worst: D[%d][%d] sim=%.6f ref=%.6f\n",
           max_m, max_n, D_sim[max_m][max_n], (float)D_ref_flat[max_m*N+max_n]);
  return pass;
}

static bool check_s32(const char* name,
                      int32_t D_sim[M][N], const int64_t* D_ref_flat,
                      uint32_t c0, uint32_t c1, uint32_t c2) {
  int64_t max_err = 0; int max_m = 0, max_n = 0;
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      int64_t e = llabs((int64_t)D_sim[m][n] - D_ref_flat[m*N+n]);
      if (e > max_err) { max_err = e; max_m = m; max_n = n; }
    }
  bool pass = (max_err == 0);
  printf("[%-40s] %s  max_err=%lld  fill=%u  wgmma=%u cycles\n",
         name, pass ? "PASS" : "FAIL", (long long)max_err, c1-c0, c2-c1);
  if (!pass)
    printf("  Worst: D[%d][%d] sim=%d ref=%lld\n",
           max_m, max_n, D_sim[max_m][max_n], (long long)D_ref_flat[max_m*N+max_n]);
  return pass;
}

// ===========================================================================
// Generic test runner helpers
// ===========================================================================

// Run a f32-accumulator test.
//   K          — K dimension
//   bytes_a/b  — device memory sizes for A and B
//   h_A / h_B  — host pointers to device-formatted input data
//   A_dbl/B_dbl — double arrays (M*K and K*N) for the reference matmul;
//                 must already reflect type quantization (e.g. bf16-rounded)
//   launch     — lambda: (d_A, d_B, d_D, d_clk) → void
static bool run_f32(const char* name, int K,
                    size_t bytes_a, size_t bytes_b,
                    const void* h_A, const void* h_B,
                    const double* A_dbl, const double* B_dbl,
                    std::function<void(const void*, const void*, float*, uint32_t*)> launch) {
  void *d_A, *d_B;
  float *d_D;
  uint32_t *d_clk;
  cudaMalloc(&d_A,   bytes_a);
  cudaMalloc(&d_B,   bytes_b);
  cudaMalloc(&d_D,   WGSIZE * D_ELEMS * sizeof(float));
  cudaMalloc(&d_clk, 3 * sizeof(uint32_t));
  cudaMemcpy(d_A, h_A, bytes_a, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, bytes_b, cudaMemcpyHostToDevice);

  launch(d_A, d_B, d_D, d_clk);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("[%s] CUDA error: %s\n", name, cudaGetErrorString(err));
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_D); cudaFree(d_clk);
    return false;
  }

  std::vector<float>    flat(WGSIZE * D_ELEMS);
  uint32_t clk[3];
  cudaMemcpy(flat.data(), d_D,   WGSIZE*D_ELEMS*sizeof(float),  cudaMemcpyDeviceToHost);
  cudaMemcpy(clk,         d_clk, 3*sizeof(uint32_t),             cudaMemcpyDeviceToHost);
  cudaFree(d_A); cudaFree(d_B); cudaFree(d_D); cudaFree(d_clk);

  static float   D_sim[M][N];
  static double  D_ref[M*N];
  reassemble_D_f32(flat.data(), D_sim);

  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      double s = 0.0;
      for (int k = 0; k < K; ++k) s += A_dbl[m*K+k] * B_dbl[k*N+n];
      D_ref[m*N+n] = s;
    }

  return check_f32(name, D_sim, D_ref, clk[0], clk[1], clk[2]);
}

// Run a s32-accumulator test.
//   use_and — if true, reference uses AND-accumulate (b1); else multiply-accumulate
static bool run_s32(const char* name, int K,
                    size_t bytes_a, size_t bytes_b,
                    const void* h_A, const void* h_B,
                    const int64_t* A_i64, const int64_t* B_i64,
                    bool use_and,
                    std::function<void(const void*, const void*, int32_t*, uint32_t*)> launch) {
  void *d_A, *d_B;
  int32_t *d_D;
  uint32_t *d_clk;
  cudaMalloc(&d_A,   bytes_a);
  cudaMalloc(&d_B,   bytes_b);
  cudaMalloc(&d_D,   WGSIZE * D_ELEMS * sizeof(int32_t));
  cudaMalloc(&d_clk, 3 * sizeof(uint32_t));
  cudaMemcpy(d_A, h_A, bytes_a, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, bytes_b, cudaMemcpyHostToDevice);

  launch(d_A, d_B, d_D, d_clk);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("[%s] CUDA error: %s\n", name, cudaGetErrorString(err));
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_D); cudaFree(d_clk);
    return false;
  }

  std::vector<int32_t> flat(WGSIZE * D_ELEMS);
  uint32_t clk[3];
  cudaMemcpy(flat.data(), d_D,   WGSIZE*D_ELEMS*sizeof(int32_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(clk,         d_clk, 3*sizeof(uint32_t),              cudaMemcpyDeviceToHost);
  cudaFree(d_A); cudaFree(d_B); cudaFree(d_D); cudaFree(d_clk);

  static int32_t D_sim[M][N];
  static int64_t D_ref[M*N];
  reassemble_D_s32(flat.data(), D_sim);

  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      int64_t s = 0;
      for (int k = 0; k < K; ++k) {
        int64_t a = A_i64[m*K+k], b = B_i64[k*N+n];
        s += use_and ? ((a & b) & 1) : (a * b);
      }
      D_ref[m*N+n] = s;
    }

  return check_s32(name, D_sim, D_ref, clk[0], clk[1], clk[2]);
}

// ===========================================================================
// Host-side type quantization helpers
// ===========================================================================

// Round float to the nearest bf16-representable value.
static double quant_bf16(double f) {
  return (double)__bfloat162float(__float2bfloat16((float)f));
}

// Round float to the nearest tf32-representable value (10 mantissa bits).
static double quant_tf32(double f) {
  float fp = (float)f;
  uint32_t b; memcpy(&b, &fp, 4);
  b &= 0xFFFFE000u;            // keep sign + 8 exp + top 10 mantissa bits
  memcpy(&fp, &b, 4);
  return (double)fp;
}

// Round float to nearest fp8 e4m3 value.
static double quant_e4m3(double f) {
  return (double)(float)(__nv_fp8_e4m3)((float)f);
}

// Round float to nearest fp8 e5m2 value.
static double quant_e5m2(double f) {
  return (double)(float)(__nv_fp8_e5m2)((float)f);
}

// Convert float to raw fp8 e4m3 byte.
static uint8_t float_to_e4m3(float f) {
  __nv_fp8_e4m3 v(f);
  uint8_t b; memcpy(&b, &v, 1);
  return b;
}

// Convert float to raw fp8 e5m2 byte.
static uint8_t float_to_e5m2(float f) {
  __nv_fp8_e5m2 v(f);
  uint8_t b; memcpy(&b, &v, 1);
  return b;
}

// ===========================================================================
// Per-type test functions
// ===========================================================================

// ---- F16 ----
static bool test_f16(const char* name, int sw,
                     const float* A_host, const float* B_host) {
  constexpr int K = 16;
  std::vector<half>   A_h(M*K), B_h(K*N);
  std::vector<double> A_d(M*K), B_d(K*N);
  for (int i = 0; i < M*K; ++i) {
    A_h[i] = __float2half(A_host[i]);
    A_d[i] = (double)__half2float(A_h[i]);
  }
  for (int i = 0; i < K*N; ++i) {
    B_h[i] = __float2half(B_host[i]);
    B_d[i] = (double)__half2float(B_h[i]);
  }
  auto launch = [sw](const void* dA, const void* dB, float* dD, uint32_t* dC) {
    switch (sw) {
      case 0: kernel_f16<0><<<1,WGSIZE>>>((const half*)dA,(const half*)dB,dD,dC); break;
      case 1: kernel_f16<1><<<1,WGSIZE>>>((const half*)dA,(const half*)dB,dD,dC); break;
      case 2: kernel_f16<2><<<1,WGSIZE>>>((const half*)dA,(const half*)dB,dD,dC); break;
      case 3: kernel_f16<3><<<1,WGSIZE>>>((const half*)dA,(const half*)dB,dD,dC); break;
    }
  };
  return run_f32(name, K, M*K*sizeof(half), K*N*sizeof(half),
                 A_h.data(), B_h.data(), A_d.data(), B_d.data(), launch);
}

// ---- BF16 ----
static bool test_bf16(const char* name,
                      const float* A_host, const float* B_host) {
  constexpr int K = 16;
  std::vector<__nv_bfloat16> A_b(M*K), B_b(K*N);
  std::vector<double>        A_d(M*K), B_d(K*N);
  for (int i = 0; i < M*K; ++i) {
    A_b[i] = __float2bfloat16(A_host[i]);
    A_d[i] = quant_bf16(A_host[i]);
  }
  for (int i = 0; i < K*N; ++i) {
    B_b[i] = __float2bfloat16(B_host[i]);
    B_d[i] = quant_bf16(B_host[i]);
  }
  auto launch = [](const void* dA, const void* dB, float* dD, uint32_t* dC) {
    kernel_bf16<<<1,WGSIZE>>>((const __nv_bfloat16*)dA,(const __nv_bfloat16*)dB,dD,dC);
  };
  return run_f32(name, K, M*K*sizeof(__nv_bfloat16), K*N*sizeof(__nv_bfloat16),
                 A_b.data(), B_b.data(), A_d.data(), B_d.data(), launch);
}

// ---- TF32 ----
static bool test_tf32(const char* name,
                      const float* A_host, const float* B_host) {
  constexpr int K = 8;
  std::vector<float>  A_f(M*K), B_f(K*N);
  std::vector<double> A_d(M*K), B_d(K*N);
  for (int i = 0; i < M*K; ++i) {
    A_d[i] = quant_tf32(A_host[i]);
    A_f[i] = (float)A_d[i];
  }
  for (int i = 0; i < K*N; ++i) {
    B_d[i] = quant_tf32(B_host[i]);
    B_f[i] = (float)B_d[i];
  }
  auto launch = [](const void* dA, const void* dB, float* dD, uint32_t* dC) {
    kernel_tf32<<<1,WGSIZE>>>((const float*)dA,(const float*)dB,dD,dC);
  };
  return run_f32(name, K, M*K*sizeof(float), K*N*sizeof(float),
                 A_f.data(), B_f.data(), A_d.data(), B_d.data(), launch);
}

// ---- FP8 E4M3 ----
static bool test_e4m3(const char* name,
                      const float* A_host, const float* B_host) {
  constexpr int K = 32;
  std::vector<uint8_t> A_b(M*K), B_b(K*N);
  std::vector<double>  A_d(M*K), B_d(K*N);
  for (int i = 0; i < M*K; ++i) {
    A_b[i] = float_to_e4m3(A_host[i]);
    A_d[i] = quant_e4m3(A_host[i]);
  }
  for (int i = 0; i < K*N; ++i) {
    B_b[i] = float_to_e4m3(B_host[i]);
    B_d[i] = quant_e4m3(B_host[i]);
  }
  auto launch = [](const void* dA, const void* dB, float* dD, uint32_t* dC) {
    kernel_e4m3<<<1,WGSIZE>>>((const uint8_t*)dA,(const uint8_t*)dB,dD,dC);
  };
  return run_f32(name, K, M*K, K*N,
                 A_b.data(), B_b.data(), A_d.data(), B_d.data(), launch);
}

// ---- FP8 E5M2 ----
static bool test_e5m2(const char* name,
                      const float* A_host, const float* B_host) {
  constexpr int K = 32;
  std::vector<uint8_t> A_b(M*K), B_b(K*N);
  std::vector<double>  A_d(M*K), B_d(K*N);
  for (int i = 0; i < M*K; ++i) {
    A_b[i] = float_to_e5m2(A_host[i]);
    A_d[i] = quant_e5m2(A_host[i]);
  }
  for (int i = 0; i < K*N; ++i) {
    B_b[i] = float_to_e5m2(B_host[i]);
    B_d[i] = quant_e5m2(B_host[i]);
  }
  auto launch = [](const void* dA, const void* dB, float* dD, uint32_t* dC) {
    kernel_e5m2<<<1,WGSIZE>>>((const uint8_t*)dA,(const uint8_t*)dB,dD,dC);
  };
  return run_f32(name, K, M*K, K*N,
                 A_b.data(), B_b.data(), A_d.data(), B_d.data(), launch);
}

// ---- S8 ----
static bool test_s8(const char* name,
                    const int8_t* A_host, const int8_t* B_host) {
  constexpr int K = 32;
  std::vector<int64_t> A_i(M*K), B_i(K*N);
  for (int i = 0; i < M*K; ++i) A_i[i] = A_host[i];
  for (int i = 0; i < K*N; ++i) B_i[i] = B_host[i];
  auto launch = [](const void* dA, const void* dB, int32_t* dD, uint32_t* dC) {
    kernel_s8<<<1,WGSIZE>>>((const int8_t*)dA,(const int8_t*)dB,dD,dC);
  };
  return run_s32(name, K, M*K, K*N,
                 A_host, B_host, A_i.data(), B_i.data(), false, launch);
}

// ---- U8 ----
static bool test_u8(const char* name,
                    const uint8_t* A_host, const uint8_t* B_host) {
  constexpr int K = 32;
  std::vector<int64_t> A_i(M*K), B_i(K*N);
  for (int i = 0; i < M*K; ++i) A_i[i] = A_host[i];
  for (int i = 0; i < K*N; ++i) B_i[i] = B_host[i];
  auto launch = [](const void* dA, const void* dB, int32_t* dD, uint32_t* dC) {
    kernel_u8<<<1,WGSIZE>>>((const uint8_t*)dA,(const uint8_t*)dB,dD,dC);
  };
  return run_s32(name, K, M*K, K*N,
                 A_host, B_host, A_i.data(), B_i.data(), false, launch);
}

// ---- B1 (sim-only) ----
static bool test_b1(const char* name,
                    const uint8_t* A_host, const uint8_t* B_host) {
  constexpr int K = B1_K;
  std::vector<int64_t> A_i(M*K), B_i(K*N);
  for (int i = 0; i < M*K; ++i) A_i[i] = A_host[i] & 1;
  for (int i = 0; i < K*N; ++i) B_i[i] = B_host[i] & 1;
  auto launch = [](const void* dA, const void* dB, int32_t* dD, uint32_t* dC) {
    kernel_b1<<<1,WGSIZE>>>((const uint8_t*)dA,(const uint8_t*)dB,dD,dC);
  };
  return run_s32(name, K, (size_t)M*K, (size_t)K*N,
                 A_host, B_host, A_i.data(), B_i.data(), true, launch);
}

// ===========================================================================
// main
// ===========================================================================
int main() {
  {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("wgmma_verify v2  —  %s (SM %d.%d)\n",
           prop.name, prop.major, prop.minor);
  }
  printf("================================================================\n");

  bool all_pass = true;

  // =========================================================================
  // F16  m64n16k16  D=f32  (swizzle modes 0–3)
  // =========================================================================
  printf("\n--- F16 m64n16k16  D=f32 ---\n");
  {
    static float A[M][16], B[16][N];
    // all-ones
    for (int m = 0; m < M; ++m) for (int k = 0; k < 16; ++k) A[m][k] = 1.f;
    for (int k = 0; k < 16; ++k) for (int n = 0; n < N; ++n) B[k][n] = 1.f;
    for (int sw = 0; sw < 4; ++sw) {
      char name[64]; snprintf(name, sizeof(name), "f16 all-ones  swizzle=%d", sw);
      all_pass &= test_f16(name, sw, (float*)A, (float*)B);
    }
    // sequential
    for (int m = 0; m < M; ++m)
      for (int k = 0; k < 16; ++k) A[m][k] = (m*16 + k + 1) * 0.1f;
    for (int k = 0; k < 16; ++k)
      for (int n = 0; n < N; ++n) B[k][n] = (k*N + n + 1) * 0.1f;
    for (int sw = 0; sw < 4; ++sw) {
      char name[64]; snprintf(name, sizeof(name), "f16 sequential swizzle=%d", sw);
      all_pass &= test_f16(name, sw, (float*)A, (float*)B);
    }
  }

  // =========================================================================
  // BF16  m64n16k16  D=f32
  // =========================================================================
  printf("\n--- BF16 m64n16k16  D=f32 ---\n");
  {
    static float A[M][16], B[16][N];
    for (int m = 0; m < M; ++m) for (int k = 0; k < 16; ++k) A[m][k] = 1.f;
    for (int k = 0; k < 16; ++k) for (int n = 0; n < N; ++n) B[k][n] = 1.f;
    all_pass &= test_bf16("bf16 all-ones", (float*)A, (float*)B);

    for (int m = 0; m < M; ++m)
      for (int k = 0; k < 16; ++k) A[m][k] = (m*16 + k + 1) * 0.1f;
    for (int k = 0; k < 16; ++k)
      for (int n = 0; n < N; ++n) B[k][n] = (k*N + n + 1) * 0.1f;
    all_pass &= test_bf16("bf16 sequential", (float*)A, (float*)B);
  }

  // =========================================================================
  // TF32  m64n16k8  D=f32
  // =========================================================================
  printf("\n--- TF32 m64n16k8  D=f32 ---\n");
  {
    static float A[M][8], B[8][N];
    for (int m = 0; m < M; ++m) for (int k = 0; k < 8; ++k) A[m][k] = 1.f;
    for (int k = 0; k < 8; ++k)  for (int n = 0; n < N; ++n) B[k][n] = 1.f;
    all_pass &= test_tf32("tf32 all-ones", (float*)A, (float*)B);

    for (int m = 0; m < M; ++m)
      for (int k = 0; k < 8; ++k) A[m][k] = (m*8 + k + 1) * 0.1f;
    for (int k = 0; k < 8; ++k)
      for (int n = 0; n < N; ++n) B[k][n] = (k*N + n + 1) * 0.1f;
    all_pass &= test_tf32("tf32 sequential", (float*)A, (float*)B);
  }

  // =========================================================================
  // FP8 E4M3  m64n16k32  D=f32
  // Values {1,2,3,4} are exactly representable in e4m3; no quantisation error.
  // =========================================================================
  printf("\n--- FP8 E4M3 m64n16k32  D=f32 ---\n");
  {
    static float A[M][32], B[32][N];
    for (int m = 0; m < M; ++m) for (int k = 0; k < 32; ++k) A[m][k] = 1.f;
    for (int k = 0; k < 32; ++k) for (int n = 0; n < N; ++n) B[k][n] = 1.f;
    all_pass &= test_e4m3("e4m3 all-ones", (float*)A, (float*)B);

    for (int m = 0; m < M; ++m)
      for (int k = 0; k < 32; ++k) A[m][k] = (float)(1 + (m+k) % 4);
    for (int k = 0; k < 32; ++k)
      for (int n = 0; n < N; ++n) B[k][n] = (float)(1 + (k+n) % 4);
    all_pass &= test_e4m3("e4m3 sequential", (float*)A, (float*)B);
  }

  // =========================================================================
  // FP8 E5M2  m64n16k32  D=f32
  // =========================================================================
  printf("\n--- FP8 E5M2 m64n16k32  D=f32 ---\n");
  {
    static float A[M][32], B[32][N];
    for (int m = 0; m < M; ++m) for (int k = 0; k < 32; ++k) A[m][k] = 1.f;
    for (int k = 0; k < 32; ++k) for (int n = 0; n < N; ++n) B[k][n] = 1.f;
    all_pass &= test_e5m2("e5m2 all-ones", (float*)A, (float*)B);

    for (int m = 0; m < M; ++m)
      for (int k = 0; k < 32; ++k) A[m][k] = (float)(1 + (m+k) % 4);
    for (int k = 0; k < 32; ++k)
      for (int n = 0; n < N; ++n) B[k][n] = (float)(1 + (k+n) % 4);
    all_pass &= test_e5m2("e5m2 sequential", (float*)A, (float*)B);
  }

  // =========================================================================
  // S8  m64n16k32  D=s32
  // =========================================================================
  printf("\n--- S8 m64n16k32  D=s32 ---\n");
  {
    static int8_t A[M][32], B[32][N];
    for (int m = 0; m < M; ++m) for (int k = 0; k < 32; ++k) A[m][k] = 1;
    for (int k = 0; k < 32; ++k) for (int n = 0; n < N; ++n) B[k][n] = 1;
    all_pass &= test_s8("s8 all-ones", (int8_t*)A, (int8_t*)B);

    // Small signed values to avoid overflow: max sum = 32 * 5 * 5 = 800 < INT32_MAX
    for (int m = 0; m < M; ++m)
      for (int k = 0; k < 32; ++k) A[m][k] = (int8_t)(1 + (m+k) % 5);
    for (int k = 0; k < 32; ++k)
      for (int n = 0; n < N; ++n) B[k][n] = (int8_t)(1 + (k+n) % 5);
    all_pass &= test_s8("s8 sequential", (int8_t*)A, (int8_t*)B);
  }

  // =========================================================================
  // U8  m64n16k32  D=s32
  // =========================================================================
  printf("\n--- U8 m64n16k32  D=s32 ---\n");
  {
    static uint8_t A[M][32], B[32][N];
    for (int m = 0; m < M; ++m) for (int k = 0; k < 32; ++k) A[m][k] = 1;
    for (int k = 0; k < 32; ++k) for (int n = 0; n < N; ++n) B[k][n] = 1;
    all_pass &= test_u8("u8 all-ones", (uint8_t*)A, (uint8_t*)B);

    for (int m = 0; m < M; ++m)
      for (int k = 0; k < 32; ++k) A[m][k] = (uint8_t)(1 + (m+k) % 5);
    for (int k = 0; k < 32; ++k)
      for (int n = 0; n < N; ++n) B[k][n] = (uint8_t)(1 + (k+n) % 5);
    all_pass &= test_u8("u8 sequential", (uint8_t*)A, (uint8_t*)B);
  }

  // =========================================================================
  // B1  m64n16k256  D=s32   (GPGPU-Sim functional sim only)
  // D[m][n] = popcount of positions k where A[m][k]=1 AND B[k][n]=1.
  // =========================================================================
  printf("\n--- B1 m64n16k256  D=s32  (sim-only layout) ---\n");
  {
    static uint8_t A[M][B1_K], B[B1_K][N];
    // all-ones: every bit = 1 → D[m][n] = K = 256
    for (int m = 0; m < M; ++m) for (int k = 0; k < B1_K; ++k) A[m][k] = 1;
    for (int k = 0; k < B1_K; ++k) for (int n = 0; n < N; ++n) B[k][n] = 1;
    all_pass &= test_b1("b1 all-ones", (uint8_t*)A, (uint8_t*)B);

    // checkerboard: A[m][k] = k%2,  B[k][n] = k%2
    // D[m][n] = #{k : k%2=1 AND k%2=1} = K/2 = 128
    for (int m = 0; m < M; ++m) for (int k = 0; k < B1_K; ++k) A[m][k] = k % 2;
    for (int k = 0; k < B1_K; ++k) for (int n = 0; n < N; ++n) B[k][n] = k % 2;
    all_pass &= test_b1("b1 checkerboard", (uint8_t*)A, (uint8_t*)B);
  }

  printf("\n================================================================\n");
  printf("Overall: %s\n", all_pass ? "PASS" : "FAIL");
  return all_pass ? 0 : 1;
}
