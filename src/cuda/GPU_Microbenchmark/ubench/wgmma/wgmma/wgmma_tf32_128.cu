// =============================================================================
// wgmma_tf32_128.cu
//
// Functional-correctness microbenchmark for the Hopper WGMMA F32+=TF32*TF32
// atom at its native N=128 tile size — the shape/dtype combo CUTLASS example
// 48 (Hopper warp-specialized GEMM) uses for its TF32 kernel:
//
//   wgmma.mma_async.sync.aligned.m64n128k8.f32.tf32.tf32
//   wgmma.fence.sync.aligned
//   wgmma.commit_group.sync.aligned
//   wgmma.wait_group.sync.aligned
//
// wgmma_verify.cu hand-writes wgmma PTX with an explicit output-operand list,
// which is only practical up to N=16 (8 D-registers/thread). The m64n128k8
// atom has 64 D-registers/thread, so this benchmark instead goes through
// CUTLASS/CuTe's MMA_64x128x8_F32TF32TF32_SS_TN atom
// (cute/arch/mma_sm90_gmma.hpp) — the compiler emits the real instruction
// with all 64 accumulator registers wired up, and CuTe's partition_C()
// handles the fragment-to-global-(m,n) mapping so no hand-derived layout
// formula is needed (unlike wgmma_verify.cu's d_frag_pos()).
//
// GEMM shape: D[M×N] = A[M×K] * B^T[N×K]   (TN / K-major smem, single tile,
//   exactly the atom's native shape — one wgmma call, no K-loop)
//   M=64  N=128  K=8
//
// Smem fill is single-threaded (thread 0 only), same workaround as
// wgmma_no_tma.cu: GPGPU-Sim's sequential execution resolves intra-warp smem
// write ordering differently than real H100 concurrent execution, so a
// TiledCopy-based fill can produce spurious conflicts.
//
// Numeric note: GPGPU-Sim's tf32 functional model currently reads operands
// as full-precision float32 (no mantissa truncation on read — see
// quantize_to_type()/element_to_double() in instructions.cc). Real hardware
// truncates to 10 mantissa bits inside the tensor core. To stay correct
// against both, this test follows wgmma_verify.cu's convention: store raw
// (untruncated) float32 bits in the tf32 smem buffers, but compute the host
// reference from tf32-truncated inputs, and use the same loose tolerance
// (1e-3 absolute / 1% relative) that already absorbs the difference.
//
// Usage:
//   ./wgmma_tf32_128                        – real H100
//   PTX_SIM_USE_PTX_FILE=1 ./wgmma_tf32_128 – GPGPU-Sim functional sim
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>

#include <cute/tensor.hpp>

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/helper_cuda.hpp"

using namespace cute;

static constexpr int M = 64, N = 128, K = 8;

// ---------------------------------------------------------------------------
// Round float to the nearest tf32-representable value (10 mantissa bits).
// Matches wgmma_verify.cu's quant_tf32(); used only for the host reference.
// ---------------------------------------------------------------------------
static double quant_tf32(double f) {
  float fp = (float)f;
  uint32_t b; memcpy(&b, &fp, 4);
  b &= 0xFFFFE000u;
  memcpy(&fp, &b, 4);
  return (double)fp;
}

// ============================================================================
// Shared memory (K-major, no swizzle — GMMA::Layout_K_INTER_Atom)
// ============================================================================
template <class ASmemLayout, class BSmemLayout>
struct SharedStorage {
  alignas(128) cute::ArrayEngine<tfloat32_t, cosize_v<ASmemLayout>> A;
  alignas(128) cute::ArrayEngine<tfloat32_t, cosize_v<BSmemLayout>> B;
};

// ============================================================================
// Kernel: single-tile GEMM — exactly one wgmma.m64n128k8 call.
// ============================================================================
template <class TiledMma, class ASmemLayout, class BSmemLayout>
__global__ static
__launch_bounds__(decltype(size(TiledMma{}))::value)
void wgmma_tf32_128_kernel(tfloat32_t const* A_g, tfloat32_t const* B_g,
                           float* C_g, uint32_t* clk_g,
                           TiledMma mma, ASmemLayout sA_layout, BSmemLayout sB_layout)
{
  extern __shared__ char shared_memory[];
  using SS = SharedStorage<ASmemLayout, BSmemLayout>;
  SS& smem = *reinterpret_cast<SS*>(shared_memory);
  Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), sA_layout);  // (M,K)
  Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), sB_layout);  // (N,K)

  Tensor gA = make_tensor(make_gmem_ptr(A_g), make_shape(Int<M>{}, Int<K>{}),
                          make_stride(Int<K>{}, Int<1>{}));            // row-major (M,K)
  Tensor gB = make_tensor(make_gmem_ptr(B_g), make_shape(Int<N>{}, Int<K>{}),
                          make_stride(Int<K>{}, Int<1>{}));            // row-major (N,K) == B^T
  Tensor gC = make_tensor(make_gmem_ptr(C_g), make_shape(Int<M>{}, Int<N>{}),
                          make_stride(Int<N>{}, Int<1>{}));            // row-major (M,N)

  const int tid = threadIdx.x;
  if (tid == 0) { uint32_t c; asm volatile("mov.u32 %0,%%clock;\n" : "=r"(c)); clk_g[0] = c; }

  // Single-threaded smem fill (see file header for why).
  if (tid == 0) {
    for (int m = 0; m < M; ++m) for (int k = 0; k < K; ++k) sA(m, k) = gA(m, k);
    for (int n = 0; n < N; ++n) for (int k = 0; k < K; ++k) sB(n, k) = gB(n, k);
  }
  __syncthreads();
  if (tid == 0) { uint32_t c; asm volatile("mov.u32 %0,%%clock;\n" : "=r"(c)); clk_g[1] = c; }

  ThrMMA thr_mma = mma.get_slice(tid);
  Tensor tCsA = thr_mma.partition_A(sA);
  Tensor tCsB = thr_mma.partition_B(sB);
  Tensor tCgC = thr_mma.partition_C(gC);

  Tensor tCrA = thr_mma.make_fragment_A(tCsA);
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);
  Tensor tCrC = thr_mma.make_fragment_C(tCgC);
  clear(tCrC);

  warpgroup_fence_operand(tCrC);
  warpgroup_arrive();                    // wgmma.fence.sync.aligned
  cute::gemm(mma, tCrA, tCrB, tCrC);     // wgmma.mma_async.sync.aligned.m64n128k8.f32.tf32.tf32
  warpgroup_commit_batch();              // wgmma.commit_group.sync.aligned
  warpgroup_wait<0>();                   // wgmma.wait_group.sync.aligned 0
  warpgroup_fence_operand(tCrC);
  if (tid == 0) { uint32_t c; asm volatile("mov.u32 %0,%%clock;\n" : "=r"(c)); clk_g[2] = c; }

  axpby(1.0f, tCrC, 0.0f, tCgC);
}

// ============================================================================
// Host launcher
// ============================================================================
static void run_kernel(tfloat32_t const* A, tfloat32_t const* B, float* C, uint32_t* clk,
                       cudaStream_t stream = 0) {
  auto sA_layout = tile_to_shape(GMMA::Layout_K_INTER_Atom<tfloat32_t>{}, make_shape(Int<M>{}, Int<K>{}));
  auto sB_layout = tile_to_shape(GMMA::Layout_K_INTER_Atom<tfloat32_t>{}, make_shape(Int<N>{}, Int<K>{}));

  TiledMMA tiled_mma = make_tiled_mma(SM90_64x128x8_F32TF32TF32_SS_TN<>{});

  using SS = SharedStorage<decltype(sA_layout), decltype(sB_layout)>;
  int smem_size = int(sizeof(SS));

  dim3 block(size(tiled_mma));  // 128 threads = 1 warpgroup
  dim3 grid(1, 1, 1);

  auto kernel_ptr = &wgmma_tf32_128_kernel<decltype(tiled_mma), decltype(sA_layout), decltype(sB_layout)>;
  CUTE_CHECK_ERROR(cudaFuncSetAttribute(
    kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

  kernel_ptr<<<grid, block, smem_size, stream>>>(A, B, C, clk, tiled_mma, sA_layout, sB_layout);
  CUTE_CHECK_LAST();
}

// ============================================================================
// Test runner: fill A/B, launch, compare against a tf32-truncated CPU
// reference, print PASS/FAIL like wgmma_verify.cu.
// ============================================================================
static bool run_test(const char* name, const float* A_host, const float* B_host) {
  std::vector<tfloat32_t> A_d(M * K), B_d(N * K);
  std::vector<double>     A_ref(M * K), B_ref(N * K);
  for (int i = 0; i < M * K; ++i) {
    uint32_t bits; memcpy(&bits, &A_host[i], 4);
    A_d[i] = tfloat32_t::bitcast(bits);       // raw float32 bits (no truncation)
    A_ref[i] = quant_tf32(A_host[i]);          // tf32-truncated reference input
  }
  for (int i = 0; i < N * K; ++i) {
    uint32_t bits; memcpy(&bits, &B_host[i], 4);
    B_d[i] = tfloat32_t::bitcast(bits);
    B_ref[i] = quant_tf32(B_host[i]);
  }

  tfloat32_t *d_A, *d_B;
  float *d_C;
  uint32_t *d_clk;
  cudaMalloc(&d_A, M * K * sizeof(tfloat32_t));
  cudaMalloc(&d_B, N * K * sizeof(tfloat32_t));
  cudaMalloc(&d_C, M * N * sizeof(float));
  cudaMalloc(&d_clk, 3 * sizeof(uint32_t));
  cudaMemcpy(d_A, A_d.data(), M * K * sizeof(tfloat32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B_d.data(), N * K * sizeof(tfloat32_t), cudaMemcpyHostToDevice);

  run_kernel(d_A, d_B, d_C, d_clk);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("[%s] CUDA error: %s\n", name, cudaGetErrorString(err));
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_clk);
    return false;
  }

  std::vector<float> D_sim(M * N);
  uint32_t clk[3];
  cudaMemcpy(D_sim.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(clk, d_clk, 3 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_clk);

  // CPU reference: D[m][n] = sum_k A_ref[m][k] * B_ref[n][k]
  double max_err = 0; int max_m = 0, max_n = 0;
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      double s = 0.0;
      for (int k = 0; k < K; ++k) s += A_ref[m * K + k] * B_ref[n * K + k];
      double e = fabs((double)D_sim[m * N + n] - s);
      if (e > max_err) { max_err = e; max_m = m; max_n = n; }
    }
  }
  double ref_check = 0.0;
  for (int k = 0; k < K; ++k) ref_check += A_ref[max_m * K + k] * B_ref[max_n * K + k];
  double ref_mag = fabs(ref_check);
  double tol = (ref_mag < 1.0) ? 1e-3 : ref_mag * 1e-2;
  bool pass = (max_err <= tol);

  printf("[%-24s] %s  max_err=%.4e  fill=%u  wgmma=%u cycles\n",
         name, pass ? "PASS" : "FAIL", max_err, clk[1] - clk[0], clk[2] - clk[1]);
  if (!pass)
    printf("  Worst: D[%d][%d] sim=%.6f ref=%.6f\n",
           max_m, max_n, D_sim[max_m * N + max_n], (float)ref_check);
  return pass;
}

// ============================================================================
// main
// ============================================================================
int main() {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("wgmma_tf32_128  —  %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
  printf("Tests wgmma.mma_async.sync.aligned.m64n128k8.f32.tf32.tf32 (CUTLASS SM90_64x128x8_F32TF32TF32_SS_TN)\n");
  printf("plus wgmma.fence/commit_group/wait_group.sync.aligned\n");
  printf("================================================================\n");

  bool all_pass = true;

  // all-ones: D[m][n] = K = 8  (trivially verifiable)
  {
    static float A[M][K], B[N][K];
    for (int m = 0; m < M; ++m) for (int k = 0; k < K; ++k) A[m][k] = 1.f;
    for (int n = 0; n < N; ++n) for (int k = 0; k < K; ++k) B[n][k] = 1.f;
    all_pass &= run_test("tf32 m64n128k8 all-ones", (float*)A, (float*)B);
  }

  // sequential: small deterministic values vs. tf32-truncated double reference
  {
    static float A[M][K], B[N][K];
    for (int m = 0; m < M; ++m) for (int k = 0; k < K; ++k) A[m][k] = (m * K + k + 1) * 0.1f;
    for (int n = 0; n < N; ++n) for (int k = 0; k < K; ++k) B[n][k] = (k * N + n + 1) * 0.1f;
    all_pass &= run_test("tf32 m64n128k8 sequential", (float*)A, (float*)B);
  }

  printf("\n================================================================\n");
  printf("Overall: %s\n", all_pass ? "PASS" : "FAIL");
  return all_pass ? 0 : 1;
}
