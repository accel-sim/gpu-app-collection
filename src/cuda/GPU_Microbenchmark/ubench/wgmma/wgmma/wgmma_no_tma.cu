// =============================================================================
// wgmma_no_tma.cu
//
// Simplified Hopper WGMMA GEMM copied from CuTe tutorial wgmma_sm90.cu with:
//   - TMA removed   (smem loads use cp.async, same as SM80 kernels)
//   - mbarrier removed (sync uses __syncthreads + cp_async_wait)
//   - warp specialization removed (single role, all 128 threads do load+mma)
//   - F32 accumulator (like CUTLASS example 48)
//   - N=16 tile (smallest WGMMA shape; fast to simulate in GPGPU-Sim)
//
// The key test: cute::gemm() issues wgmma.mma_async PTX through the compiler,
// not via hand-written asm().  This verifies GPGPU-Sim handles compiled WGMMA.
//
// GEMM shape: C[M×N] = A[M×K] * B^T[N×K]   (NT layout, column-major)
//   MMA tile : m64n16k16  (accumulated over bK=64 → 4 wgmma calls per tile)
//   Global   : M=64, N=16, K=64
//
// Build (SM90a required):
//   nvcc -gencode=arch=compute_90a,code=sm_90a \
//        -gencode=arch=compute_90a,code=compute_90a \
//        -I${CUTLASS_ROOT}/include -I${CUTLASS_ROOT}/tools/util/include \
//        -std=c++17 -O2 -o wgmma_no_tma wgmma_no_tma.cu -lcudart
//
// Run on real H100:
//   ./wgmma_no_tma
// Run under GPGPU-Sim (functional):
//   PTX_SIM_MODE_FUNC=1 ./wgmma_no_tma
// =============================================================================

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cassert>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#include "cutlass/cluster_launch.hpp"
#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"

using namespace cute;

// ============================================================================
// Shared memory layout struct (same as wgmma_sm90.cu)
// ============================================================================

template <class ElementA, class ElementB,
          class SmemLayoutA, class SmemLayoutB>
struct SharedStorage {
  alignas(128) cute::ArrayEngine<ElementA, cosize_v<SmemLayoutA>> A;
  alignas(128) cute::ArrayEngine<ElementB, cosize_v<SmemLayoutB>> B;
};

// ============================================================================
// Kernel  (adapted from gemm_device in wgmma_sm90.cu)
//
// Changes vs. the original:
//   1. No pipeline prefetch — single stage (bP=1).
//   2. Inner loop: copy → cp_async_fence → cp_async_wait<0> → __syncthreads
//      → wgmma → __syncthreads.
//   3. warpgroup_fence_operand / warpgroup_arrive / warpgroup_commit_batch /
//      warpgroup_wait stay exactly as in wgmma_sm90.cu.
// ============================================================================

template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class TiledCopyA,
          class TB, class BStride, class BSmemLayout, class TiledCopyB,
          class TC, class CStride, class TiledMma>
__global__ static
__launch_bounds__(decltype(size(TiledMma{}))::value)
void gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
                 TA const* A, AStride dA, ASmemLayout /*sA_layout*/, TiledCopyA copy_a,
                 TB const* B, BStride dB, BSmemLayout /*sB_layout*/, TiledCopyB copy_b,
                 TC      * C, CStride dC,                             TiledMma   mma)
{
  CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});
  CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});
  CUTE_STATIC_ASSERT_V(size(copy_a) == size(mma));
  CUTE_STATIC_ASSERT_V(size(copy_b) == size(mma));
  static_assert(is_static<ASmemLayout>::value);
  static_assert(is_static<BSmemLayout>::value);

  // ---- Full and tiled global tensors ----------------------------------------
  Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA); // (M,K)
  Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB); // (N,K)
  Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC); // (M,N)

  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});   // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});   // (BLK_N,BLK_K,k)
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});   // (BLK_M,BLK_N)

  // ---- Shared memory --------------------------------------------------------
  extern __shared__ char shared_memory[];
  using SS = SharedStorage<TA, TB, ASmemLayout, BSmemLayout>;
  SS& smem = *reinterpret_cast<SS*>(shared_memory);
  Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), ASmemLayout{}); // (BLK_M,BLK_K,1)
  Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), BSmemLayout{}); // (BLK_N,BLK_K,1)

  // ---- Copy partitioning (gmem → smem) -------------------------------------
  ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
  Tensor tAgA = thr_copy_a.partition_S(gA);                              // (CPY,CPY_M,CPY_K,k)
  Tensor tAsA = thr_copy_a.partition_D(as_position_independent_swizzle_tensor(sA));

  ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
  Tensor tBgB = thr_copy_b.partition_S(gB);                              // (CPY,CPY_N,CPY_K,k)
  Tensor tBsB = thr_copy_b.partition_D(as_position_independent_swizzle_tensor(sB));

  // ---- MMA partitioning (smem → regs → accum) ------------------------------
  ThrMMA thr_mma = mma.get_slice(threadIdx.x);
  Tensor tCsA = thr_mma.partition_A(sA);                                 // (MMA,MMA_M,MMA_K,1)
  Tensor tCsB = thr_mma.partition_B(sB);                                 // (MMA,MMA_N,MMA_K,1)
  Tensor tCgC = thr_mma.partition_C(gC);                                 // (MMA,MMA_M,MMA_N)

  Tensor tCrA  = thr_mma.make_fragment_A(tCsA);
  Tensor tCrB  = thr_mma.make_fragment_B(tCsB);
  Tensor tCrC  = thr_mma.make_fragment_C(tCgC);
  clear(tCrC);

  // ---- K-tile loop (no pipelining — single smem stage) ---------------------
  auto K_TILE_MAX = size<3>(tAgA);

  CUTE_NO_UNROLL
  for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile) {
    // Load this K-tile into smem stage 0 (cp.async, no TMA)
    copy(copy_a, tAgA(_,_,_,k_tile), tAsA(_,_,_,0));
    copy(copy_b, tBgB(_,_,_,k_tile), tBsB(_,_,_,0));
    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();

    // WGMMA: accumulate D += A * B  (generated as wgmma.mma_async PTX by compiler)
    warpgroup_fence_operand(tCrC);
    warpgroup_arrive();
    cute::gemm(mma, tCrA(_,_,_,0), tCrB(_,_,_,0), tCrC);  // issues 4 wgmma PTX calls
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    warpgroup_fence_operand(tCrC);

    __syncthreads();
  }

  // ---- Epilogue: store accumulators to C -----------------------------------
  axpby(TC(1), tCrC, TC(0), tCgC);
}

// ============================================================================
// Host launcher  (NT layout: A col-major M×K, B col-major N×K, C col-major M×N)
// Adapted from gemm_nt() in wgmma_sm90.cu, with MMA atom changed to
//   SM90_64x16x16_F32F16F16_SS (N=16 tile, F32 accumulator).
// ============================================================================

void gemm_nt(int m, int n, int k,
             float alpha,
             cute::half_t const* A, int ldA,   // col-major M×K
             cute::half_t const* B, int ldB,   // col-major N×K  (B^T is K×N)
             float beta,
             float      * C, int ldC,          // col-major M×N
             cudaStream_t stream = 0)
{
  using TA = cute::half_t;
  using TB = cute::half_t;
  using TC = float;

  auto M = int(m), N = int(n), K = int(k);
  auto prob_shape = make_shape(M, N, K);                      // (M, N, K)

  // NT strides: A col-major (stride-1 in M), B col-major (stride-1 in N)
  auto dA = make_stride(Int<1>{}, ldA);                       // (dM=1, dK=ldA)
  auto dB = make_stride(Int<1>{}, ldB);                       // (dN=1, dK=ldB)
  auto dC = make_stride(Int<1>{}, ldC);                       // (dM=1, dN=ldC)

  // CTA tile sizes
  auto bM = Int<64>{};
  auto bN = Int<16>{};
  auto bK = Int<64>{};   // bK/16 = 4 wgmma calls per K-tile
  auto bP = Int< 1>{};   // single pipeline stage (no TMA double-buffer)
  auto cta_tiler = make_shape(bM, bN, bK);

  // Smem layouts (MN-major = col-major).
  // Atom widths for half_t: SW128→64, SW64→32, SW32→16.
  // bM=64 fits SW128 (64), bN=16 fits SW32 (16).
  auto sA = tile_to_shape(GMMA::Layout_MN_SW128_Atom<TA>{}, make_shape(bM, bK, bP));
  auto sB = tile_to_shape(GMMA::Layout_MN_SW32_Atom<TB>{},  make_shape(bN, bK, bP));

  // Copy atoms: SM80 cp.async (NOT TMA) — col-major thr layout
  TiledCopy copyA = make_tiled_copy(
    Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, TA>{},
    Layout<Shape<_16,_8>>{},   // 16×8 threads, m-major
    Layout<Shape< _8,_1>>{});  // 8×1 values = 16 bytes per thread
  TiledCopy copyB = make_tiled_copy(
    Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, TB>{},
    Layout<Shape<_16,_8>>{},
    Layout<Shape< _8,_1>>{});

  // WGMMA tiled MMA: m64n16k16 F32←F16×F16  (MN-major smem for both A and B)
  TiledMMA tiled_mma = make_tiled_mma(
    SM90_64x16x16_F32F16F16_SS<GMMA::Major::MN, GMMA::Major::MN>{});

  // Shared memory size
  int smem_size = int(sizeof(SharedStorage<TA, TB, decltype(sA), decltype(sB)>));

  dim3 dimBlock(size(tiled_mma));          // 128 threads (1 warpgroup)
  dim3 dimCluster(1, 1, 1);
  dim3 dimGrid(ceil_div(m, int(bM)), ceil_div(n, int(bN)));

  auto kernel_ptr = reinterpret_cast<void const*>(
    &gemm_device<decltype(prob_shape), decltype(cta_tiler),
                 TA, decltype(dA), decltype(sA), decltype(copyA),
                 TB, decltype(dB), decltype(sB), decltype(copyB),
                 TC, decltype(dC), decltype(tiled_mma)>);

  CUTE_CHECK_ERROR(cudaFuncSetAttribute(
    kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

  cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smem_size, stream};
  cutlass::Status status = cutlass::launch_kernel_on_cluster(
    params, kernel_ptr,
    prob_shape, cta_tiler,
    A, dA, sA, copyA,
    B, dB, sB, copyB,
    C, dC, tiled_mma);

  CUTE_CHECK_LAST();
  if (status != cutlass::Status::kSuccess)
    fprintf(stderr, "launch_kernel_on_cluster failed\n");
}

// ============================================================================
// main: allocate, run, verify
// ============================================================================

int main() {
  cudaDeviceProp props;
  int dev;
  CUTE_CHECK_ERROR(cudaGetDevice(&dev));
  CUTE_CHECK_ERROR(cudaGetDeviceProperties(&props, dev));
  printf("wgmma_no_tma  —  %s  (SM %d.%d)\n", props.name, props.major, props.minor);

  if (props.major != 9) {
    printf("This test requires SM90a (Hopper).  Skipping.\n");
    return 0;
  }

  // Problem size: keep small for GPGPU-Sim
  constexpr int M = 64, N = 16, K = 64;
  printf("GEMM: C[%dx%d] = A[%dx%d] * B^T[%dx%d]  (NT col-major, FP16→FP32)\n",
         M, N, M, K, N, K);
  printf("Tile: m64n16k64 (4 wgmma.m64n16k16 calls via cute::gemm per K-tile)\n");
  printf("------------------------------------------------------------\n");

  // Host allocation (col-major)
  thrust::host_vector<cute::half_t> h_A(M * K), h_B(N * K);
  thrust::host_vector<float>        h_C(M * N, 0.f);

  // Fill with small integer values (avoids FP16 rounding in reference)
  for (int j = 0; j < M*K; ++j) h_A[j] = cute::half_t(float((j % 5) - 2));  // {-2,-1,0,1,2}
  for (int j = 0; j < N*K; ++j) h_B[j] = cute::half_t(float((j % 3) - 1));  // {-1,0,1}

  // Device
  thrust::device_vector<cute::half_t> d_A = h_A;
  thrust::device_vector<cute::half_t> d_B = h_B;
  thrust::device_vector<float>        d_C = h_C;

  // Run WGMMA kernel
  gemm_nt(M, N, K, 1.0f,
          d_A.data().get(), M,   // A: ldA = M  (col-major M×K)
          d_B.data().get(), N,   // B: ldB = N  (col-major N×K)
          0.0f,
          d_C.data().get(), M);  // C: ldC = M  (col-major M×N)

  CUTE_CHECK_ERROR(cudaDeviceSynchronize());
  h_C = d_C;

  // ---- CPU reference -------------------------------------------------------
  // C_ref[m][n] = sum_k  A[m][k] * B[n][k]
  // Col-major indexing: A[m + k*M], B[n + k*N], C[m + n*M]
  std::vector<double> ref(M * N, 0.0);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n)
      for (int kk = 0; kk < K; ++kk)
        ref[m + n*M] += double(float(h_A[m + kk*M])) * double(float(h_B[n + kk*N]));

  // ---- Check ---------------------------------------------------------------
  double max_err = 0.0;
  int    max_m = 0, max_n = 0;
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      double e = std::fabs(double(h_C[m + n*M]) - ref[m + n*M]);
      if (e > max_err) { max_err = e; max_m = m; max_n = n; }
    }

  double ref_mag = std::fabs(ref[max_m + max_n*M]);
  double tol     = (ref_mag < 1.0) ? 1e-3 : ref_mag * 1e-2;
  bool   pass    = (max_err <= tol);

  printf("Result: %s  max_err=%.4e  (at C[%d][%d]  ref=%.4f  sim=%.4f)\n",
         pass ? "PASS" : "FAIL", max_err, max_m, max_n,
         float(ref[max_m + max_n*M]), h_C[max_m + max_n*M]);

  return pass ? 0 : 1;
}
