/***************************************************************************************************
 * GMMA Latency Microbenchmark - Common Definitions
 *
 * This header contains shared kernel templates and helper macros used by all GMMA latency tests.
 *
 **************************************************************************************************/

#ifndef LAT_GMMA_COMMON_H
#define LAT_GMMA_COMMON_H

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstdint>
#include <vector>
#include <string>

// Output CSV file; defined in lat_gmma.cu, NULL disables file output.
extern FILE* g_lat_gmma_outfile;

#include "cute/arch/util.hpp"
#include "../../../hw_def/hw_def.h"

// CUTLASS cute library headers
#include <cutlass/cutlass.h>
#include "cutlass/numeric_types.h"
#include <cute/tensor.hpp>
#include <cute/arch/mma_sm90.hpp>
#include <cute/arch/mma_sm90_gmma.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cute/numeric/integer_sequence.hpp>

using namespace cute;

// ============================================================================
// Base Kernel Template
// ============================================================================

template<
  class ElementA,
  class ElementB,
  class ElementC,
  class TileShape_MNK,
  class RepeatTimes = cute::Int<1024>
>
__global__ void wgmma_latency_kernel(uint32_t *startClk, uint32_t *stopClk, uint32_t *checksum) {
  int thread_idx = threadIdx.x + blockDim.x * threadIdx.y + threadIdx.z * blockDim.x * blockDim.y ;
  int warp_group_idx = __shfl_sync(0xFFFFFFFF, thread_idx / cutlass::NumThreadsPerWarpGroup, 0);

  static constexpr GMMA::Major GmmaMajorA = cute::GMMA::Major::K;
  static constexpr GMMA::Major GmmaMajorB = cute::GMMA::Major::K;

  // Create the GMMA operation
  auto gmma_op = cute::GMMA::ss_op_selector<
  ElementA, ElementB, ElementC, TileShape_MNK, GmmaMajorA, GmmaMajorB>();
  using MMA_Op = decltype(gmma_op);

  // Create the TiledMma based on element types and tile shape
  using TiledMma = decltype(cute::make_tiled_mma(gmma_op));
  using MMA_Traits = typename TiledMma::Traits;
  TiledMma tiled_mma;
  MMA_Traits traits;

  // Create the fragment A, B, C
  // Define the smem layouts using GMMA layout helpers for K-major
  // Using Layout_K_INTER_Atom which has minimal swizzling (Swizzle<0,4,3> = identity)
  // Layout_K_INTER_Atom_Bits has shape (8, 128 bits) = (8, 4) for 32-bit elements
  constexpr int PIPE = 1;
  using SmemLayoutA = decltype(tile_to_shape(GMMA::Layout_K_INTER_Atom<ElementA>{},
    make_shape(shape<0>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<PIPE>{})));
  using SmemLayoutB = decltype(tile_to_shape(GMMA::Layout_K_INTER_Atom<ElementB>{},
      make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<PIPE>{})));

  // Allocate shared memory with proper size (using type aliases for constexpr evaluation)
  // There will be warnings as CUDA cannot determine if the sharedmem size is static
  // But for this ubench case, since we are only measuring GMMA instruction,
  // we can ignore the warnings on shmem allocation.
  // CUTLASS bypass this by passing the shmem size during kernel launch via
  // sizeof(typename GemmKernel::SharedStorage), but this means we have to
  // rewrite this function as part of a Class to access the SharedStorage type
  // defined with the M,N,K shape.
  __shared__ ElementA smem_A[cosize_v<SmemLayoutA>];
  __shared__ ElementB smem_B[cosize_v<SmemLayoutB>];

  // Create the layout objects for tensor construction
  SmemLayoutA sA_layout{};
  SmemLayoutB sB_layout{};


  // Create the tensors with GMMA-compatible layouts
  Tensor sA = make_tensor(make_smem_ptr(smem_A), sA_layout);  // (BLK_M, BLK_K, PIPE)
  Tensor sB = make_tensor(make_smem_ptr(smem_B), sB_layout);  // (BLK_N, BLK_K, PIPE)
  constexpr int MmaWarpGroups = size(TiledMma{}) / cutlass::NumThreadsPerWarpGroup;
  Layout warp_group_thread_layout = make_layout(Int<MmaWarpGroups>{},
                                                Int<cutlass::NumThreadsPerWarpGroup>{});
  auto thread_mma = tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx));

  Tensor tCsA = thread_mma.partition_A(sA);                                                 // (MMA,MMA_M,MMA_K,PIPE)
  Tensor tCsB = thread_mma.partition_B(sB);                                                 // (MMA,MMA_N,MMA_K,PIPE)

  // Allocate "fragments/descriptors"
  Tensor tCrA = thread_mma.make_fragment_A(tCsA);                                           // (MMA,MMA_M,MMA_K,PIPE)
  Tensor tCrB = thread_mma.make_fragment_B(tCsB);                                           // (MMA,MMA_N,MMA_K,PIPE)

  // Get fragment registers for accumulator with MN size
  auto accum = partition_fragment_C(tiled_mma, take<0,2>(TileShape_MNK{}));

  __syncthreads();

  // Start timing (only thread 0)
  uint32_t start = 0;
  if (thread_idx == 0) {
    asm volatile("mov.u32 %0, %%clock;" : "=r"(start) :: "memory");
  }
  __syncthreads();

  // Fence accumulator operands
  warpgroup_fence_operand(accum);

  // Arrive and execute WGMMA
  warpgroup_arrive();
  constexpr int repeat_times = static_value<RepeatTimes>();

  #pragma unroll
  for (int j = 0; j < repeat_times; j++) {
    // Call the fma method
    cute::gemm(tiled_mma, tCrA(_,_,_,0), tCrB(_,_,_,0), accum);
  }
  // Wait for WGMMA to complete
  warpgroup_commit_batch();
  warpgroup_wait<0>();
  warpgroup_fence_operand(accum);

  __syncthreads();

  // Stop timing
  uint32_t stop = 0;
  if (thread_idx == 0) {
    asm volatile("mov.u32 %0, %%clock;" : "=r"(stop) :: "memory");
  }

  // Write results
  if (thread_idx == 0) {
    startClk[blockIdx.x] = start;
    stopClk[blockIdx.x] = stop;

    // Compute checksum to prevent optimization
    uint32_t sum = reinterpret_cast<uint32_t*>(accum.data())[0];
    // Simple checksum over accumulator
    checksum[blockIdx.x] = sum;
  }
}

// ============================================================================
// Host Function Template
// ============================================================================

template<class ElementA, class ElementB, class ElementC, class TileShape_MNK, class RepeatTimes = cute::Int<1024>>
float run_wgmma_latency_test_typed() {
  // Allocate device memory
  uint32_t *startClk_g, *stopClk_g, *checksum_g;
  gpuErrchk(cudaMalloc(&startClk_g, sizeof(uint32_t)));
  gpuErrchk(cudaMalloc(&stopClk_g, sizeof(uint32_t)));
  gpuErrchk(cudaMalloc(&checksum_g, sizeof(uint32_t)));

  // Launch kernel with 128 threads (warpgroup size)
  dim3 grid(1);
  dim3 block(128);
  wgmma_latency_kernel<ElementA, ElementB, ElementC, TileShape_MNK, RepeatTimes><<<grid, block>>>(startClk_g, stopClk_g, checksum_g);

  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  // Copy results back
  uint32_t startClk, stopClk, checksum;
  gpuErrchk(cudaMemcpy(&startClk, startClk_g, sizeof(uint32_t), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(&stopClk, stopClk_g, sizeof(uint32_t), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(&checksum, checksum_g, sizeof(uint32_t), cudaMemcpyDeviceToHost));

  // Calculate latency
  constexpr int repeat_times = static_value<RepeatTimes>();
  float latency = ((float)(stopClk - startClk)) / ((float)repeat_times);

  // Cleanup
  cudaFree(startClk_g);
  cudaFree(stopClk_g);
  cudaFree(checksum_g);

  return latency;
}

// ============================================================================
// Helper Macro for Testing
// ============================================================================

#define TEST_MMA_CONFIG(EA, EB, EC, M, N, K, DESC) \
  do { \
    try { \
      using TileShape = decltype(make_shape(Int<M>{}, Int<N>{}, Int<K>{})); \
      float lat = run_wgmma_latency_test_typed<EA, EB, EC, TileShape>(); \
      printf("%-50s: %6.2f cycles\n", DESC, lat); \
      if (g_lat_gmma_outfile) fprintf(g_lat_gmma_outfile, "%s,%.2f\n", DESC, lat); \
    } catch (...) { \
      printf("%-50s: FAILED\n", DESC); \
      if (g_lat_gmma_outfile) fprintf(g_lat_gmma_outfile, "%s,FAILED\n", DESC); \
    } \
  } while(0)

#endif // LAT_GMMA_COMMON_H
