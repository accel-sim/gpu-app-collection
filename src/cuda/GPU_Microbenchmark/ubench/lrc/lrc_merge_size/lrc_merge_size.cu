// LRC Merge Size Microbenchmark
// Use mbarrier and threadblock cluster to ensure
// best synchronization among warps sending request
// to the same L2 sector

#include <assert.h>
#include <bits/getopt_core.h>
#include <cstdint>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../../hw_def/common/gpuConfig.h"
#include "../../../hw_def/hw_def.h"

#include <cuda.h>
#include <cuda/ptx>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

/**
 * @brief Kernel to test LRC merge size being sector or cacheline
 * 
 * @param data Same sector data for all warps to access for LRC merge
 * @param dsink Global sink to avoid optimization
 * @return __global__ 
 */
__global__ void __cluster_dims__(4, 1, 1) lrc_merge_size_kernel(uint8_t *data, uint8_t *dsink) {
  // The LRC merge size should be reflected from NCU metrics

  // Shmem buffer
  __shared__ volatile uint8_t smem_buffer[16];

  // Get thread index within a cluster
  cg::cluster_group cluster = cg::this_cluster();
  unsigned block_rank = cluster.block_rank();


  /**
   * LRC merge size test
   */
  // Base on the block rank in a cluster, each block have one warp to access the data + i
  // with i being the block rank
  // Only one sector access per warp
  if (threadIdx.x % 32 == 0) {
    uint64_t data_value;
    // As sector size is 32B, so we access at 32B stride
    uint8_t *ptr = data + block_rank * 32;
    asm volatile("{\t\n"
                "ld.global.cg.u64 %0, [%1];\n\t"
                "}"
                : "=l"(data_value)
                : "l"(ptr));
    smem_buffer[0] = (uint8_t)data_value;
  }
  __syncthreads();

  // Write to global sink to prevent optimization
  dsink[0] = smem_buffer[0];
}

int main(int argc, char *argv[]) {
  initializeDeviceProp(0, argc, argv);

  // Initialize the data and global sink with single value
  uint8_t *data_g;
  const unsigned SECTOR_SIZE = 32;
  const unsigned CACHELINE_SIZE = 128;

  // cudaMalloc is aligned to 256B, so this array is in a single cacheline
  gpuErrchk(cudaMalloc(&data_g, sizeof(uint8_t) * CACHELINE_SIZE));
  gpuErrchk(cudaMemset(data_g, 0, sizeof(uint8_t) * CACHELINE_SIZE));
  uint8_t *dsink_g;
  gpuErrchk(cudaMalloc(&dsink_g, sizeof(uint8_t)));
  gpuErrchk(cudaMemset(dsink_g, 0, sizeof(uint8_t)));
  
  printf("=== LRC Merge Size ===\n");
  printf("Profile with ncu to measure LRC merge size.\n");

  printf("Launching threadblocks normally...\n");
  // 2 clusters, 8 threadblocks, each TB has 1 warp, each warp has 32 threads
  lrc_merge_size_kernel<<<8, 32>>>(data_g, dsink_g);
  
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  printf("Kernel completed. Use ncu to analyze LRC merge size.\n");

  // Cleanup
  cudaFree(data_g);
  cudaFree(dsink_g);

  return 0;
}
