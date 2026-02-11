// LRC Max Merged Microbenchmark (NCU-based)
//
// Discovers the maximum number of read requests that can be coalesced
// into a single LRC (L2 Request Coalescer) entry on NVIDIA GPUs.
//
// Principle: Launch N blocks (1 warp each, 1 per SM) that all pointer-chase
// through the same sequence of L2 sectors (bypassing L1 with ld.global.cg).
// Blocks naturally stay roughly in lockstep since they all do identical work.
//
// Measurement: Use ncu hardware counters to compare SM-side sector requests
// (pre-LRC) with L2-side sector reads (post-LRC). The compression ratio
// reveals max_merged.
//
// Usage:
//   ./lrc_max_merged [N] [THREADS_PER_BLOCK] [ITERS] [SYNC_INTERVAL]
//     - N: Number of blocks (default: all SMs)
//     - THREADS_PER_BLOCK: Number of threads per block (default: 256)
//     - ITERS: Number of iterations (default: 4096)
//     - SYNC_INTERVAL: Number of iterations between grid-wide sync (default: ITERS / 8)
//
// NCU profiling:
//   ncu --metrics lrc__lts2lrc_sectors_op_read.sum.sum,lrc__xbar2gpc_sectors_op_read.sum.sum \
//       ./lrc_max_merged <N> <THREADS_PER_BLOCK> <ITERS> <SYNC_INTERVAL>
//
// Compile: nvcc -Xptxas -dlcm=cg lrc_max_merged.cu -o lrc_max_merged

#include <assert.h>
#include <cooperative_groups.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace cg = cooperative_groups;

#include "../../../hw_def/hw_def.h"

#define ARRAY_SIZE 4096    // uint64_t elements -> 32KB total
#define SECTOR_STRIDE 4    // 4 * sizeof(uint64_t) = 32 bytes = 1 sector
#define COMPUTE_ITERS 64   // Number of multiply-adds to insert after each load

__global__ void lrc_max_merged_kernel(uint64_t *data, uint64_t *dsink, uint32_t ITERS, uint32_t SYNC_INTERVAL) {
  // All threads in all blocks start at the same pointer (sector 0)
  uint64_t ptr = (uint64_t)data;

  // Block-level sync before starting
  asm volatile("bar.sync 0;");

  // Global synchronization for all blocks in the grid
  cg::grid_group grid = cg::this_grid();
  grid.sync();

  // Pointer chase across sectors:
  // - ld.global.cg bypasses L1, caches in L2 -> goes through LRC
  // - Data dependency (ptr = *ptr) prevents compiler optimization
  // - Each iteration: all blocks read same sector (same ptr value)
  //   since they started at the same address and follow the same chain
  // - ptr advances to next sector after each load
  //
  // After each load we insert a chain of dependent ALU ops so that
  // the loop body takes long enough for all blocks to converge on the
  // same sector before any block advances to the next one.  Without
  // this padding the loop is too tight and blocks slip out of the
  // coalescing window.
  uint64_t sink = 0;
  for (uint32_t i = 0; i < ITERS; i++) {
    asm volatile("ld.global.cg.u64 %0, [%0];" : "+l"(ptr)::"memory");

    // Dependent ALU padding: a chain of multiply-adds that the
    // compiler cannot remove (volatile asm, data dependency on ptr).
    // Each op depends on the previous result, serialising them.
    uint64_t tmp = ptr;
    #pragma unroll
    for (uint32_t j = 0; j < COMPUTE_ITERS; j++) {
      asm volatile("mad.lo.u64 %0, %0, %1, %2;"
                   : "+l"(tmp) : "l"((uint64_t)5), "l"((uint64_t)3) : );
    }
    sink += tmp;  // prevent dead-code elimination of the chain

    asm volatile("bar.sync 0;");
    // Periodic synchronization for grid-wide sync
    if (SYNC_INTERVAL > 0 && (i + 1) % SYNC_INTERVAL == 0)
      grid.sync();
  }
  
  asm volatile("bar.sync 0;");

  // Prevent dead code elimination
  dsink[blockIdx.x * blockDim.x + threadIdx.x] = ptr + sink;
  asm volatile("bar.sync 0;");
}

int main(int argc, char *argv[]) {
  initializeDeviceProp(0, argc, argv);

  unsigned sm_count = config.SM_NUMBER;

  // Number of blocks = CLI arg or all SMs
  unsigned N = sm_count;
  unsigned threads_per_block = config.THREADS_PER_BLOCK;
  uint32_t ITERS = 4096;
  uint32_t SYNC_INTERVAL = ITERS / 8;
  if (argc > 1) {
    N = (unsigned)atoi(argv[1]);
    threads_per_block = (unsigned)atoi(argv[2]);
    ITERS = (uint32_t)atoi(argv[3]);
    SYNC_INTERVAL = (uint32_t)atoi(argv[4]);
  }

  // Pointer chain array must fit in L2
  size_t array_bytes = ARRAY_SIZE * sizeof(uint64_t);
  assert(array_bytes < config.L2_SIZE);

  unsigned num_sectors = ARRAY_SIZE / SECTOR_STRIDE;

  printf("=== LRC Max Merged (NCU-based) ===\n");
  printf("SM_COUNT=%u, N_BLOCKS=%u, ITERS=%d, NUM_SECTORS=%u\n", sm_count, N,
         ITERS, num_sectors);
  printf("Profile with ncu to measure L2 sector compression.\n");

  // Allocate device memory
  uint64_t *posArray_g, *dsink_g;
  gpuErrchk(cudaMalloc(&posArray_g, array_bytes));
  gpuErrchk(cudaMalloc(&dsink_g, N * threads_per_block * sizeof(uint64_t)));

  // Initialize pointer chain on host using device pointer arithmetic
  // Chain: sector 0 -> sector 1 -> ... -> sector N-1 -> sector 0
  uint64_t *init = (uint64_t *)malloc(array_bytes);
  memset(init, 0, array_bytes);
  for (unsigned s = 0; s < num_sectors - 1; s++)
    init[s * SECTOR_STRIDE] =
        (uint64_t)(posArray_g + (s + 1) * SECTOR_STRIDE);
  init[(num_sectors - 1) * SECTOR_STRIDE] =
      (uint64_t)(posArray_g); // cycle back

  gpuErrchk(
      cudaMemcpy(posArray_g, init, array_bytes, cudaMemcpyHostToDevice));
  free(init);

  // Launch: N blocks, threads per block to enforce 2 block/SM
  // Cooperative launch required for grid-wide sync
  void *kernelArgs[] = {(void *)&posArray_g, (void *)&dsink_g, (void *)&ITERS, (void *)&SYNC_INTERVAL};
  gpuErrchk(cudaLaunchCooperativeKernel(
      (void *)lrc_max_merged_kernel, N, threads_per_block, kernelArgs));
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  printf("Kernel completed. Use ncu to analyze L2 sector counts.\n");

  // Cleanup
  cudaFree(posArray_g);
  cudaFree(dsink_g);

  return 0;
}
