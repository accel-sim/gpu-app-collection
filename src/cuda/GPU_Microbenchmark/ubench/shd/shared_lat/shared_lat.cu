#include <cuda.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#define ITERS 2048
#ifdef TUNER
#include "../../../hw_def/hw_def.h"
#define SHARED_MEM_SIZE (32 * 1024 / 8)
// Launch only one thread to calcaulte the latency using a pointer-chasing
// array technique
#define THREADS_NUM 1
// iterate over the array ITERS times

#else
#include "../../../hw_def/common/gpuConfig.h"
#define SHARED_MEM_SIZE_BYTE (48 * 1024) // size in bytes, max 96KB for v100
#define SHARED_MEM_SIZE (SHARED_MEM_SIZE_BYTE / 8)
#define THREADS_NUM 32 // Launch only one thread to calcaulte the latency using a pointer-chasing array technique
// #define WARP_SIZE 32

#endif

// Measure latency of ITERS reads.
__global__ void shared_lat(uint32_t *startClk, uint32_t *stopClk,
                           uint64_t *dsink, uint32_t stride)
{

  // thread index
  uint32_t tid = threadIdx.x;
  uint32_t bid = blockIdx.x;
  uint32_t uid = bid * blockDim.x + tid;
  uint32_t n_threads = blockDim.x * gridDim.x;

  __shared__ uint64_t s[SHARED_MEM_SIZE]; // static shared memory

  // one thread to initialize the pointer-chasing array
  for (uint32_t i = uid; i < (SHARED_MEM_SIZE - stride); i += n_threads)
    s[i] = (i + stride) % SHARED_MEM_SIZE;

  if (uid == 0)
  {
    // initalize pointer chaser
    uint64_t p_chaser = 0;

    // start timing
    uint32_t start = 0;
    asm volatile("mov.u32 %0, %%clock;" : "=r"(start)::"memory");

    // pointer-chasing ITERS times
    for (uint32_t i = 0; i < ITERS; ++i)
    {
      p_chaser = s[p_chaser];
    }

    // stop timing
    uint32_t stop = 0;
    asm volatile("mov.u32 %0, %%clock;" : "=r"(stop)::"memory");

    // write time and data back to memory
    startClk[uid] = start;
    stopClk[uid] = stop;
    dsink[uid] = p_chaser;
  }
}

int main(int argc, char *argv[])
{

  intilizeDeviceProp(0, argc, argv);
#ifdef TUNER
  config.BLOCKS_NUM = 1;
  config.TOTAL_THREADS = THREADS_NUM * config.BLOCKS_NUM;
  config.THREADS_PER_SM = THREADS_NUM * config.BLOCKS_NUM;

  assert(SHARED_MEM_SIZE * sizeof(uint64_t) < config.MAX_SHARED_MEM_SIZE_PER_BLOCK);
#endif
  uint32_t *startClk = (uint32_t *)malloc(sizeof(uint32_t));
  uint32_t *stopClk = (uint32_t *)malloc(sizeof(uint32_t));
  uint64_t *dsink = (uint64_t *)malloc(sizeof(uint64_t));

  uint32_t *startClk_g;
  uint32_t *stopClk_g;
  uint64_t *dsink_g;

  gpuErrchk(cudaMalloc(&startClk_g, sizeof(uint32_t)));
  gpuErrchk(cudaMalloc(&stopClk_g, sizeof(uint32_t)));
  gpuErrchk(cudaMalloc(&dsink_g, sizeof(uint64_t)));

  shared_lat<<<1, THREADS_NUM>>>(startClk_g, stopClk_g, dsink_g, 1);
  gpuErrchk(cudaPeekAtLastError());

  gpuErrchk(cudaMemcpy(startClk, startClk_g, sizeof(uint32_t),
                       cudaMemcpyDeviceToHost));
  gpuErrchk(
      cudaMemcpy(stopClk, stopClk_g, sizeof(uint32_t), cudaMemcpyDeviceToHost));
  gpuErrchk(
      cudaMemcpy(dsink, dsink_g, sizeof(uint64_t), cudaMemcpyDeviceToHost));

  float lat = (float)(stopClk[0] - startClk[0]) / ITERS;
  printf("Shared Memory Latency  = %f cycles\n", lat);
  printf("Total Clk number = %u \n", stopClk[0] - startClk[0]);

  std::cout << "\n//Accel_Sim config: \n";
  std::cout << "-gpgpu_smem_latency " << (unsigned)(lat) << std::endl;

  return 1;
}
