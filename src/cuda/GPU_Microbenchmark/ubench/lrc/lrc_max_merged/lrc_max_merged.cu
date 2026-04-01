// LRC Max Merged Microbenchmark
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
 * @brief Synchronize within a cluster using mbarrier
 * 
 * @return
 */
__device__ __forceinline__ void sync_cluster(void) {
  /**
   * Synchronization
   */
  // Initialize mbarrier
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ uint64_t bar;
  // Barrier handles
  uint64_t *clusterleader_bar_handle;
  uint64_t *local_bar_handle;

  // Setup remote barrier address
  cg::cluster_group cluster = cg::this_cluster();
  unsigned int clusterBlockRank = cluster.block_rank();
  unsigned int clusterSize = cluster.dim_blocks().x;

  // Check if this threadblock is the cluster leader
  bool isClusterLeader = clusterBlockRank == 0;
  // Check if this thread is the thread leader
  bool isThreadLeader = threadIdx.x == 0;

  // Only cluster leader should initialize the barrier
  // other threadblocks should use remote barrier address
  if (isClusterLeader) {
    // Get total number of threads in the cluster
    unsigned int totalThreads = cluster.num_threads();

    // Only one thread should initialize the barrier
    if (isThreadLeader) {
      cuda::ptx::mbarrier_init(&bar, totalThreads);
    }
    __syncthreads();
    clusterleader_bar_handle = &bar;
    local_bar_handle = &bar;
  } else {
    // Other threadblocks should init local barrier
    // to wait for clusterleader's signal that all threadblocks
    // have arrived
    if (isThreadLeader) {
      cuda::ptx::mbarrier_init(&bar, 1);
    }
    __syncthreads();
    clusterleader_bar_handle = cluster.map_shared_rank(&bar, 0);
    local_bar_handle = &bar;
  }

  // Here is the synchronization part
  // 1. All threads in the cluster should arrive at the clusterleader's barrier
  // 2. Since mbarrier only supports waiting on local barrier:
  //    1. ClusterLeader will wait on its barrier that get released when all threadblocks have arrived
  //    2. Other threadblocks will wait on their local barrier that get released when clusterleader release its barrier
  // 3. Once clusterleader's barrier is released, all other threadblocks should launch the load to same sector address

  // First all threads arrive at the clusterleader's barrier
  cuda::ptx::mbarrier_arrive(cuda::ptx::sem_release, cuda::ptx::scope_cluster, cuda::ptx::space_cluster, clusterleader_bar_handle, 1);

  // Then based on the role, we will either wait on the barrier or release the barrier
  if (isClusterLeader) {
    // ClusterLeader will wait on its barrier that get released when all threadblocks have arrived
    while (!cuda::ptx::mbarrier_test_wait_parity(cuda::ptx::sem_acquire, cuda::ptx::scope_cluster, clusterleader_bar_handle, 0)) {}

    // Now ClusterLeader should release other threadblocks to proceed, we will execute this in a single warp
    // for each threadblock, we will release its barrier
    if (threadIdx.x < clusterSize && threadIdx.x != clusterBlockRank) {
      uint64_t *remote_bar_handle = cluster.map_shared_rank(&bar, threadIdx.x);
      cuda::ptx::mbarrier_arrive(cuda::ptx::sem_release, cuda::ptx::scope_cluster, cuda::ptx::space_cluster, remote_bar_handle, 1);
    }
  } else {
    // Other threadblocks will wait on their local barrier that get released when clusterleader release its barrier
    while (!cuda::ptx::mbarrier_test_wait_parity(cuda::ptx::sem_acquire, cuda::ptx::scope_cluster, local_bar_handle, 0)) {}
  }
  __syncthreads();
}

/**
 * @brief Kernel to flush L2 cache by reading through a large buffer
 *
 * Each thread reads one sector (32 bytes) using ld.global.cg to bypass L1
 * and pollute L2 with junk data, evicting all prior contents.
 */
__global__ void flush_l2_kernel(uint64_t *flush_buf, size_t num_sectors) {
  size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (idx < num_sectors) {
    // Each sector is 32 bytes = 4 uint64_t elements
    size_t offset = idx * 4;
    uint64_t val;
    asm volatile("ld.global.cg.u64 %0, [%1];" : "=l"(val) : "l"(&flush_buf[offset]));
    // Add 1
    val++;
    // Write back to consume val and prevent dead code elimination
    asm volatile("st.global.cg.u64 [%0], %1;" : : "l"(&flush_buf[offset]), "l"(val));
  }
}

/**
 * @brief Flush L2 cache by allocating and reading a buffer 2x the L2 size
 */
void flush_l2() {
  size_t flush_size = config.L2_SIZE * 2;
  size_t num_sectors = flush_size / 32;

  uint64_t *flush_buf;
  gpuErrchk(cudaMalloc(&flush_buf, flush_size));
  gpuErrchk(cudaMemset(flush_buf, 0, flush_size));

  unsigned threads = 256;
  unsigned blocks = (num_sectors + threads - 1) / threads;
  flush_l2_kernel<<<blocks, threads>>>(flush_buf, num_sectors);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  cudaFree(flush_buf);
}

/**
 * @brief Kernel to test LRC maximum merged count per entry
 * 
 * @param data Same sector data for all warps to access for LRC merge
 * @param dsink Global sink to avoid optimization
 * @return __global__ 
 */
__global__ void lrc_max_merged_kernel(uint64_t *data, uint64_t *dsink, unsigned active_threads_per_cluster) {
  // The LRC max merged count should be reflected from NCU metrics

  // Shmem buffer
  __shared__ volatile uint64_t smem_buffer[16];

  // Get thread index within a cluster
  cg::cluster_group cluster = cg::this_cluster();
  unsigned thread_within_cluster = cluster.thread_rank();

  // Synchronize within a cluster
  // If threadblocks are within the same GPC, they will be relatively in sync
  // anyway
  // sync_cluster();

  /**
   * LRC max merged test
   */
  // Now threadblocks within a cluster should be relatively in sync for testing LRC coalescing
  // Load the same sector data into shared memory
  // Use ld.global.cg to bypass L1 cache MSHR
  // Send down one sector per warp, make this explicit despite in
  // hardware with this "if", loads are still coalesced into 1 sector
  if (thread_within_cluster < active_threads_per_cluster) {
    // Only one sector access per warp
    if (threadIdx.x % 32 == 0) {
      uint64_t data_value;
      asm volatile("{\t\n"
                  "ld.global.cg.u64 %0, [%1];\n\t"
                  "}"
                  : "=l"(data_value)
                  : "l"(&data[0]));
      smem_buffer[0] = data_value;
    }
  }
  __syncthreads();

  // Write to global sink to prevent optimization
  dsink[0] = smem_buffer[0];
}

__global__ void lrc_max_merged_kernel_cooperative(uint64_t *data, uint64_t *dsink) {
  __shared__ volatile uint64_t smem_buffer[16];

  // Global synchronization for all blocks in the grid
  cg::grid_group grid = cg::this_grid();
  grid.sync();

  if (threadIdx.x % 32 == 0) {
    uint64_t data_value;
    asm volatile("{\t\n"
                "ld.global.cg.u64 %0, [%1];\n\t"
                "}"
                : "=l"(data_value)
                : "l"(&data[0]));
    smem_buffer[0] = data_value;
  }
  __syncthreads();

  // Write to global sink to prevent optimization
  dsink[0] = smem_buffer[0];
}

int main(int argc, char *argv[]) {
  initializeDeviceProp(0, argc, argv);
  enum KernelLaunchMode {
    NORMAL = 0,
    CLUSTER = 1,
    COOPERATIVE = 2,
  };
  const char *launch_mode_str[] = {
    "NORMAL",
    "CLUSTER",
    "COOPERATIVE",
  };

  // Size of threadblock cluster
  const unsigned MAX_CLUSTER_SIZE = 16;
  unsigned N = MAX_CLUSTER_SIZE;
  unsigned cluster_size = MAX_CLUSTER_SIZE;
  unsigned threads_per_block = 256;
  bool provide_active_threads_per_cluster = false;
  unsigned active_threads_per_cluster = cluster_size * threads_per_block;
  bool flush_l2_enabled = false;
  KernelLaunchMode launch_mode = NORMAL;
  const char *optstring = "N:C:T:A:m:F";
  // CLI parsing
  int opt;
  while ((opt = getopt(argc, argv, optstring)) != -1) {
    switch (opt) {
      case 'N':
        N = atoi(optarg);
        break;
      case 'C':
        cluster_size = atoi(optarg);
        assert(cluster_size <= MAX_CLUSTER_SIZE && "cluster_size is out of range, must be less than 16 on Hopper");
        break;
      case 'T':
        threads_per_block = atoi(optarg);
        break;
      case 'A':
        active_threads_per_cluster = atoi(optarg);
        provide_active_threads_per_cluster = true;
        break;
      case 'm':
        launch_mode = static_cast<KernelLaunchMode>(atoi(optarg));
        assert(launch_mode == NORMAL || launch_mode == CLUSTER || launch_mode == COOPERATIVE && "launch_mode must be 0 (NORMAL), 1 (CLUSTER), or 2 (COOPERATIVE)");
        break;
      case 'F':
        flush_l2_enabled = true;
        break;
      default:
        printf("Usage: %s -N <number of threadblocks:default=16> -C <cluster_size:default=16> -T <threads_per_block:default=256> -A <active_threads_per_cluster:default=256> -m <launch_mode:default=0 (NORMAL), 1 (CLUSTER), or 2 (COOPERATIVE)> -F (flush L2 cache before kernel)\n", argv[0]);
        return 1;
    }
  }

  if (!provide_active_threads_per_cluster) {
    // Default to all threads in a cluster
    active_threads_per_cluster = cluster_size * threads_per_block;
  }

  if (launch_mode == CLUSTER || launch_mode == NORMAL) {
    assert(active_threads_per_cluster <= cluster_size * threads_per_block && "active_threads_per_cluster is out of range, must be less than cluster_size * threads_per_block");
  }

  // Initialize the data and global sink with single value
  uint64_t *data_g;
  gpuErrchk(cudaMalloc(&data_g, sizeof(uint64_t)));
  gpuErrchk(cudaMemset(data_g, 0, sizeof(uint64_t)));
  uint64_t *dsink_g;
  gpuErrchk(cudaMalloc(&dsink_g, sizeof(uint64_t)));
  gpuErrchk(cudaMemset(dsink_g, 0, sizeof(uint64_t)));
  
  printf("=== LRC Max Merged ===\n");
  unsigned num_cluster = launch_mode == CLUSTER ? N / cluster_size : N;
  unsigned max_concurrent_access = active_threads_per_cluster / 32 * num_cluster;
  printf("N_BLOCKS=%u, THREADS_PER_BLOCK=%u, MAX_CONCURRENT_ACCESS=%u, LAUNCH_MODE=%s\n", N, threads_per_block, max_concurrent_access, launch_mode_str[launch_mode]);
  if (launch_mode == CLUSTER) {
    printf("CLUSTER_SIZE=%u\n", cluster_size);
  }
  printf("Profile with ncu to measure LRC max merged count.\n");

  if (flush_l2_enabled) {
    printf("Flushing L2 cache (size=%zu bytes)...\n", config.L2_SIZE);
    flush_l2();
  }

  if (launch_mode == CLUSTER) {
    printf("Launching with threadblock cluster to schedule threadblocks in the same GPC...\n");
    // Launch configurable cluster shape
    // Enable upto 16 cluster size
    gpuErrchk(cudaFuncSetAttribute(lrc_max_merged_kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
    // Kernel launch configuration
    cudaLaunchConfig_t config = {0};
    config.gridDim          = dim3{N};  // N blocks
    config.blockDim         = dim3{threads_per_block}; // threads_per_block threads per block
    cudaLaunchAttribute attribute[1];
    attribute[0].id               = cudaLaunchAttributeClusterDimension;
    attribute[0].val.clusterDim.x = cluster_size; // Cluster size is also Nx1x1
    attribute[0].val.clusterDim.y = 1;
    attribute[0].val.clusterDim.z = 1;
    config.attrs    = attribute;
    config.numAttrs = 1;

    cudaLaunchKernelEx(&config, lrc_max_merged_kernel, data_g, dsink_g, active_threads_per_cluster);
  } else if (launch_mode == NORMAL) {
    printf("Launching threadblocks normally...\n");
    lrc_max_merged_kernel<<<N, threads_per_block>>>(data_g, dsink_g, active_threads_per_cluster);
  } else if (launch_mode == COOPERATIVE) {
    printf("Launching with cooperative kernel...\n");
    void *kernelArgs[] = {(void *)&data_g, (void *)&dsink_g};
    gpuErrchk(cudaLaunchCooperativeKernel(
        (void *)lrc_max_merged_kernel_cooperative, N, threads_per_block, kernelArgs));
  }
  
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  printf("Kernel completed. Use ncu to analyze LRC max merged count.\n");

  // Cleanup
  cudaFree(data_g);
  cudaFree(dsink_g);

  return 0;
}
