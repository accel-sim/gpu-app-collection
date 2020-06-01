#if 0
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

void scan(int *data, int n, int *partial_sums, cudaStream_t stream = NULL)
{
  thrust::device_ptr<int> vec(data);
  thrust::exclusive_scan(vec, vec + n, vec);
}
#endif

#if 1

/*
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// Shuffle intrinsics SDK sample
// This sample demonstrates the use of the shuffle intrinsic
// First, a simple example of a prefix sum using the shuffle to
// perform a scan operation is provided.

// Scan using shfl - takes log2(n) steps
// This function demonstrates basic use of the shuffle intrinsic, __shfl_up,
// to perform a scan operation across a block.
// First, it performs a scan (prefix sum in this case) inside a warp
// Then to continue the scan operation across the block,
// each warp's sum is placed into shared memory.  A single warp
// then performs a shuffle scan on that shared memory.  The results
// are then uniformly added to each warp's threads.
// This pyramid type approach is continued by placing each block's
// final sum in global memory and prefix summing that via another kernel call, then
// uniformly adding across the input data via the uniform_add<<<>>> kernel.

__global__ void shfl_scan_test(int *data, int width, int *partial_sums=NULL)
{
// TODO: this creates compile error on ECX!
#ifndef ECX_TARGET
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 300
    __shared__ int shfl_mem[256];             // assume block size = 256
#endif

    extern __shared__ int sums[];
    int id = ((blockIdx.x * blockDim.x) + threadIdx.x);
    int lane_id = id % WARP_SIZE;

    // determine a warp_id within a block
    int warp_id = threadIdx.x / WARP_SIZE;

    // Below is the basic structure of using a shfl instruction
    // for a scan.
    // Record "value" as a variable - we accumulate it along the way
    int value = data[id];

    // Now accumulate in log steps up the chain
    // compute sums, with another thread's value who is
    // distance delta away (i).  Note
    // those threads where the thread 'i' away would have
    // been out of bounds of the warp are unaffected.  This
    // creates the scan sum.

#pragma unroll
    for (int i=1; i<=width; i*=2)
    {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 300
        int n = __shfl_up(value, i, width, shfl_mem);
#else
        int n = __shfl_up(value, i, width);
#endif
        if (lane_id >= i) value += n;
    }

    // value now holds the scan value for the individual thread
    // next sum the largest values for each warp

    // write the sum of the warp to smem
    if (threadIdx.x % WARP_SIZE == WARP_SIZE-1)
    {
        sums[warp_id] = value;
    }
    __syncthreads();

    //
    // scan sum the warp sums
    // the same shfl scan operation, but performed on warp sums
    //
    if (warp_id == 0)
    {
        int warp_sum = sums[lane_id];
        for (int i=1; i<=width; i*=2)
        {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 300
	    int n = __shfl_up(warp_sum, i, width, shfl_mem);
#else
            int n = __shfl_up(warp_sum, i, width);
#endif
            if (lane_id >= i) warp_sum += n;
        }
        sums[lane_id] = warp_sum;
    }
    __syncthreads();

    // perform a uniform add across warps in the block
    // read neighbouring warp's sum and add it to threads value
    int blockSum = 0;
    if (warp_id > 0)
    {
        blockSum = sums[warp_id-1];
    }

    value += blockSum;

    // Now write out our result
    if (partial_sums != NULL)
      data[id] = value-data[id];	// exclusive scan
    else 
      data[id] = value;

    // last thread has sum, write write out the block's sum
    if (partial_sums != NULL && threadIdx.x == blockDim.x-1)
    {
        partial_sums[blockIdx.x] = value;
    }
#endif
}

// Uniform add: add partial sums array
__global__ void uniform_add(int *data, int *partial_sums, int len)
{
    __shared__ int buf;
    int id = ((blockIdx.x * blockDim.x) + threadIdx.x);
    if (id > len) return;
    if (threadIdx.x == 0) {
        int sum = partial_sums[blockIdx.x];
        // TODO: need to parallelize this last reduction
	int block = blockIdx.x - blockIdx.x % 256 - 1;
        while (block >= 0) {
	  sum += partial_sums[block];
          block -= blockDim.x;
        }
        buf = sum;
    }
    __syncthreads();
    data[id] += buf;
}

void scan(int *data, int n, int *d_partial_sums, cudaStream_t stream = NULL)
{
    // TODO: round up to multiply of 256, do it with bits
    if (n % 256 != 0) {
      n = ((n + 255) / 256) * 256;
    }
    int blockSize = 256;
    int gridSize = n/blockSize;
    int nWarps = blockSize/32;
    int shmem_sz = nWarps * sizeof(int);
    int n_partialSums = n/blockSize;
    int partial_sz = n_partialSums*sizeof(int);
    int p_blockSize = min(n_partialSums, blockSize);
    int p_gridSize = (n_partialSums + p_blockSize-1)/p_blockSize;

    cudaMemsetAsync(d_partial_sums, 0, partial_sz, stream);

    shfl_scan_test<<<gridSize,blockSize, shmem_sz, stream>>>(data, 32, d_partial_sums);
    shfl_scan_test<<<p_gridSize,p_blockSize, shmem_sz, stream>>>(d_partial_sums, 32);
    if (gridSize > 1) 
      uniform_add<<<gridSize-1, blockSize, 0, stream>>>(data+blockSize, d_partial_sums, n);
}

#endif

