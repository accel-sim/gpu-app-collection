#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

#include <cuda_pipeline.h>

template <typename T>
__global__ void pipeline_kernel_async(T *global, uint64_t *clock,
                                      size_t copy_count, size_t loop)
{
    extern __shared__ char s[];
    T *shared = reinterpret_cast<T *>(s);

    size_t block_offset = blockIdx.x * blockDim.x * copy_count;

    uint64_t clock_start = clock64();
    for (int j = 0; j < loop; j++)
    {
#pragma unroll(43)
        for (size_t i = 0; i < copy_count; ++i)
        {
            __pipeline_memcpy_async(&shared[blockDim.x * i + threadIdx.x],
                                    &global[block_offset + blockDim.x * i + threadIdx.x],
                                    sizeof(T));

        }
            __pipeline_commit();
            __pipeline_wait_prior(0);
    }
       

    uint64_t clock_end = clock64();

    __syncthreads();
    if (threadIdx.x == 0)
        atomicAdd(reinterpret_cast<unsigned long long *>(clock),
                  clock_end - clock_start);
}
int main(int argc, char **argv)
{
    using T = float;
    size_t loop = 1024;
    size_t num_blocks = 4;
    size_t threads_per_block = 256;
    size_t elems_per_block = 0; // optional arg

    if (argc < 4 || argc > 5)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <loop> <num_blocks> <threads_per_block> [elems_per_block]\n";
        return 1;
    }

    loop = std::atoi(argv[1]);
    num_blocks = std::atoi(argv[2]);
    threads_per_block = std::atoi(argv[3]);
    if (argc == 5)
        elems_per_block = std::atoi(argv[4]); // user override

    // Get device max shared memory
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    size_t max_shared_mem = prop.sharedMemPerBlockOptin;
    if (max_shared_mem == 0)
        max_shared_mem = prop.sharedMemPerBlock;

    // If not provided, compute elems_per_block from shared mem
    if (elems_per_block == 0)
    {
        elems_per_block = max_shared_mem / sizeof(T);
    }

    // Round down to multiple of threads_per_block for even distribution
    elems_per_block = (elems_per_block / threads_per_block) * threads_per_block;

    // Total elements = per-block capacity × number of blocks
    size_t total_elems = elems_per_block * num_blocks;
    size_t bytes = total_elems * sizeof(T);

    size_t copies_per_thread = elems_per_block / threads_per_block;

    std::cout << "Threads per block        = " << threads_per_block << "\n";
    std::cout << "Max shared mem per block = " << max_shared_mem / 1024 << " KB\n";
    std::cout << "Elems per block          = " << elems_per_block << "\n";
    std::cout << "Copies per thread        = " << copies_per_thread << "\n";
    std::cout << "Total elems              = " << total_elems << "\n";

    // Host data
    T *h_data = new T[total_elems];
    for (size_t i = 0; i < total_elems; i++)
        h_data[i] = static_cast<T>(i);

    // Device memory
    T *d_data;
    cudaMalloc(&d_data, bytes);
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);

    uint64_t *d_clock;
    uint64_t zero = 0;
    cudaMalloc(&d_clock, sizeof(uint64_t));
    cudaMemcpy(d_clock, &zero, sizeof(uint64_t), cudaMemcpyHostToDevice);

    // Opt-in to use max shared memory if needed
    cudaFuncSetAttribute(pipeline_kernel_async<T>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         max_shared_mem);

    // Launch kernel
    size_t shared_mem_size = elems_per_block * sizeof(T);
    size_t copy_count = elems_per_block / threads_per_block;

    pipeline_kernel_async<T><<<num_blocks, threads_per_block, shared_mem_size>>>(
        d_data, d_clock, copy_count, loop);

    // Copy and print clock result
    uint64_t h_clock;
    cudaMemcpy(&h_clock, d_clock, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    printf("Total clock cycles (summed across blocks): %llu\n", h_clock);

    // Clean up
    cudaFree(d_data);
    cudaFree(d_clock);
    delete[] h_data;

    return 0;
}
