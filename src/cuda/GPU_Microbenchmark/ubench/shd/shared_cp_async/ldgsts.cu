#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

#include <cuda_pipeline.h>

template <typename T>
__global__ void pipeline_kernel_async(T *global, uint64_t *clock, size_t copy_count, size_t loop)
{
    extern __shared__ char s[];
    T *shared = reinterpret_cast<T *>(s);

    uint64_t clock_start = clock64();
    for (int j = 0; j < loop; j++)
    {
        for (size_t i = 0; i < copy_count; ++i)
        {
            __pipeline_memcpy_async(&shared[blockDim.x * i + threadIdx.x],
                                    &global[blockDim.x * i + threadIdx.x], sizeof(T));
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
    const size_t threads_per_block = 32;
    size_t copy_count = 100;
    size_t loop = 1024;
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <copy_count> <loop>\n";
        return 1;
    }

    copy_count = std::atoi(argv[1]);
    loop = std::atoi(argv[2]);

    const size_t total_elements = threads_per_block * copy_count;
    const size_t bytes = total_elements * sizeof(T);

    // Allocate and initialize host memory
    T *h_data = new T[total_elements];
    for (size_t i = 0; i < total_elements; ++i)
    {
        h_data[i] = static_cast<T>(i);
    }

    // Allocate device memory
    T *d_data;
    cudaMalloc(&d_data, bytes);
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);

    uint64_t *d_clock;
    uint64_t zero = 0;
    cudaMalloc(&d_clock, sizeof(uint64_t));
    cudaMemcpy(d_clock, &zero, sizeof(uint64_t), cudaMemcpyHostToDevice);

    // Launch kernel
    size_t shared_mem_size = threads_per_block * copy_count * sizeof(T);
    pipeline_kernel_async<T><<<1, threads_per_block, shared_mem_size>>>(d_data, d_clock, copy_count, loop);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaDeviceSynchronize();

    // Copy and print clock result
    uint64_t h_clock;
    cudaMemcpy(&h_clock, d_clock, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    printf("Total clock cycles: %llu\n", h_clock);

    // Clean up
    cudaFree(d_data);
    cudaFree(d_clock);
    delete[] h_data;

    return 0;
}
