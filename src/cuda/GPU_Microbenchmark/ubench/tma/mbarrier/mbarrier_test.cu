#include <cuda.h>
#include <cuda/barrier> // Standard C++ header
#include <stdio.h>

__global__ void mbarrier_kernel(int* out) {
    // 1. Declare barrier in Shared Memory
    __shared__ cuda::barrier<cuda::thread_scope_block> bar;

    // 2. Initialize (Thread 0 only)
    if (threadIdx.x == 0) {
        init(&bar, blockDim.x); 
    }
    __syncthreads();

    // 3. Arrive and Wait
    // This generates the 'mbarrier.arrive' and 'mbarrier.try_wait' 
    // instructions automatically.
    cuda::barrier<cuda::thread_scope_block>::arrival_token token = bar.arrive();
    bar.wait(std::move(token));

    // Success check
    if (threadIdx.x == 0) {
        *out = 1; 
    }
}

int main() {
    int* d_out;
    int h_out = 0;
    cudaMalloc(&d_out, sizeof(int));
    cudaMemset(d_out, 0, sizeof(int));

    mbarrier_kernel<<<1, 32>>>(d_out);
    
    cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
    
    if (h_out == 1) printf("mbarrier test passed (Standard C++).\n");
    else printf("mbarrier test failed.\n");

    return 0;
}