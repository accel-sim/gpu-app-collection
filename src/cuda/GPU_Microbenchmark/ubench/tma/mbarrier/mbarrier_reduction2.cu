#include <cuda.h>
#include <cuda/barrier>
#include <stdio.h>

#define BLOCK_SIZE 256

// Helper to burn time and force a race condition
__device__ void sabotage_delay(int cycles) {
    long long start = clock64();
    while (clock64() - start < cycles) {
        // Busy wait to hold up the thread
    }
}

// --------------------------------------------------------
// KERNEL 1: Reference Implementation (__syncthreads)
// --------------------------------------------------------
__global__ void reduction_reference(int* in, int* out) {
    __shared__ int s_data[BLOCK_SIZE];

    // Load data into shared memory
    int tid = threadIdx.x;

    // SABOTAGE: Delay writers here too to prove __syncthreads works
    if (tid >= BLOCK_SIZE / 2) {
        sabotage_delay(50000); 
    }

    s_data[tid] = in[tid];
    __syncthreads(); // Standard barrier

    // Standard tree reduction

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_data[tid] += s_data[tid + stride];
        }
        __syncthreads(); // Barrier essential for correctness
    }

    // Write result
    if (tid == 0) *out = s_data[0];
}

// --------------------------------------------------------
// KERNEL 2: mbarrier Implementation
// --------------------------------------------------------
__global__ void reduction_mbarrier(int* in, int* out) {
    __shared__ int s_data[BLOCK_SIZE];
    
    // 1. Declare mbarrier in shared memory
    __shared__ cuda::barrier<cuda::thread_scope_block> bar;

    // 2. Initialize barrier (Thread 0 only)
    if (threadIdx.x == 0) {
        init(&bar, blockDim.x); 
    }
    __syncthreads(); // Wait for init to finish

    // Load data
    int tid = threadIdx.x;

    // --- THE SABOTAGE ---
    // We delay the threads that provide data (High IDs).
    // The threads that read data (Low IDs) will rush ahead.
    if (tid >= BLOCK_SIZE / 2) {
        sabotage_delay(50000); // Delay 500 cycles
    }
    
    // Write Data (Late)
    s_data[tid] = in[tid];

    // THE TEST:
    // If barrier is broken, Low IDs reach here, see High IDs aren't done,
    // ignore it, and read garbage data below.
    auto token = bar.arrive(); 
    bar.wait(std::move(token));

    // Standard tree reduction using mbarrier
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_data[tid] += s_data[tid + stride];
        }
        
        // 3. BARRIER SYNCHRONIZATION == __syncthreads
        token = bar.arrive();
        bar.wait(std::move(token));
    }

    if (tid == 0) *out = s_data[0];
}

int main() {
    const int N = BLOCK_SIZE;
    const int BYTES = N * sizeof(int);

    // Host data
    int* h_in = (int*)malloc(BYTES);
    int h_ref_out = 0;
    int h_mbar_out = 0;

    // Initialize input (Array of 1s, so Sum should be 256)
    for (int i = 0; i < N; i++) h_in[i] = 1;

    // Device data
    int *d_in, *d_ref_out, *d_mbar_out;
    cudaMalloc(&d_in, BYTES);
    cudaMalloc(&d_ref_out, sizeof(int));
    cudaMalloc(&d_mbar_out, sizeof(int));
    cudaMemcpy(d_in, h_in, BYTES, cudaMemcpyHostToDevice);

    // ----------------------------------------------------
    // Run Reference Kernel
    // ----------------------------------------------------
    reduction_reference<<<1, BLOCK_SIZE>>>(d_in, d_ref_out);
    cudaMemcpy(&h_ref_out, d_ref_out, sizeof(int), cudaMemcpyDeviceToHost);
    
    // ----------------------------------------------------
    // Run mbarrier Kernel
    // ----------------------------------------------------
    reduction_mbarrier<<<1, BLOCK_SIZE>>>(d_in, d_mbar_out);
    cudaMemcpy(&h_mbar_out, d_mbar_out, sizeof(int), cudaMemcpyDeviceToHost);

    // ----------------------------------------------------
    // Verify
    // ----------------------------------------------------
    printf("Reference Sum: %d\n", h_ref_out);
    printf("mbarrier  Sum: %d\n", h_mbar_out);

    bool pass = (h_ref_out == h_mbar_out && h_ref_out == N);
    if (pass) {
        printf("SUCCESS: mbarrier matches reference output!\n");
    } else {
        printf("FAILURE: Results mismatch.\n");
        printf("Likely Cause: Threads raced ahead without waiting.\n");
    }

    // Cleanup
    free(h_in);
    cudaFree(d_in);
    cudaFree(d_ref_out);
    cudaFree(d_mbar_out);

    return pass ? 0 : 1;
}