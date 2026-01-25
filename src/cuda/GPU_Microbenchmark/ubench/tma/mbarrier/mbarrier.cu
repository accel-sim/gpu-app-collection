// Adapt from https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=tma#using-tma-to-transfer-multi-dimensional-arrays
#include <cudaTypedefs.h> // PFN_cuTensorMapEncodeTiled, CUtensorMap
#include <cuda.h>         // CUtensormap
#include <cuda/barrier>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unordered_map>
using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace ptx = cuda::ptx;

/*
 * Test cuda program to map mbarrier related PTX instructions to SASS.
 * 
 */

#define CUDA_SAFECALL(call)                                                 \
    {                                                                       \
        call;                                                               \
        cudaError err = cudaGetLastError();                                 \
        if (cudaSuccess != err)                                             \
        {                                                                   \
            fprintf(                                                        \
                stderr,                                                     \
                "Cuda error in function '%s' file '%s' in line %i : %s.\n", \
                #call, __FILE__, __LINE__, cudaGetErrorString(err));        \
            fflush(stderr);                                                 \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    }

__global__ __noinline__ void test_mbarrier_kernel() {
    // mbarrier object is 64bit in shared memory
    __shared__ uint64_t mbarrier;
    uint64_t state;
    int32_t count; 
    // Initialize the mbarrier with PTX asm
    int block_size = blockDim.x * blockDim.y;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        asm("mbarrier.init.shared::cta.b64 [%0], %1;" : : "l"(&mbarrier), "r"(block_size) : "memory");
    }
    __syncthreads();

    // Expect on the mbarrier
#if __CUDA_ARCH__ >= 900
    int bytes_per_thread = 4;
    asm("mbarrier.expect_tx.shared::cta.b64 [%0], %1;" : : "l"(&mbarrier), "r"(bytes_per_thread) : "memory");
    __syncthreads();

    // Complete on the mbarrier
    asm("mbarrier.complete_tx.shared::cta.b64 [%0], %1;" : : "l"(&mbarrier), "r"(bytes_per_thread) : "memory");
    __syncthreads();

    // All threads in the block arrive on the mbarrier
    asm("mbarrier.arrive.b64 %0, [%1], %2;" : "=l"(state) : "l"(&mbarrier), "n"(1) : "memory");
    __syncthreads();

    // Arrive and expect on the mbarrier
    asm("mbarrier.arrive.expect_tx.b64 %0, [%1], %2;" : "=l"(state) : "l"(&mbarrier), "n"(2) : "memory");
    __syncthreads();

    // Arrive and drop
    asm("mbarrier.arrive_drop.b64 %0, [%1], %2;" : "=l"(state) : "l"(&mbarrier), "n"(3) : "memory");
    __syncthreads();
#else
    // For sm_80
    // All threads in the block arrive on the mbarrier
    asm("mbarrier.arrive.noComplete.b64 %0, [%1], %2;" : "=l"(state) : "l"(&mbarrier), "n"(1) : "memory");
    __syncthreads();

    // Arrive and drop
    asm("mbarrier.arrive_drop.noComplete.b64 %0, [%1], %2;" : "=l"(state) : "l"(&mbarrier), "n"(3) : "memory");
    __syncthreads();
#endif

#if __CUDA_ARCH__ >= 900
    // Arrive and drop
    asm("mbarrier.arrive_drop.expect_tx.b64 %0, [%1], %2;" : "=l"(state) : "l"(&mbarrier), "n"(4) : "memory");
    __syncthreads();
#endif

    // Get pending count
    asm("mbarrier.pending_count.b64 %0, %1;" : "=r"(count) : "l"(state) : "memory");
    // Prevent optimizing away
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        printf("Pending count: %d\n", count);
    }
    __syncthreads();

    // cp async barrier arrive
    asm("cp.async.mbarrier.arrive.shared::cta.b64 [%0];" : : "l"(&mbarrier) : "memory");
    __syncthreads();
    asm("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" : : "l"(&mbarrier) : "memory");
    __syncthreads();

    // Wait on the mbarrier
#if __CUDA_ARCH__ >= 900
    asm ("\n\t"
         ".reg .pred complete;\n\t"
         "mbarrier.test_wait.parity.b64 complete, [%0], %1;"
         : : "l"(&mbarrier), "n"(0) : "memory"
    );
    __syncthreads();
    asm ("\n\t"
        "mbarrier.try_wait.parity.b64 complete, [%0], %1;"
        : : "l"(&mbarrier), "n"(0) : "memory"
    );
    __syncthreads();
#endif
}

int main(int argc, char *argv[]) {
    CUDA_SAFECALL((test_mbarrier_kernel<<<1, 1>>>()));
    CUDA_SAFECALL(cudaDeviceSynchronize());

    printf("Mbarrier test completed\n");
    return 0;
}
