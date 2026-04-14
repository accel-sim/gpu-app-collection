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

__global__ __noinline__ void test_mbarrier_kernel(uint64_t *sink) {
    // mbarrier object is 64bit in shared memory
    __shared__ uint64_t mbarrier;
    uint64_t state;
    int32_t count; 
    // Initialize the mbarrier with PTX asm
    int block_size = blockDim.x * blockDim.y;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        asm("mbarrier.init.shared::cta.b64 [%0], %1;" : : "l"(&mbarrier), "r"(0x1234) : "memory");
    }
    __syncthreads();

    // Expect on the mbarrier
#if __CUDA_ARCH__ >= 900
    int bytes_per_thread = 4;
    asm("barrier.sync 0;");
    // SYNCS.ARRIVE.TRANS64.RED.A0TR
    asm("mbarrier.expect_tx.shared::cta.b64 [%0], %1;" : : "l"(&mbarrier), "r"(bytes_per_thread) : "memory");
    asm("mbarrier.expect_tx.shared::cluster.b64 [%0], %1;" : : "l"(&mbarrier), "r"(2*bytes_per_thread) : "memory");

    // Complete on the mbarrier
    asm("barrier.sync 1;");
    // SYNCS.ARRIVE.TRANS64.RED.A0TX
    asm("mbarrier.complete_tx.shared::cta.b64 [%0], %1;" : : "l"(&mbarrier), "r"(3*bytes_per_thread) : "memory");

    // All threads in the block arrive on the mbarrier
    asm("barrier.sync 2;");
    // SYNCS.ARRIVE.TRANS64.RED.A1T0 (no count specified, return value is not used)
    asm("mbarrier.arrive.b64 %0, [%1];" : "=l"(state) : "l"(&mbarrier) : "memory");
    asm("mbarrier.arrive.b64 _, [%0];" :: "l"(&mbarrier) : "memory");
    // SYNCS.ARRIVE.TRANS64.A1T0 (return value is used)
    asm("mbarrier.arrive.b64 %0, [%1];" : "=l"(state) : "l"(&mbarrier) : "memory");
    *sink += state;
    // SYNCS.ARRIVE.TRANS64.RED.ART0 (return value state is not used)
    asm("mbarrier.arrive.b64 %0, [%1], %2;" : "=l"(state) : "l"(&mbarrier), "n"(4) : "memory");
    asm("mbarrier.arrive.b64 _ , [%0], %1;" : : "l"(&mbarrier), "n"(4) : "memory");
    // SYNCS.ARRIVE.TRANS64.ART0 (return value state is used)
    asm("mbarrier.arrive.b64 %0, [%1], %2;" : "=l"(state) : "l"(&mbarrier), "n"(5) : "memory");
    *sink += state;

    // ARRIVE with no complete
    asm("barrier.sync 3;");
    // SYNCS.ARRIVE.TRANS64.RED.ART0
    asm("mbarrier.arrive.noComplete.b64 %0, [%1], %2;" : "=l"(state) : "l"(&mbarrier), "n"(5) : "memory");
    // SYNCS.ARRIVE.TRANS64.TMASK.ART0 (return value is used)
    asm("mbarrier.arrive.noComplete.b64 %0, [%1], %2;" : "=l"(state) : "l"(&mbarrier), "n"(6) : "memory");
    *sink += state;

    // Arrive and expect on the mbarrier
    asm("barrier.sync 4;");
    // SYNCS.ARRIVE.TRANS64.RED (return value is not used)
    asm("mbarrier.arrive.expect_tx.b64 %0, [%1], %2;" : "=l"(state) : "l"(&mbarrier), "n"(6) : "memory");
    asm("mbarrier.arrive.expect_tx.b64 _, [%0], %1;" :: "l"(&mbarrier), "n"(6) : "memory");
    // SYNCS.ARRIVE.TRANS64 (return value is used)
    asm("mbarrier.arrive.expect_tx.b64 %0, [%1], %2;" : "=l"(state) : "l"(&mbarrier), "n"(6) : "memory");
    *sink += state;

    // Arrive and drop
    asm("barrier.sync 5;");
    // SYNCS.ARRIVE.TRANS64.RED.OPTOUT.A1T0 (not using return value)
    asm("mbarrier.arrive_drop.b64 %0, [%1];" : "=l"(state) : "l"(&mbarrier) : "memory");
    asm("mbarrier.arrive_drop.b64 _, [%0];" :: "l"(&mbarrier) : "memory");
    // SYNCS.ARRIVE.TRANS64.OPTOUT.A1T0 (using return value)
    asm("mbarrier.arrive_drop.b64 %0, [%1];" : "=l"(state) : "l"(&mbarrier) : "memory");
    *sink += state;

    // SYNCS.ARRIVE.TRANS64.RED.OPTOUT.ART0 (not using return value)
    asm("mbarrier.arrive_drop.b64 %0, [%1], %2;" : "=l"(state) : "l"(&mbarrier), "n"(3) : "memory");
    asm("mbarrier.arrive_drop.b64 _, [%0], %1;" :: "l"(&mbarrier), "n"(3) : "memory");
    // SYNCS.ARRIVE.TRANS64.OPTOUT.ART0 (using return value)
    asm("mbarrier.arrive_drop.b64 %0, [%1], %2;" : "=l"(state) : "l"(&mbarrier), "n"(3) : "memory");
    *sink += state;

    asm("barrier.sync 6;");
    // SYNCS.ARRIVE.TRANS64.RED.OPTOUT.ART0
    asm("mbarrier.arrive_drop.noComplete.b64 %0, [%1], %2;" : "=l"(state) : "l"(&mbarrier), "n"(3) : "memory");
    asm("mbarrier.arrive_drop.noComplete.b64 _, [%0], %1;" :: "l"(&mbarrier), "n"(3) : "memory");
    // SYNCS.ARRIVE.TRANS64.TMASK.OPTOUT.ART0
    asm("mbarrier.arrive_drop.noComplete.b64 %0, [%1], %2;" : "=l"(state) : "l"(&mbarrier), "n"(3) : "memory");
    *sink += state;

#else
    // For sm_80
    // All threads in the block arrive on the mbarrier
    asm("barrier.sync 5;");
    asm("mbarrier.arrive.noComplete.b64 %0, [%1], %2;" : "=l"(state) : "l"(&mbarrier), "n"(1) : "memory");

    // Arrive and drop with no complete
    asm("barrier.sync 6;");
    asm("mbarrier.arrive_drop.noComplete.b64 %0, [%1], %2;" : "=l"(state) : "l"(&mbarrier), "n"(3) : "memory");
#endif

#if __CUDA_ARCH__ >= 900
    // Arrive and drop expect tx   
    asm("barrier.sync 7;");
    // SYNCS.ARRIVE.TRANS64.RED.OPTOUT
    asm("mbarrier.arrive_drop.expect_tx.b64 %0, [%1], %2;" : "=l"(state) : "l"(&mbarrier), "n"(4) : "memory");
    asm("mbarrier.arrive_drop.expect_tx.b64 _, [%0], %1;" :: "l"(&mbarrier), "n"(5) : "memory");
    // SYNCS.ARRIVE.TRANS64.OPTOUT RA, [URB], RC; RC is the transaction count
    asm("mbarrier.arrive_drop.expect_tx.b64 %0, [%1], %2;" : "=l"(state) : "l"(&mbarrier), "n"(6) : "memory");
    *sink += state;
#endif

    // Get pending count
    asm("barrier.sync 8;");
    // This is just a list of arithematic operations on the opaque state variable
    asm("mbarrier.pending_count.b64 %0, %1;" : "=r"(count) : "l"(state) : "memory");
    // Prevent optimizing away
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        printf("Pending count: %d\n", count);
    }
    __syncthreads();

    // cp async barrier arrive
    asm("barrier.sync 9;");
    // SYNCS.ARRIVE.TRANS64.RED.A0T1 RZ, [URB], RZ
    // ARRIVES.LDGSTSBAR.64.TRANSCNT [URB]
    asm("cp.async.mbarrier.arrive.shared::cta.b64 [%0];" : : "l"(&mbarrier) : "memory");
    asm("barrier.sync 10;");
    // ARRIVES.LDGSTSBAR.64.ARVCNT [URB]
    asm("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" : : "l"(&mbarrier) : "memory");

    // Wait on the mbarrier
#if __CUDA_ARCH__ >= 900
    asm("barrier.sync 11;");
    // SYNCS.PHASECHK.TRANS64 PT, [UR4], RZ
    asm ("\n\t"
         ".reg .pred complete;\n\t"
         "mbarrier.test_wait.parity.b64 complete, [%0], %1;"
         : : "l"(&mbarrier), "n"(0) : "memory"
    );
    asm("barrier.sync 12;");
    // SYNCS.PHASECHK.TRANS64 PT, [UR4], R0
    asm ("\n\t"
        "mbarrier.test_wait.parity.b64 complete, [%0], %1;"
        : : "l"(&mbarrier), "n"(1) : "memory"
    );
    asm("barrier.sync 0;");
    // SYNCS.PHASECHK.TRANS64.TRYWAIT PT, [UR4], RZ
    asm ("\n\t"
        "mbarrier.try_wait.parity.b64 complete, [%0], %1;"
        : : "l"(&mbarrier), "n"(0) : "memory"
    );
    asm("barrier.sync 1;");
    // SYNCS.PHASECHK.TRANS64.TRYWAIT PT, [UR4], R0
    asm ("\n\t"
        "mbarrier.try_wait.parity.b64 complete, [%0], %1;"
        : : "l"(&mbarrier), "n"(1) : "memory"
    );
    asm("barrier.sync 2;");
    // SYNCS.PHASECHK.TRANS64.TRYWAIT PT, [R4+URZ], RZ
    // @!PT NANOSLEEP.SYNCS 0x1234
    asm ("\n\t"
        "mbarrier.try_wait.parity.b64 complete, [%0], %1, %2;"
        : : "l"(&mbarrier), "n"(0), "n"(0x1234) : "memory"
    );
    asm("barrier.sync 3;");
    // SYNCS.PHASECHK.TRANS64.TRYWAIT PT, [R5+URZ], R3
    // @!PT NANOSLEEP.SYNCS 0x4321
    // @!PT SYNCS.PHASECHK.TRANS64 PT, [R5+URZ], R3 ;
    asm ("\n\t"
        "mbarrier.try_wait.parity.b64 complete, [%0], %1, %2;"
        : : "l"(&mbarrier), "n"(1), "n"(0x4321) : "memory"
    );
    __syncthreads();

    state = 0;
    asm("barrier.sync 4;");
    // SYNCS.PHASECHK.TRANS64 PT, [UR4], RZ
    asm ("\n\t"
        ".reg .pred complete2;\n\t"
        "mbarrier.test_wait.b64 complete2, [%0], %1;"
        : : "l"(&mbarrier), "l"(state) : "memory"
    );
    asm("barrier.sync 5;");
    // SYNCS.PHASECHK.TRANS64.TRYWAIT PT, [UR4], RZ
    asm ("\n\t"
        "mbarrier.try_wait.b64 complete2, [%0], %1;"
        : : "l"(&mbarrier), "l"(state) : "memory"
    );
    asm("barrier.sync 6;");
    // SYNCS.PHASECHK.TRANS64.TRYWAIT PT, [R6+URZ], RZ
    // @!PT NANOSLEEP.SYNCS 0x1234
    // @!PT SYNCS.PHASECHK.TRANS64 PT, [R6+URZ], RZ ;
    asm ("\n\t"
        "mbarrier.try_wait.b64 complete2, [%0], %1, %2;"
        : : "l"(&mbarrier), "l"(state), "n"(0x1234) : "memory"
    );
#endif
}

int main(int argc, char *argv[]) {
    printf("This is a test program mean to compare the mbarrier PTX and SASS mapping, it is not tested at all for functionality/run to finished\n");
    // uint64_t *sink;
    // cudaMalloc(&sink, sizeof(uint64_t));
    // CUDA_SAFECALL((test_mbarrier_kernel<<<1, 1>>>(sink)));
    // CUDA_SAFECALL(cudaDeviceSynchronize());

    // printf("Mbarrier test completed\n");
    return 0;
}
