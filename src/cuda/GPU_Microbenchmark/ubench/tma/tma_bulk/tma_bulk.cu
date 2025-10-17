#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda/barrier>
#include <cuda/ptx>

/*
 * Test application for TMA bulk operations.
 * 
 * Usage: ./tma_bulk -n <n> -o <opcode>
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

// Adapt from https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=tma#using-tma-to-transfer-one-dimensional-arrays
using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace ptx = cuda::ptx;

#if defined(__CUDA_MINIMUM_ARCH__) && __CUDA_MINIMUM_ARCH__ < 900
static_assert(false, "Device code is being compiled with older architectures that are incompatible with TMA.");
#endif // __CUDA_MINIMUM_ARCH__

static constexpr size_t buf_len = 1024;
#define DEFAULT_RUN_ITERS 128

__global__ void test_UBLKPF(int32_t *data, int run_iters)
{
    size_t offset = blockIdx.x * blockDim.x;

    // Trigger a bulk prefetch
    uint64_t prefetch_addr = uint64_t(data + offset);
    uint32_t prefetch_count = buf_len * sizeof(int32_t);
    if (threadIdx.x == 0)
    {
        asm volatile(
            "cp.async.bulk.prefetch.L2.global"
            " [%0], %1;"
            :
            : "l"(prefetch_addr),
              "r"(prefetch_count)
            : "memory");
        ptx::cp_async_bulk_commit_group();
        ptx::cp_async_bulk_wait_group_read(ptx::n32_t<0>());
    }
}

__global__ void test_UBLKCP_S_G(int32_t *data, int run_iters)
{
    // Shared memory buffer. The destination shared memory buffer of
    // a bulk operations should be 16 byte aligned.
    __shared__ alignas(16) int32_t smem_data[buf_len];

    size_t offset = blockIdx.x * blockDim.x;
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier bar;
    if (threadIdx.x == 0) {
        init(&bar, blockDim.x);
        ptx::fence_proxy_async(ptx::space_shared);
    }
    __syncthreads();

    for (int i = 0; i < run_iters; i++) {
        // Initiate TMA transfer to copy global to shared memory.
        if (threadIdx.x == 0)
        {
            cuda::memcpy_async(
                smem_data,
                data + offset,
                cuda::aligned_size_t<16>(sizeof(smem_data)),
                bar);
        }
        barrier::arrival_token token = bar.arrive();
        bar.wait(std::move(token));
    }
}

__global__ void test_UBLKCP_G_S(int32_t *data, int run_iters)
{
    // Shared memory buffer. The destination shared memory buffer of
    // a bulk operations should be 16 byte aligned.
    __shared__ alignas(16) int32_t smem_data[buf_len];

    size_t offset = blockIdx.x * blockDim.x;

    // Compute a unique value for each thread across all blocks
    for (int i = threadIdx.x; i < buf_len; i += blockDim.x)
    {
        smem_data[i] = threadIdx.x + blockIdx.x * blockDim.x;
    }

    ptx::fence_proxy_async(ptx::space_shared); // b)
    __syncthreads();

    for (int i = 0; i < run_iters; i++) {
        if (threadIdx.x == 0) {
            ptx::cp_async_bulk(
                ptx::space_global,
                ptx::space_shared,
                data + offset, smem_data, sizeof(smem_data));
                ptx::cp_async_bulk_commit_group();
                ptx::cp_async_bulk_wait_group_read(ptx::n32_t<0>());
        }
    }
}

__global__ void test_UBLKRED_G_S(int32_t *data, int run_iters)
{
    // Shared memory buffer. The destination shared memory buffer of
    // a bulk operations should be 16 byte aligned.
    __shared__ alignas(16) int32_t smem_data[buf_len];

    size_t offset = blockIdx.x * blockDim.x;

    // Compute a unique value for each thread across all blocks
    for (int i = threadIdx.x; i < buf_len; i += blockDim.x)
    {
        smem_data[i] = threadIdx.x + blockIdx.x * blockDim.x;
    }

    ptx::fence_proxy_async(ptx::space_shared); // b)
    __syncthreads();

    for (int i = 0; i < run_iters; i++) {
        if (threadIdx.x == 0)
        {   
            // Use max so the result wont change compared with
            // before the reduction, as TMA can only get source values
            ptx::cp_reduce_async_bulk(
                ptx::space_global,
                ptx::space_shared,
                ptx::op_max,
                data + offset, smem_data, sizeof(smem_data));
            ptx::cp_async_bulk_commit_group();
            ptx::cp_async_bulk_wait_group_read(ptx::n32_t<0>());
        }
    }
}

int main(int argc, char *argv[])
{
    // Parse command line arguments
    int n = 1024 * 16;
    const char* opcode = "UBLKPF";
    int opt;
    int run_iters = DEFAULT_RUN_ITERS;
    while ((opt = getopt(argc, argv, "n:o:i:")) != -1) {
        switch (opt) {
            case 'n':
                n = atoi(optarg);
                break;
            case 'o':
                opcode = strdup(optarg);
                break;
            case 'i':
                run_iters = atoi(optarg);
                break;
            default:
                fprintf(stderr, "Usage: %s -n <n> -o <opcode>\n", argv[0]);
                fprintf(stderr, "  -n <n>: number of elements\n");
                fprintf(stderr, "  -o <opcode>: opcode\n");
                fprintf(stderr, "  -o UBLKPF: prefetch\n");
                fprintf(stderr, "  -o UBLKCP_S_G: bulk copy shared to global\n");
                fprintf(stderr, "  -o UBLKCP_G_S: bulk copy global to shared\n");
                fprintf(stderr, "  -o UBLKRED_G_S: bulk reduce global to shared\n");
                fprintf(stderr, "  -i <run_iters>: number of iterations\n");
                return 1;
        }
    }

    // Check if opcode is valid
    if (strcmp(opcode, "UBLKPF") != 0 &&
        strcmp(opcode, "UBLKCP_S_G") != 0 &&
        strcmp(opcode, "UBLKCP_G_S") != 0 &&
        strcmp(opcode, "UBLKRED_G_S") != 0) {
        fprintf(stderr, "Invalid opcode\n");
        return 1;
    }

    // Host input vectors
    int32_t *h_a;

    // Host destination vectors
    int32_t *h_b;

    // Device input vectors
    int32_t *d_a;

    // Size, in bytes, of each vector
    size_t bytes = n * sizeof(int32_t);

    // Allocate memory for each vector on host
    h_a = (int32_t *)malloc(bytes);
    h_b = (int32_t *)malloc(bytes);
    // Allocate memory for each vector on GPU
    cudaMalloc(&d_a, bytes);

    uint32_t i;
    // Initialize vectors on host with unique values
    for (i = 0; i < n; i++)
    {
        h_a[i] = i;
    }

    // Copy host vectors to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);

    uint32_t blockSize, gridSize;

    // Number of threads in each thread block
    blockSize = 1024;

    // Number of thread blocks in grid
    gridSize = (uint32_t)ceil((float)n / blockSize);

    // Execute the kernel based on the opcode
    if (strcmp(opcode, "UBLKPF") == 0) {
        CUDA_SAFECALL((test_UBLKPF<<<gridSize, blockSize>>>(d_a, run_iters)));
    } else if (strcmp(opcode, "UBLKCP_S_G") == 0) {
        CUDA_SAFECALL((test_UBLKCP_S_G<<<gridSize, blockSize>>>(d_a, run_iters)));
    } else if (strcmp(opcode, "UBLKCP_G_S") == 0) {
        CUDA_SAFECALL((test_UBLKCP_G_S<<<gridSize, blockSize>>>(d_a, run_iters)));
    } else if (strcmp(opcode, "UBLKRED_G_S") == 0) {
        CUDA_SAFECALL((test_UBLKRED_G_S<<<gridSize, blockSize>>>(d_a, run_iters)));
    }

    // Copy array back to host
    cudaMemcpy(h_b, d_a, bytes, cudaMemcpyDeviceToHost);

    // Dump the values to a file in hex format
    // For load operations, use host source array
    int32_t *ptr = h_a;

    // For store operations, use destination array
    if (strcmp(opcode, "UBLKCP_G_S") == 0 || 
        strcmp(opcode, "UBLKRED_G_S") == 0) {
        ptr = h_b;
    }

    char filename[100];
    sprintf(filename, "tma_bulk_test_%s_%d.txt", opcode, n);
    FILE *f = fopen(filename, "w");
    for (i = 0; i < n; i++)
    {
        fprintf(f, "0x%x ", ptr[i]);
        // Add line break after every 512 values
        if ((i + 1) % 512 == 0)
            fprintf(f, "\n");
    }
    fclose(f);
    printf("Values dumped to %s\n", filename);

    // Release host memory
    free(h_a);

    // Release device memory
    cudaFree(d_a);

    return 0;
}
