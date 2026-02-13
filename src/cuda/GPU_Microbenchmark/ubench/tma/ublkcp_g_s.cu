/**
 * UBLKCP_G_S Unit Test: TMA Bulk Copy Shared -> Global
 * =====================================================
 * Tests that cp.async.bulk correctly stores data from shared to global memory.
 *
 * Strategy:
 *   1. Global memory is poisoned with 0xDEADBEEF
 *   2. Kernel generates pattern (0, 1, 2, ...) in shared memory
 *   3. TMA stores shared -> global
 *   PASS: Global contains (0, 1, 2, ...)
 *   FAIL: Global still contains poison (TMA store was a NOP)
 *
 * Usage: ./ublkcp_g_s [-n <elements>] [-i <iterations>]
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <unistd.h>
#include <cuda/barrier>
#include <cuda/ptx>

#define CUDA_SAFECALL(call)                                                 \
    {                                                                       \
        call;                                                               \
        cudaError err = cudaGetLastError();                                 \
        if (cudaSuccess != err) {                                           \
            fprintf(stderr,                                                 \
                "Cuda error in '%s' at %s:%d : %s.\n",                     \
                #call, __FILE__, __LINE__, cudaGetErrorString(err));        \
            fflush(stderr);                                                 \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    }

namespace ptx = cuda::ptx;

#if defined(__CUDA_MINIMUM_ARCH__) && __CUDA_MINIMUM_ARCH__ < 900
static_assert(false, "Requires sm_90 or newer for TMA bulk operations.");
#endif

static constexpr size_t buf_len = 1024;
static constexpr int32_t POISON = (int32_t)0xDEADBEEF;

// ============================================================================
// Kernel
// ============================================================================
__global__ void test_UBLKCP_G_S(int32_t *data, int run_iters)
{
    __shared__ alignas(16) int32_t smem_data[buf_len];

    size_t offset = blockIdx.x * blockDim.x;

    // Generate the test pattern (0, 1, 2, ...) in shared memory
    for (int i = threadIdx.x; i < buf_len; i += blockDim.x) {
        smem_data[i] = offset + i;
    }

    ptx::fence_proxy_async(ptx::space_shared);
    __syncthreads();

    // TMA bulk store: shared -> global
    for (int iter = 0; iter < run_iters; iter++) {
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

// ============================================================================
// Verification
// ============================================================================
static bool verify(int32_t *h_result, int n)
{
    int poison_count = 0;
    int mismatch_count = 0;

    for (int i = 0; i < n; i++) {
        if (h_result[i] == POISON) {
            if (poison_count == 0)
                printf("  NOP detected at index %d: got 0x%08X, expected %d\n",
                       i, (unsigned)h_result[i], i);
            poison_count++;
        } else if (h_result[i] != i) {
            if (mismatch_count == 0)
                printf("  Mismatch at index %d: got %d, expected %d\n",
                       i, h_result[i], i);
            mismatch_count++;
        }
    }

    if (poison_count > 0) {
        printf("  FAIL: %d/%d elements still contain poison (TMA store was NOP)\n",
               poison_count, n);
        return false;
    }
    if (mismatch_count > 0) {
        printf("  FAIL: %d/%d elements have wrong values\n", mismatch_count, n);
        return false;
    }
    return true;
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char *argv[])
{
    int n = 1024;
    int run_iters = 128;
    int opt;
    while ((opt = getopt(argc, argv, "n:i:")) != -1) {
        switch (opt) {
            case 'n': n = atoi(optarg); break;
            case 'i': run_iters = atoi(optarg); break;
            default:
                fprintf(stderr, "Usage: %s [-n <elements>] [-i <iterations>]\n", argv[0]);
                return 1;
        }
    }

    size_t bytes = n * sizeof(int32_t);
    uint32_t blockSize = 1024;
    uint32_t gridSize = (uint32_t)ceil((float)n / blockSize);

    // Host buffer — poisoned
    int32_t *h_result = (int32_t *)malloc(bytes);
    for (int i = 0; i < n; i++) {
        h_result[i] = POISON;
    }

    // Device buffer
    int32_t *d_a;
    cudaMalloc(&d_a, bytes);
    cudaMemcpy(d_a, h_result, bytes, cudaMemcpyHostToDevice);

    // Run
    CUDA_SAFECALL((test_UBLKCP_G_S<<<gridSize, blockSize>>>(d_a, run_iters)));
    cudaDeviceSynchronize();

    // Copy back
    cudaMemcpy(h_result, d_a, bytes, cudaMemcpyDeviceToHost);

    // Verify
    printf("=== UBLKCP_G_S (TMA Bulk Store: Shared -> Global) ===\n");
    bool pass = verify(h_result, n);
    printf("RESULT: %s\n", pass ? "PASS" : "FAIL");

    free(h_result);
    cudaFree(d_a);
    return pass ? 0 : 1;
}
