/**
 * UBLKCP_S_G Unit Test: TMA Bulk Copy Global -> Shared
 * =====================================================
 * Tests that cp.async.bulk correctly loads data from global to shared memory.
 *
 * Strategy:
 *   1. Source global (d_src) initialized with pattern (0, 1, 2, ...)
 *   2. Destination global (d_dst) poisoned with 0xDEADBEEF
 *   3. TMA loads d_src -> shared memory
 *   4. Scalar (trusted) copy: shared -> d_dst
 *   PASS: d_dst matches d_src
 *   FAIL: d_dst still contains poison (TMA load was a NOP)
 *
 * Usage: ./ublkcp_s_g [-n <elements>] [-i <iterations>]
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

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace ptx = cuda::ptx;

#if defined(__CUDA_MINIMUM_ARCH__) && __CUDA_MINIMUM_ARCH__ < 900
static_assert(false, "Requires sm_90 or newer for TMA bulk operations.");
#endif

static constexpr size_t buf_len = 1024;
static constexpr int32_t POISON = (int32_t)0xDEADBEEF;

// ============================================================================
// Kernel
// ============================================================================
__global__ void test_UBLKCP_S_G(int32_t *d_src, int32_t *d_dst, int run_iters)
{
    __shared__ alignas(16) int32_t smem_data[buf_len];

    size_t offset = blockIdx.x * blockDim.x;

    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier bar;
    if (threadIdx.x == 0) {
        init(&bar, blockDim.x);
        ptx::fence_proxy_async(ptx::space_shared);
    }
    __syncthreads();

    // TMA bulk load: global -> shared
    for (int iter = 0; iter < run_iters; iter++) {
        if (threadIdx.x == 0) {
            cuda::memcpy_async(
                smem_data,
                d_src + offset,
                cuda::aligned_size_t<16>(sizeof(smem_data)),
                bar);
        }
        barrier::arrival_token token = bar.arrive();
        bar.wait(std::move(token));
    }

    // Trusted scalar copy: shared -> d_dst (does NOT use TMA)
    __syncthreads();
    for (int i = threadIdx.x; i < buf_len; i += blockDim.x) {
        d_dst[offset + i] = smem_data[i];
    }
}

// ============================================================================
// Verification
// ============================================================================
static bool verify(int32_t *h_src, int32_t *h_dst, int n)
{
    int poison_count = 0;
    int mismatch_count = 0;

    for (int i = 0; i < n; i++) {
        if (h_dst[i] == POISON) {
            if (poison_count == 0)
                printf("  NOP detected at index %d: got 0x%08X, expected %d\n",
                       i, (unsigned)h_dst[i], h_src[i]);
            poison_count++;
        } else if (h_dst[i] != h_src[i]) {
            if (mismatch_count == 0)
                printf("  Mismatch at index %d: got %d, expected %d\n",
                       i, h_dst[i], h_src[i]);
            mismatch_count++;
        }
    }

    if (poison_count > 0) {
        printf("  FAIL: %d/%d elements still contain poison (TMA load was NOP)\n",
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

    // Host buffers
    int32_t *h_src = (int32_t *)malloc(bytes);
    int32_t *h_dst = (int32_t *)malloc(bytes);

    // Initialize source with pattern, destination with poison
    for (int i = 0; i < n; i++) {
        h_src[i] = i;
        h_dst[i] = POISON;
    }

    // Device buffers
    int32_t *d_src, *d_dst;
    cudaMalloc(&d_src, bytes);
    cudaMalloc(&d_dst, bytes);
    cudaMemcpy(d_src, h_src, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst, h_dst, bytes, cudaMemcpyHostToDevice);

    // Run
    CUDA_SAFECALL((test_UBLKCP_S_G<<<gridSize, blockSize>>>(d_src, d_dst, run_iters)));
    cudaDeviceSynchronize();

    // Copy back
    cudaMemcpy(h_dst, d_dst, bytes, cudaMemcpyDeviceToHost);

    // Verify
    printf("=== UBLKCP_S_G (TMA Bulk Load: Global -> Shared) ===\n");
    bool pass = verify(h_src, h_dst, n);
    printf("RESULT: %s\n", pass ? "PASS" : "FAIL");

    free(h_src);
    free(h_dst);
    cudaFree(d_src);
    cudaFree(d_dst);
    return pass ? 0 : 1;
}
