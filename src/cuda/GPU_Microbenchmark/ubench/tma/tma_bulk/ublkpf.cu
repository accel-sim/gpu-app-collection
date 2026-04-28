/**
 * UBLKPF Unit Test: TMA Bulk Prefetch (cp.async.bulk.prefetch.L2.global)
 * ======================================================================
 * Tests that cp.async.bulk.prefetch.L2.global correctly issues a prefetch
 * without corrupting memory contents.
 *
 * The PTX ISA only exposes one variant of 1D bulk prefetch:
 *   cp.async.bulk.prefetch.L2.global [srcMem], size;
 *
 * Verification Strategy:
 *   1. Global memory is initialized with a known pattern (0, 1, 2, ...)
 *   2. Kernel issues cp.async.bulk.prefetch.L2.global for each block's region
 *   3. After the prefetch, kernel reads the data back via standard loads
 *      and writes to a destination buffer
 *   4. Host verifies destination matches original source pattern
 *
 *   PASS: Prefetch executed without error, data integrity maintained
 *   FAIL: Data is corrupted after prefetch (should never happen on correct HW)
 *
 * Usage: ./ublkpf [-n <elements>] [-i <iterations>] [-s <prefetch_bytes>]
 *
 *   -n <elements>       : number of int32 elements (default: 1024)
 *   -i <iterations>     : number of prefetch iterations in-kernel (default: 1)
 *   -s <prefetch_bytes> : bytes to prefetch per block (default: buf_len * 4)
 *                         clamped to a multiple of 16, max buf_len * 4
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <unistd.h>
#include <cuda/ptx>

#define CUDA_SAFECALL(call)                                                 \
    {                                                                       \
        call;                                                               \
        cudaError err = cudaGetLastError();                                 \
        if (cudaSuccess != err) {                                           \
            fprintf(stderr,                                                 \
                "Cuda error in '%s' at %s:%d : %s.\n",                      \
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
// Kernel: prefetch + readback verification
//
// Thread 0 issues the bulk prefetch, then all threads cooperatively
// read the (now hopefully L2-resident) data via standard loads and
// write it to a separate destination buffer.  If the prefetch corrupted
// memory or caused any fault, the readback will expose it.
// ============================================================================
__global__ void test_UBLKPF(const int32_t *d_src, int32_t *d_dst,
                            int run_iters, uint32_t prefetch_bytes)
{
    size_t offset = blockIdx.x * blockDim.x;

    // Prefetch phase
    if (threadIdx.x == 0) {
        uint64_t addr = (uint64_t)(d_src + offset);
        for (int iter = 0; iter < run_iters; iter++) {
            asm volatile(
                "cp.async.bulk.prefetch.L2.global"
                " [%0], %1;"
                :
                : "l"(addr),
                  "r"(prefetch_bytes)
                : "memory");
            ptx::cp_async_bulk_commit_group();
            ptx::cp_async_bulk_wait_group_read(ptx::n32_t<0>());
        }
    }

    __syncthreads();

    // Readback phase: standard loads to verify data integrity
    uint32_t elems = prefetch_bytes / sizeof(int32_t);
    for (uint32_t i = threadIdx.x; i < elems; i += blockDim.x) {
        d_dst[offset + i] = d_src[offset + i];
    }
}

// ============================================================================
// Verification
// ============================================================================
static bool verify(const int32_t *h_src, const int32_t *h_dst, int n)
{
    int poison_count = 0;
    int mismatch_count = 0;

    for (int i = 0; i < n; i++) {
        if (h_dst[i] == POISON) {
            if (poison_count == 0)
                printf("  Poison still present at index %d: got 0x%08X, expected %d\n",
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
        printf("  FAIL: %d/%d elements still contain poison (readback did not execute)\n",
               poison_count, n);
        return false;
    }
    if (mismatch_count > 0) {
        printf("  FAIL: %d/%d elements have wrong values (data corrupted after prefetch)\n",
               mismatch_count, n);
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
    int run_iters = 1;
    int user_prefetch_bytes = -1;

    int opt;
    while ((opt = getopt(argc, argv, "n:i:s:")) != -1) {
        switch (opt) {
            case 'n':
                n = atoi(optarg);
                break;
            case 'i':
                run_iters = atoi(optarg);
                break;
            case 's':
                user_prefetch_bytes = atoi(optarg);
                break;
            default:
                fprintf(stderr,
                        "Usage: %s [-n <elements>] [-i <iterations>] [-s <prefetch_bytes>]\n",
                        argv[0]);
                fprintf(stderr, "  -n <elements>       : number of int32 elements (default: 1024)\n");
                fprintf(stderr, "  -i <iterations>     : number of prefetch iterations (default: 1)\n");
                fprintf(stderr, "  -s <prefetch_bytes> : bytes to prefetch per block (default: %zu)\n",
                        buf_len * sizeof(int32_t));
                fprintf(stderr, "                        must be a multiple of 16, max %zu\n",
                        buf_len * sizeof(int32_t));
                return 1;
        }
    }

    uint32_t max_prefetch = (uint32_t)(buf_len * sizeof(int32_t));
    uint32_t prefetch_bytes;
    if (user_prefetch_bytes < 0) {
        prefetch_bytes = max_prefetch;
    } else {
        prefetch_bytes = (uint32_t)user_prefetch_bytes;
        prefetch_bytes = (prefetch_bytes + 15) & ~15u;
        if (prefetch_bytes > max_prefetch)
            prefetch_bytes = max_prefetch;
        if (prefetch_bytes == 0)
            prefetch_bytes = 16;
    }

    size_t bytes = (size_t)n * sizeof(int32_t);
    uint32_t blockSize = 1024;
    uint32_t gridSize = (uint32_t)ceil((float)n / blockSize);

    // Host buffers
    int32_t *h_src = (int32_t *)malloc(bytes);
    int32_t *h_dst = (int32_t *)malloc(bytes);

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
    CUDA_SAFECALL((test_UBLKPF<<<gridSize, blockSize>>>(d_src, d_dst, run_iters, prefetch_bytes)));
    cudaDeviceSynchronize();

    // Copy back
    cudaMemcpy(h_dst, d_dst, bytes, cudaMemcpyDeviceToHost);

    // Verify
    uint32_t verified_elems = prefetch_bytes / sizeof(int32_t);
    if ((int)verified_elems > n) verified_elems = n;

    printf("=== UBLKPF (TMA Bulk Prefetch: cp.async.bulk.prefetch.L2.global) ===\n");
    printf("  Prefetch size: %u bytes (%u int32 elements)\n",
           prefetch_bytes, prefetch_bytes / 4);
    printf("  Iterations:    %d\n", run_iters);

    bool pass = verify(h_src, h_dst, (int)verified_elems);
    printf("RESULT: %s\n", pass ? "PASS" : "FAIL");

    // Dump results to file for easy expected-vs-actual comparison
    char filename[128];
    sprintf(filename, "tma_bulk_test_UBLKPF_%d.txt", n);
    FILE *f = fopen(filename, "w");
    if (f) {
        fprintf(f, "# UBLKPF Test Results (TMA Bulk Prefetch + Readback)\n");
        fprintf(f, "# Prefetch size: %u bytes (%u int32 elements)\n",
                prefetch_bytes, prefetch_bytes / 4);
        fprintf(f, "# Iterations: %d\n", run_iters);
        fprintf(f, "# Format: index, got (hex), expected (hex)\n");
        for (int i = 0; i < (int)verified_elems; i++) {
            const char *status = (h_dst[i] == h_src[i]) ? "OK" :
                                 (h_dst[i] == POISON)   ? "POISON" : "MISMATCH";
            fprintf(f, "%4d: 0x%08X  (expected 0x%08X) [%s]\n",
                    i, (unsigned)h_dst[i], (unsigned)h_src[i], status);
            if ((i + 1) % 512 == 0) fprintf(f, "\n");
        }
        fclose(f);
        printf("Values dumped to %s\n", filename);
    } else {
        fprintf(stderr, "Warning: could not open %s for writing\n", filename);
    }

    free(h_src);
    free(h_dst);
    cudaFree(d_src);
    cudaFree(d_dst);
    return pass ? 0 : 1;
}
