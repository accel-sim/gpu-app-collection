/**
 * UTMAPF Unit Test: TMA Tensor Prefetch (L2 hint)
 * ================================================
 * Tests that cp.async.bulk.prefetch.tensor issues without error and does not
 * corrupt data. The prefetch is a performance hint that warms the L2 cache;
 * correctness is verified by a subsequent scalar readback.
 *
 * Strategy:
 *   1. Source matrix (d_mat) initialized with unique values (1, 2, 3, ...)
 *   2. Destination buffer (d_dst) poisoned with 0xDEADBEEF
 *   3. Kernel: thread (0,0) issues TMA tensor prefetch for this block's tile
 *   4. Scalar readback: each thread copies d_mat[gy*stride+gx] -> d_dst
 *   PASS: d_dst matches original source (prefetch did not corrupt data)
 *   FAIL_NOP: d_dst still contains poison (readback did not execute)
 *   FAIL_MISMATCH: d_dst has wrong values (unexpected corruption)
 *
 * Usage: ./utmapf [-w <width>] [-h <height>] [-i <iterations>]
 */

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <unistd.h>
#include <cudaTypedefs.h>
#include <cuda.h>
#include <cuda/ptx>

#define CUDA_SAFECALL(call) \
    { \
        call; \
        cudaError err = cudaGetLastError(); \
        if (cudaSuccess != err) { \
            fprintf(stderr, "Cuda error in '%s' at %s:%d : %s.\n", \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            fflush(stderr); exit(EXIT_FAILURE); \
        } \
    }

namespace ptx = cuda::ptx;

#if defined(__CUDA_MINIMUM_ARCH__) && __CUDA_MINIMUM_ARCH__ < 900
static_assert(false, "Requires sm_90a or newer for TMA tensor operations.");
#endif

static constexpr uint32_t SMEM_W = 32;
static constexpr uint32_t SMEM_H = 32;
static constexpr int32_t  POISON = (int32_t)0xDEADBEEF;

// ============================================================================
// Kernel
// ============================================================================
__global__ void test_UTMAPF(const __grid_constant__ CUtensorMap tensor_map,
                             const int32_t *d_mat, int32_t *d_dst,
                             uint32_t width_stride, int run_iters)
{
    int x = (int)(blockDim.x * blockIdx.x);
    int y = (int)(blockDim.y * blockIdx.y);

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        for (int i = 0; i < run_iters; i++) {
            asm volatile(
                "cp.async.bulk.prefetch.tensor.2d.L2.global.tile"
                " [%0, {%1, %2}];"
                : : "l"(&tensor_map), "r"(x), "r"(y) : "memory");
            ptx::cp_async_bulk_commit_group();
            ptx::cp_async_bulk_wait_group_read(ptx::n32_t<0>());
        }
    }
    __syncthreads();

    // Scalar readback: verifies prefetch didn't corrupt the source
    uint32_t gx = (uint32_t)x + threadIdx.x;
    uint32_t gy = (uint32_t)y + threadIdx.y;
    d_dst[gy * width_stride + gx] = d_mat[gy * width_stride + gx];
}

// ============================================================================
// Helpers
// ============================================================================
static PFN_cuTensorMapEncodeTiled_v12000 get_cuTensorMapEncodeTiled()
{
    cudaDriverEntryPointQueryResult status;
    void *fn = nullptr;
    CUDA_SAFECALL(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled",
                  &fn, 12000, cudaEnableDefault, &status));
    assert(status == cudaDriverEntryPointSuccess);
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(fn);
}

// ============================================================================
// Verification
// ============================================================================
static bool verify(const int32_t *h_dst, const int32_t *h_src,
                   uint64_t height, uint64_t width,
                   uint64_t height_stride, uint64_t width_stride)
{
    int poison_count = 0, mismatch_count = 0;
    for (uint64_t r = 0; r < height; r++) {
        for (uint64_t c = 0; c < width; c++) {
            int32_t got = h_dst[r * width_stride + c];
            int32_t exp = h_src[r * width_stride + c];
            if (got == POISON) {
                if (poison_count == 0)
                    printf("  NOP at [%lu][%lu]: got 0x%08X, expected %d\n",
                           r, c, (unsigned)got, exp);
                poison_count++;
            } else if (got != exp) {
                if (mismatch_count == 0)
                    printf("  Mismatch at [%lu][%lu]: got %d, expected %d\n",
                           r, c, got, exp);
                mismatch_count++;
            }
        }
    }
    if (poison_count > 0) {
        printf("  FAIL: %d/%lu elements still contain poison\n",
               poison_count, height * width);
        return false;
    }
    if (mismatch_count > 0) {
        printf("  FAIL: %d/%lu elements have wrong values\n",
               mismatch_count, height * width);
        return false;
    }
    return true;
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char *argv[])
{
    uint64_t width = 64, height = 64;
    int run_iters = 1;
    int opt;
    while ((opt = getopt(argc, argv, "w:h:i:")) != -1) {
        switch (opt) {
            case 'w': width  = (uint64_t)atoi(optarg); break;
            case 'h': height = (uint64_t)atoi(optarg); break;
            case 'i': run_iters = atoi(optarg); break;
            default:
                fprintf(stderr, "Usage: %s [-w <width>] [-h <height>] [-i <iters>]\n",
                        argv[0]);
                return 1;
        }
    }

    uint64_t width_stride  = ((width  + SMEM_W - 1) / SMEM_W) * SMEM_W;
    uint64_t height_stride = ((height + SMEM_H - 1) / SMEM_H) * SMEM_H;
    size_t bytes = height_stride * width_stride * sizeof(int32_t);

    int32_t *h_src = (int32_t *)malloc(bytes);
    int32_t *h_dst = (int32_t *)malloc(bytes);

    // Source: unique values in-bounds, 0 for padding
    int32_t val = 1;
    for (uint64_t r = 0; r < height_stride; r++)
        for (uint64_t c = 0; c < width_stride; c++)
            h_src[r * width_stride + c] = (r < height && c < width) ? val++ : 0;

    // Destination: poisoned
    for (size_t i = 0; i < height_stride * width_stride; i++)
        h_dst[i] = POISON;

    int32_t *d_mat, *d_dst;
    cudaMalloc(&d_mat, bytes);
    cudaMalloc(&d_dst, bytes);
    cudaMemcpy(d_mat, h_src, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst, h_dst, bytes, cudaMemcpyHostToDevice);

    // Build tensor descriptor (needed to issue prefetch; descriptor encodes the tensor layout)
    CUtensorMap tensor_map{};
    constexpr uint32_t rank = 2;
    uint64_t size[rank]        = {width, height};
    uint64_t stride[1]         = {width_stride * sizeof(int32_t)};
    uint32_t box_size[rank]    = {SMEM_W, SMEM_H};
    uint32_t elem_stride[rank] = {1, 1};
    auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();
    cuTensorMapEncodeTiled(
        &tensor_map, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_INT32,
        rank, d_mat, size, stride, box_size, elem_stride,
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    dim3 grid(width_stride / SMEM_W, height_stride / SMEM_H);
    dim3 block(SMEM_W, SMEM_H);
    CUDA_SAFECALL((test_UTMAPF<<<grid, block>>>(
        tensor_map, d_mat, d_dst, (uint32_t)width_stride, run_iters)));
    cudaMemcpy(h_dst, d_dst, bytes, cudaMemcpyDeviceToHost);

    printf("=== UTMAPF (TMA Tensor Prefetch + Scalar Readback) ===\n");
    bool pass = verify(h_dst, h_src, height, width, height_stride, width_stride);
    printf("RESULT: %s\n", pass ? "PASS" : "FAIL");

    free(h_src); free(h_dst);
    cudaFree(d_mat); cudaFree(d_dst);
    return pass ? 0 : 1;
}
