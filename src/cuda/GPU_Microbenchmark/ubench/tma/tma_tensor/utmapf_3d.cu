/**
 * UTMAPF_3D Unit Test: TMA Tensor Prefetch, rank-3 (L2 hint)
 * ============================================================
 * Tests cp.async.bulk.prefetch.tensor.3d.L2.global.tile with a rank-3
 * CUtensorMap descriptor. Prefetch is a performance hint; correctness is
 * verified by a subsequent scalar readback.
 *
 * Strategy:
 *   1. Source tensor (d_mat) initialized with unique values (1, 2, 3, ...)
 *   2. Destination buffer (d_dst) poisoned with 0xDEADBEEF
 *   3. Thread (0,0,0) issues 3D TMA prefetch for this block's tile
 *   4. Scalar readback: each thread copies d_mat -> d_dst
 *   PASS: d_dst matches original source (no corruption)
 *   FAIL_NOP: d_dst still contains poison (readback did not execute)
 *
 * Usage: ./utmapf_3d [-W <width>] [-H <height>] [-D <depth>] [-i <iters>]
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

static constexpr uint32_t TILE_W = 32;
static constexpr uint32_t TILE_H = 4;
static constexpr uint32_t TILE_D = 2;
static constexpr int32_t  POISON = (int32_t)0xDEADBEEF;

// ============================================================================
// Kernel
// ============================================================================
__global__ void test_UTMAPF_3D(const __grid_constant__ CUtensorMap tensor_map,
                                const int32_t *d_mat, int32_t *d_dst,
                                uint32_t w_stride, uint32_t h_stride,
                                int run_iters)
{
    int x = (int)(blockDim.x * blockIdx.x);
    int y = (int)(blockDim.y * blockIdx.y);
    int z = (int)(blockDim.z * blockIdx.z);

    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        for (int i = 0; i < run_iters; i++) {
            asm volatile(
                "cp.async.bulk.prefetch.tensor.3d.L2.global.tile"
                " [%0, {%1, %2, %3}];"
                : : "l"(&tensor_map), "r"(x), "r"(y), "r"(z) : "memory");
            ptx::cp_async_bulk_commit_group();
            ptx::cp_async_bulk_wait_group_read(ptx::n32_t<0>());
        }
    }
    __syncthreads();

    // Scalar readback: verify prefetch didn't corrupt the source
    uint32_t gx = (uint32_t)x + threadIdx.x;
    uint32_t gy = (uint32_t)y + threadIdx.y;
    uint32_t gz = (uint32_t)z + threadIdx.z;
    uint32_t idx = gz * h_stride * w_stride + gy * w_stride + gx;
    d_dst[idx] = d_mat[idx];
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
                   uint64_t depth, uint64_t height, uint64_t width,
                   uint64_t h_stride, uint64_t w_stride)
{
    int poison_count = 0, mismatch_count = 0;
    for (uint64_t z = 0; z < depth; z++) {
        for (uint64_t y = 0; y < height; y++) {
            for (uint64_t x = 0; x < width; x++) {
                uint64_t idx = z * h_stride * w_stride + y * w_stride + x;
                int32_t got = h_dst[idx];
                int32_t exp = h_src[idx];
                if (got == POISON) {
                    if (poison_count == 0)
                        printf("  NOP at [%lu][%lu][%lu]: got 0x%08X, expected %d\n",
                               z, y, x, (unsigned)got, exp);
                    poison_count++;
                } else if (got != exp) {
                    if (mismatch_count == 0)
                        printf("  Mismatch at [%lu][%lu][%lu]: got %d, expected %d\n",
                               z, y, x, got, exp);
                    mismatch_count++;
                }
            }
        }
    }
    if (poison_count > 0) {
        printf("  FAIL: %d/%lu elements still contain poison\n",
               poison_count, depth * height * width);
        return false;
    }
    if (mismatch_count > 0) {
        printf("  FAIL: %d/%lu elements have wrong values\n",
               mismatch_count, depth * height * width);
        return false;
    }
    return true;
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char *argv[])
{
    uint64_t W = 64, H = 8, D = 4;
    int run_iters = 1;
    int opt;
    while ((opt = getopt(argc, argv, "W:H:D:i:")) != -1) {
        switch (opt) {
            case 'W': W = (uint64_t)atoi(optarg); break;
            case 'H': H = (uint64_t)atoi(optarg); break;
            case 'D': D = (uint64_t)atoi(optarg); break;
            case 'i': run_iters = atoi(optarg); break;
            default:
                fprintf(stderr,
                        "Usage: %s [-W <width>] [-H <height>] [-D <depth>] [-i <iters>]\n",
                        argv[0]);
                return 1;
        }
    }

    uint64_t w_stride = ((W + TILE_W - 1) / TILE_W) * TILE_W;
    uint64_t h_stride = ((H + TILE_H - 1) / TILE_H) * TILE_H;
    uint64_t d_stride = ((D + TILE_D - 1) / TILE_D) * TILE_D;
    size_t bytes = d_stride * h_stride * w_stride * sizeof(int32_t);

    int32_t *h_src = (int32_t *)malloc(bytes);
    int32_t *h_dst = (int32_t *)malloc(bytes);

    int32_t val = 1;
    for (uint64_t z = 0; z < d_stride; z++)
        for (uint64_t y = 0; y < h_stride; y++)
            for (uint64_t x = 0; x < w_stride; x++) {
                uint64_t idx = z * h_stride * w_stride + y * w_stride + x;
                h_src[idx] = (z < D && y < H && x < W) ? val++ : 0;
            }

    for (size_t i = 0; i < d_stride * h_stride * w_stride; i++)
        h_dst[i] = POISON;

    int32_t *d_mat, *d_dst;
    cudaMalloc(&d_mat, bytes);
    cudaMalloc(&d_dst, bytes);
    cudaMemcpy(d_mat, h_src, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst, h_dst, bytes, cudaMemcpyHostToDevice);

    CUtensorMap tensor_map{};
    constexpr uint32_t rank = 3;
    uint64_t size[rank]        = {W, H, D};
    uint64_t stride[rank - 1]  = {w_stride * sizeof(int32_t),
                                   w_stride * h_stride * sizeof(int32_t)};
    uint32_t box_size[rank]    = {TILE_W, TILE_H, TILE_D};
    uint32_t elem_stride[rank] = {1, 1, 1};
    auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();
    cuTensorMapEncodeTiled(
        &tensor_map, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_INT32,
        rank, d_mat, size, stride, box_size, elem_stride,
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    dim3 grid(w_stride / TILE_W, h_stride / TILE_H, d_stride / TILE_D);
    dim3 block(TILE_W, TILE_H, TILE_D);
    CUDA_SAFECALL((test_UTMAPF_3D<<<grid, block>>>(
        tensor_map, d_mat, d_dst,
        (uint32_t)w_stride, (uint32_t)h_stride, run_iters)));
    cudaMemcpy(h_dst, d_dst, bytes, cudaMemcpyDeviceToHost);

    printf("=== UTMAPF_3D (TMA Tensor Prefetch rank-3 + Scalar Readback) ===\n");
    bool pass = verify(h_dst, h_src, D, H, W, h_stride, w_stride);
    printf("RESULT: %s\n", pass ? "PASSED" : "FAILED");

    free(h_src); free(h_dst);
    cudaFree(d_mat); cudaFree(d_dst);
    return pass ? 0 : 1;
}
