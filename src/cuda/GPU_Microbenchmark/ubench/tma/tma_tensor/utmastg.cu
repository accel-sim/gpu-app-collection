/**
 * UTMASTG Unit Test: TMA Tensor Store (Shared -> Global)
 * =======================================================
 * Tests that cp.async.bulk.tensor correctly stores a 2D tile from shared
 * memory into global memory using a CUtensorMap descriptor.
 *
 * Strategy:
 *   1. Global destination matrix (d_mat) poisoned with 0xDEADBEEF
 *   2. Kernel: each thread writes a unique pattern into shared memory
 *   3. Thread (0,0) issues TMA tensor store: shared -> global
 *   PASS: d_mat contains expected pattern for in-bounds elements
 *   FAIL_NOP: d_mat still contains poison (TMA store did nothing)
 *   FAIL_MISMATCH: d_mat has wrong values (descriptor or store produced bad data)
 *
 * Expected pattern at global position (gx, gy): 1 + gx + gy * width_stride
 *
 * Usage: ./utmastg [-w <width>] [-h <height>] [-i <iterations>]
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
__global__ void test_UTMASTG(const __grid_constant__ CUtensorMap tensor_map,
                              int run_iters)
{
    __shared__ alignas(128) int32_t smem_buffer[SMEM_H][SMEM_W];

    int x = (int)(blockDim.x * blockIdx.x);
    int y = (int)(blockDim.y * blockIdx.y);

    // Each thread fills smem with a globally-unique pattern
    int gx = x + (int)threadIdx.x;
    int gy = y + (int)threadIdx.y;
    smem_buffer[threadIdx.y][threadIdx.x] =
        1 + gx + gy * (int)(blockDim.x * gridDim.x);

    ptx::fence_proxy_async(ptx::space_shared);
    __syncthreads();

    for (int i = 0; i < run_iters; i++) {
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            ptx::cp_async_bulk_tensor(
                ptx::space_global, ptx::space_shared,
                &tensor_map, {x, y}, &smem_buffer);
            ptx::cp_async_bulk_commit_group();
            ptx::cp_async_bulk_wait_group_read(ptx::n32_t<0>());
        }
    }
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
static bool verify(const int32_t *h_out,
                   uint64_t height, uint64_t width, uint64_t width_stride)
{
    int poison_count = 0, mismatch_count = 0;
    for (uint64_t r = 0; r < height; r++) {
        for (uint64_t c = 0; c < width; c++) {
            int32_t got = h_out[r * width_stride + c];
            int32_t exp = 1 + (int32_t)c + (int32_t)r * (int32_t)width_stride;
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

    // Host buffer: poisoned (TMA store must overwrite)
    int32_t *h_buf = (int32_t *)malloc(bytes);
    for (size_t i = 0; i < height_stride * width_stride; i++)
        h_buf[i] = POISON;

    int32_t *d_mat;
    cudaMalloc(&d_mat, bytes);
    cudaMemcpy(d_mat, h_buf, bytes, cudaMemcpyHostToDevice);

    // Build tensor descriptor
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
    CUDA_SAFECALL((test_UTMASTG<<<grid, block>>>(tensor_map, run_iters)));
    cudaMemcpy(h_buf, d_mat, bytes, cudaMemcpyDeviceToHost);

    printf("=== UTMASTG (TMA Tensor Store: Shared -> Global) ===\n");
    bool pass = verify(h_buf, height, width, width_stride);
    printf("RESULT: %s\n", pass ? "PASS" : "FAIL");

    free(h_buf);
    cudaFree(d_mat);
    return pass ? 0 : 1;
}
