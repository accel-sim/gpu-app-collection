/*
 * TMA Tensor Load with Multicast Microbenchmark
 * ==============================================
 * Tests: cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster
 *
 * This benchmark verifies TMA tensor load with multicast to multiple CTAs
 * within a cluster. Each cluster of CLUSTER_DIM_X CTAs loads a single 2D tile
 * from global memory. CTA 0 in each cluster issues the multicast TMA
 * instruction, which delivers the tile data to ALL CTAs' shared memories
 * simultaneously. Each CTA then copies its shared memory to a unique region
 * of a destination buffer for host-side verification.
 *
 * Usage: ./tma_tensor_multicast [-w <width>] [-h <height>] [-i <iterations>]
 *
 * Verification Strategy:
 * ----------------------
 *   - d_mat (global tensor): initialized with unique values (1, 2, 3, ...)
 *   - d_dst (verification buffer): poisoned with POISON_VALUE (0xDEADBEEF)
 *   - Each cluster of 2 CTAs loads the SAME tile via multicast TMA
 *   - Each CTA copies its smem to a unique d_dst region (standard stores)
 *   - Verification checks:
 *     1. Correctness: every block's d_dst region matches the expected tile
 *     2. Multicast:   CTAs within the same cluster have identical data
 *   - PASS: all blocks correct AND cluster-mates identical
 *   - FAIL_NOP: d_dst still contains poison (TMA had no effect)
 *   - FAIL_MISMATCH: values differ from expected
 *   - FAIL_MULTICAST: CTAs in same cluster received different data
 */

#include <cudaTypedefs.h>
#include <cuda.h>
#include <cuda/barrier>
#include <cooperative_groups.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cg = cooperative_groups;
namespace ptx = cuda::ptx;

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

#define GMEM_WIDTH  128
#define GMEM_HEIGHT 64
#define SMEM_WIDTH  32
#define SMEM_HEIGHT 32
#define CLUSTER_DIM_X 2
#define DEFAULT_RUN_ITERS 1
#define POISON_VALUE 0xDEADBEEF

// ============================================================================
// Kernel: TMA tensor load with multicast::cluster
//
// Grid layout (example with defaults):
//   grid_dim = (num_tiles_x * CLUSTER_DIM_X, num_tiles_y)
//            = (4*2, 2) = (8, 2) = 16 blocks total
//   Cluster (i,j) contains blocks (2*i, j) and (2*i+1, j)
//   Both blocks in a cluster load the SAME tile from global memory
//
// Each cluster loads tile at:
//   tile_x = cluster_idx_x * SMEM_WIDTH
//   tile_y = cluster_idx_y * SMEM_HEIGHT
//
// CTA with cluster_rank==0 issues the multicast TMA instruction.
// All CTAs in the cluster receive the data in their shared memory.
// Each CTA writes its smem to d_dst at a unique offset for verification.
// ============================================================================

__global__ void __cluster_dims__(CLUSTER_DIM_X, 1, 1)
test_tma_multicast_kernel(
    const __grid_constant__ CUtensorMap tensor_map,
    int *d_dst,
    int run_iters)
{
    __shared__ alignas(128) int smem_buffer[SMEM_HEIGHT][SMEM_WIDTH];

#pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier bar;

    cg::cluster_group cluster = cg::this_cluster();
    unsigned int cluster_rank = cluster.block_rank();

    unsigned int cluster_idx_x = blockIdx.x / CLUSTER_DIM_X;
    int tile_x = cluster_idx_x * SMEM_WIDTH;
    int tile_y = blockIdx.y * SMEM_HEIGHT;

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        init(&bar, blockDim.x * blockDim.y);
        ptx::fence_proxy_async(ptx::space_shared);
    }
    __syncthreads();

    if (blockIdx.x == 0 && blockIdx.y == 0 &&
        threadIdx.x == 0 && threadIdx.y == 0) {
        printf("TensorMap address: %p\n", &tensor_map);
        printf("Cluster rank: %u, tile: (%d, %d)\n", cluster_rank, tile_x, tile_y);
    }

    for (int iter = 0; iter < run_iters; iter++) {
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            // Thread 0: arrive with expected transaction bytes
            auto token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(smem_buffer));

            if (cluster_rank == 0) {
                uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_buffer));
                uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
                uint16_t mcast_mask = static_cast<uint16_t>((1u << CLUSTER_DIM_X) - 1u);

                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster"
                    " [%0], [%1, {%2, %3}], [%4], %5;"
                    :
                    : "r"(smem_addr), "l"(&tensor_map), "r"(tile_x), "r"(tile_y),
                      "r"(mbar_addr), "h"(mcast_mask)
                    : "memory"
                );
            }

            bar.wait(std::move(token));
        } else {
            bar.arrive_and_wait();
        }
    }

    // Each block writes its smem to a unique region of d_dst
    unsigned int block_flat_idx = blockIdx.x + blockIdx.y * gridDim.x;
    int dst_offset = block_flat_idx * SMEM_HEIGHT * SMEM_WIDTH;
    d_dst[dst_offset + threadIdx.y * SMEM_WIDTH + threadIdx.x] =
        smem_buffer[threadIdx.y][threadIdx.x];
}

// ============================================================================
// Driver API helper
// ============================================================================

PFN_cuTensorMapEncodeTiled_v12000 get_cuTensorMapEncodeTiled()
{
    cudaDriverEntryPointQueryResult driver_status;
    void *cuTensorMapEncodeTiled_ptr = nullptr;
    CUDA_SAFECALL(cudaGetDriverEntryPointByVersion(
        "cuTensorMapEncodeTiled", &cuTensorMapEncodeTiled_ptr,
        12000, cudaEnableDefault, &driver_status));
    assert(driver_status == cudaDriverEntryPointSuccess);
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(cuTensorMapEncodeTiled_ptr);
}

// ============================================================================
// Verification
// ============================================================================

enum verify_result_t {
    VERIFY_PASS,
    VERIFY_FAIL_NOP,
    VERIFY_FAIL_MISMATCH,
    VERIFY_FAIL_MULTICAST
};

static const char* verify_result_str(verify_result_t r) {
    switch (r) {
        case VERIFY_PASS:           return "PASS";
        case VERIFY_FAIL_NOP:       return "FAIL (NOP - TMA had no effect)";
        case VERIFY_FAIL_MISMATCH:  return "FAIL (MISMATCH - wrong values)";
        case VERIFY_FAIL_MULTICAST: return "FAIL (MULTICAST - CTAs in cluster got different data)";
        default: return "UNKNOWN";
    }
}

int main(int argc, char *argv[])
{
    uint64_t width  = GMEM_WIDTH;
    uint64_t height = GMEM_HEIGHT;
    int run_iters   = DEFAULT_RUN_ITERS;

    int opt;
    while ((opt = getopt(argc, argv, "w:h:i:")) != -1) {
        switch (opt) {
            case 'w': width  = uint64_t(atoi(optarg)); break;
            case 'h': height = uint64_t(atoi(optarg)); break;
            case 'i': run_iters = atoi(optarg); break;
            default:
                fprintf(stderr, "Usage: %s [-w <width>] [-h <height>] [-i <iterations>]\n", argv[0]);
                fprintf(stderr, "  Tile: %dx%d, Cluster: %d CTAs\n",
                        SMEM_WIDTH, SMEM_HEIGHT, CLUSTER_DIM_X);
                return 1;
        }
    }

    uint64_t height_stride = ((height + (SMEM_HEIGHT - 1)) / SMEM_HEIGHT) * SMEM_HEIGHT;
    uint64_t width_stride  = ((width  + (SMEM_WIDTH  - 1)) / SMEM_WIDTH)  * SMEM_WIDTH;

    printf("=== TMA Tensor Multicast Benchmark ===\n");
    printf("Global tensor: %lu x %lu (stride: %lu x %lu)\n",
           width, height, width_stride, height_stride);
    printf("Tile size: %d x %d\n", SMEM_WIDTH, SMEM_HEIGHT);
    printf("Cluster: %d CTAs in x\n", CLUSTER_DIM_X);
    printf("Iterations: %d\n", run_iters);

    // ---- Allocate and initialize source tensor ----
    size_t mat_bytes = height_stride * width_stride * sizeof(int);
    int *mat = (int *)malloc(mat_bytes);
    int *d_mat;
    cudaMalloc(&d_mat, mat_bytes);

    uint64_t val = 1;
    for (uint64_t r = 0; r < height_stride; r++) {
        for (uint64_t c = 0; c < width_stride; c++) {
            mat[r * width_stride + c] = (r < height && c < width) ? (int)(val++) : 0;
        }
    }
    cudaMemcpy(d_mat, mat, mat_bytes, cudaMemcpyHostToDevice);

    // ---- Create tensor map ----
    CUtensorMap tensor_map{};
    constexpr uint32_t rank = 2;
    uint64_t size[rank]          = {width, height};
    uint64_t stride[rank - 1]    = {width_stride * sizeof(int)};
    uint32_t box_size[rank]      = {SMEM_WIDTH, SMEM_HEIGHT};
    uint32_t elem_stride[rank]   = {1, 1};

    auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();
    CUresult res = cuTensorMapEncodeTiled(
        &tensor_map,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_INT32,
        rank,
        d_mat,
        size,
        stride,
        box_size,
        elem_stride,
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "cuTensorMapEncodeTiled failed: %d\n", res);
        return 1;
    }

    // ---- Grid / block configuration ----
    // Each cluster handles one tile; cluster has CLUSTER_DIM_X blocks
    uint32_t num_tiles_x = width_stride  / SMEM_WIDTH;
    uint32_t num_tiles_y = height_stride / SMEM_HEIGHT;
    dim3 grid_dim(num_tiles_x * CLUSTER_DIM_X, num_tiles_y);
    dim3 block_dim(SMEM_WIDTH, SMEM_HEIGHT);
    uint32_t total_blocks = grid_dim.x * grid_dim.y;

    printf("Grid: (%u, %u), Block: (%u, %u)\n",
           grid_dim.x, grid_dim.y, block_dim.x, block_dim.y);
    printf("Clusters: %u x %u = %u total (%u blocks/cluster)\n",
           num_tiles_x, num_tiles_y, num_tiles_x * num_tiles_y, CLUSTER_DIM_X);
    printf("Total blocks: %u\n\n", total_blocks);

    // ---- Allocate and poison destination buffer ----
    size_t dst_elements = (size_t)total_blocks * SMEM_HEIGHT * SMEM_WIDTH;
    size_t dst_bytes    = dst_elements * sizeof(int);
    int *h_dst = (int *)malloc(dst_bytes);
    int *d_dst;
    cudaMalloc(&d_dst, dst_bytes);

    for (size_t idx = 0; idx < dst_elements; idx++)
        h_dst[idx] = (int)POISON_VALUE;
    cudaMemcpy(d_dst, h_dst, dst_bytes, cudaMemcpyHostToDevice);

    // ---- Launch kernel ----
    CUDA_SAFECALL((test_tma_multicast_kernel<<<grid_dim, block_dim>>>(
        tensor_map, d_dst, run_iters)));
    CUDA_SAFECALL(cudaDeviceSynchronize());
    CUDA_SAFECALL(cudaMemcpy(h_dst, d_dst, dst_bytes, cudaMemcpyDeviceToHost));

    // ---- Verify correctness ----
    int poison_count   = 0;
    int mismatch_count = 0;

    for (uint32_t bx = 0; bx < grid_dim.x; bx++) {
        for (uint32_t by = 0; by < grid_dim.y; by++) {
            uint32_t cluster_idx_x = bx / CLUSTER_DIM_X;
            int tile_x = cluster_idx_x * SMEM_WIDTH;
            int tile_y = by * SMEM_HEIGHT;

            size_t dst_offset = (size_t)(bx + by * grid_dim.x) * SMEM_HEIGHT * SMEM_WIDTH;

            for (int r = 0; r < SMEM_HEIGHT; r++) {
                for (int c = 0; c < SMEM_WIDTH; c++) {
                    int got      = h_dst[dst_offset + r * SMEM_WIDTH + c];
                    int expected = mat[(tile_y + r) * width_stride + (tile_x + c)];

                    if (got == (int)POISON_VALUE && expected != (int)POISON_VALUE) {
                        if (poison_count == 0)
                            printf("  Poison at block(%u,%u)[%d][%d]: "
                                   "got 0x%08X, expected 0x%08X\n",
                                   bx, by, r, c, (unsigned)got, (unsigned)expected);
                        poison_count++;
                    } else if (got != expected) {
                        if (mismatch_count == 0)
                            printf("  Mismatch at block(%u,%u)[%d][%d]: "
                                   "got 0x%08X, expected 0x%08X\n",
                                   bx, by, r, c, (unsigned)got, (unsigned)expected);
                        mismatch_count++;
                    }
                }
            }
        }
    }

    // ---- Verify multicast: CTAs in same cluster must have identical data ----
    int multicast_fail = 0;

    for (uint32_t by = 0; by < grid_dim.y; by++) {
        for (uint32_t ci = 0; ci < num_tiles_x; ci++) {
            uint32_t bx0 = ci * CLUSTER_DIM_X;
            size_t off0 = (size_t)(bx0 + by * grid_dim.x) * SMEM_HEIGHT * SMEM_WIDTH;

            for (uint32_t rank = 1; rank < CLUSTER_DIM_X; rank++) {
                size_t off1 = (size_t)(bx0 + rank + by * grid_dim.x)
                              * SMEM_HEIGHT * SMEM_WIDTH;
                for (int i = 0; i < SMEM_HEIGHT * SMEM_WIDTH; i++) {
                    if (h_dst[off0 + i] != h_dst[off1 + i]) {
                        if (multicast_fail == 0)
                            printf("  Multicast mismatch: cluster(%u,%u) "
                                   "rank0=0x%08X rank%u=0x%08X at elem %d\n",
                                   ci, by,
                                   (unsigned)h_dst[off0 + i], rank,
                                   (unsigned)h_dst[off1 + i], i);
                        multicast_fail++;
                    }
                }
            }
        }
    }

    // ---- Report results ----
    verify_result_t result = VERIFY_PASS;
    if (poison_count > 0)        result = VERIFY_FAIL_NOP;
    else if (mismatch_count > 0) result = VERIFY_FAIL_MISMATCH;
    else if (multicast_fail > 0) result = VERIFY_FAIL_MULTICAST;

    printf("\n");
    printf("================================================================================\n");
    printf("TEST: TMA Tensor Load with Multicast (Global -> Shared::Cluster)\n");
    printf("  Instruction: cp.async.bulk.tensor.2d.shared::cluster.global\n");
    printf("               .mbarrier::complete_tx::bytes.multicast::cluster\n");
    printf("================================================================================\n");
    printf("RESULT: %s\n", verify_result_str(result));
    if (result == VERIFY_PASS) {
        printf("  All %u blocks received correct tile data\n", total_blocks);
        printf("  Multicast verified: CTAs in each cluster have identical data\n");
    }
    if (poison_count > 0)
        printf("  %d elements still contain poison\n", poison_count);
    if (mismatch_count > 0)
        printf("  %d elements have wrong values\n", mismatch_count);
    if (multicast_fail > 0)
        printf("  %d elements differ between CTAs in same cluster\n", multicast_fail);
    printf("================================================================================\n\n");

    // ---- Dump raw output for debugging ----
    char filename[128];
    sprintf(filename, "tma_multicast_test_%lu_%lu.txt", height, width);
    FILE *f = fopen(filename, "w");
    fprintf(f, "# TMA Multicast Test Output\n");
    fprintf(f, "# Format: block(bx,by) [row][col] = value (hex)\n\n");
    for (uint32_t bx = 0; bx < grid_dim.x; bx++) {
        for (uint32_t by = 0; by < grid_dim.y; by++) {
            size_t dst_offset = (size_t)(bx + by * grid_dim.x) * SMEM_HEIGHT * SMEM_WIDTH;
            fprintf(f, "--- block(%u,%u) cluster_rank=%u tile=(%d,%d) ---\n",
                    bx, by, bx % CLUSTER_DIM_X,
                    (int)(bx / CLUSTER_DIM_X) * SMEM_WIDTH,
                    (int)(by * SMEM_HEIGHT));
            for (int r = 0; r < SMEM_HEIGHT; r++) {
                for (int c = 0; c < SMEM_WIDTH; c++) {
                    fprintf(f, "0x%08x ", (unsigned)h_dst[dst_offset + r * SMEM_WIDTH + c]);
                }
                fprintf(f, "\n");
            }
            fprintf(f, "\n");
        }
    }
    fclose(f);
    printf("Values dumped to %s\n", filename);

    // ---- Cleanup ----
    free(mat);
    free(h_dst);
    cudaFree(d_mat);
    cudaFree(d_dst);

    return (result == VERIFY_PASS) ? 0 : 1;
}
