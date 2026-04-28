/*
 * TMA Tensor Multicast Mask Test
 * ==============================
 * Tests: cp.async.bulk.tensor.2d.shared::cluster.global
 *        .mbarrier::complete_tx::bytes.multicast::cluster
 *
 * Minimal configuration: 1 cluster, 2 CTAs, 1 tile (32x32 ints).
 * The multicast mask is a runtime parameter, enabling targeted testing
 * of which CTAs receive data.
 *
 * Usage: ./tma_tensor_multicast_mask [-m <mask>]
 *   -m 1  (0b01): only rank 0 receives data
 *   -m 2  (0b10): only rank 1 receives data
 *   -m 3  (0b11): both ranks receive (broadcast)
 *
 * Verification Strategy:
 * ----------------------
 *   - Shared memory is initialized to POISON (0xDEADBEEF) before TMA
 *   - CTA 0 issues multicast TMA with the given mask
 *   - CTAs IN the mask: mbarrier expects TMA bytes, waits for delivery
 *   - CTAs NOT in the mask: mbarrier expects 0 bytes, completes immediately
 *   - After TMA:
 *     - In-mask CTAs: smem should contain tile data -> PASS
 *     - Not-in-mask CTAs: smem should still be POISON -> PASS
 *   - Each CTA copies smem to its own d_dst region for host-side checking
 *
 * Key mask=0b10 case:
 *   Rank 0 ISSUES the instruction but does NOT receive data.
 *   Rank 1 does NOT issue the instruction but DOES receive data.
 *   This tests that the simulator correctly routes data to a remote CTA.
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

#define TILE_DIM      32
#define CLUSTER_DIM_X 2
#define POISON_VALUE  0xDEADBEEF

// ============================================================================
// Kernel
//
// Layout: grid=(2,1), block=(32,32), cluster_dims=(2,1,1)
//   -> 1 cluster containing CTA rank 0 and CTA rank 1
//   -> Both CTAs address the same tile at origin (0,0)
//
// Flow per CTA:
//   1. Poison smem with POISON_VALUE
//   2. If in_mask: barrier_arrive_tx with sizeof(smem) expected bytes
//      If not in_mask: barrier arrive with 0 expected bytes
//   3. Rank 0 issues multicast TMA (regardless of whether rank 0 is in mask)
//   4. All threads wait on barrier
//   5. Copy smem -> d_dst for host verification
// ============================================================================

__global__ void __cluster_dims__(CLUSTER_DIM_X, 1, 1)
test_tma_multicast_mask_kernel(
    const __grid_constant__ CUtensorMap tensor_map,
    int *d_dst,
    uint16_t mcast_mask)
{
    __shared__ alignas(128) int smem[TILE_DIM][TILE_DIM];

#pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier bar;

    cg::cluster_group cluster = cg::this_cluster();
    unsigned int rank = cluster.block_rank();
    bool in_mask = (mcast_mask >> rank) & 1;

    // Step 1: Poison smem so we can detect whether TMA wrote anything
    smem[threadIdx.y][threadIdx.x] = (int)POISON_VALUE;

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        init(&bar, blockDim.x * blockDim.y);
        ptx::fence_proxy_async(ptx::space_shared);
    }
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        printf("[rank %u] in_mask=%d, blockIdx=(%u,%u)\n",
               rank, (int)in_mask, blockIdx.x, blockIdx.y);
    }

    // Step 2 & 3: Arrive on barrier and optionally issue TMA
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        barrier::arrival_token token;
        if (in_mask) {
            token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(smem));
        } else {
            token = bar.arrive();
        }

        if (rank == 0) {
            uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
            uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));

            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cluster.global"
                ".mbarrier::complete_tx::bytes.multicast::cluster"
                " [%0], [%1, {%2, %3}], [%4], %5;"
                :
                : "r"(smem_addr), "l"(&tensor_map),
                  "r"(0), "r"(0),
                  "r"(mbar_addr), "h"(mcast_mask)
                : "memory"
            );
        }

        bar.wait(std::move(token));
    } else {
        bar.arrive_and_wait();
    }

    __syncthreads();

    // Step 4: Copy smem -> d_dst for host verification
    int dst_base = rank * TILE_DIM * TILE_DIM;
    d_dst[dst_base + threadIdx.y * TILE_DIM + threadIdx.x] =
        smem[threadIdx.y][threadIdx.x];
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
// Main
// ============================================================================

int main(int argc, char *argv[])
{
    uint16_t mcast_mask = 0x3;

    int opt;
    while ((opt = getopt(argc, argv, "m:")) != -1) {
        switch (opt) {
            case 'm': mcast_mask = (uint16_t)atoi(optarg); break;
            default:
                fprintf(stderr, "Usage: %s [-m <mask>]\n", argv[0]);
                fprintf(stderr, "  -m 1  (0b01): only rank 0 receives\n");
                fprintf(stderr, "  -m 2  (0b10): only rank 1 receives\n");
                fprintf(stderr, "  -m 3  (0b11): both ranks receive\n");
                return 1;
        }
    }

    mcast_mask &= (1u << CLUSTER_DIM_X) - 1;
    if (mcast_mask == 0) {
        fprintf(stderr, "Error: mask=0 is invalid (no destination CTA)\n");
        return 1;
    }

    printf("=== TMA Multicast Mask Test ===\n");
    printf("Cluster: %d CTAs, Tile: %dx%d ints\n", CLUSTER_DIM_X, TILE_DIM, TILE_DIM);
    printf("Multicast mask: 0x%X (0b", mcast_mask);
    for (int i = CLUSTER_DIM_X - 1; i >= 0; i--)
        printf("%d", (mcast_mask >> i) & 1);
    printf(")\n");
    for (int i = 0; i < CLUSTER_DIM_X; i++)
        printf("  Rank %d: %s\n", i,
               (mcast_mask >> i) & 1 ? "RECEIVES data" : "does NOT receive (expects poison)");
    printf("\n");

    // ---- Source tensor: 1 tile, values 1..TILE_DIM*TILE_DIM ----
    const uint64_t width  = TILE_DIM;
    const uint64_t height = TILE_DIM;
    size_t mat_bytes = width * height * sizeof(int);
    int *mat = (int *)malloc(mat_bytes);
    int *d_mat;
    cudaMalloc(&d_mat, mat_bytes);

    for (uint64_t i = 0; i < width * height; i++)
        mat[i] = (int)(i + 1);
    cudaMemcpy(d_mat, mat, mat_bytes, cudaMemcpyHostToDevice);

    // ---- Tensor map ----
    CUtensorMap tensor_map{};
    constexpr uint32_t tma_rank = 2;
    uint64_t size[tma_rank]        = {width, height};
    uint64_t stride[tma_rank - 1]  = {width * sizeof(int)};
    uint32_t box_size[tma_rank]    = {TILE_DIM, TILE_DIM};
    uint32_t elem_stride[tma_rank] = {1, 1};

    auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();
    CUresult res = cuTensorMapEncodeTiled(
        &tensor_map,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_INT32,
        tma_rank,
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

    // ---- Destination buffer: CLUSTER_DIM_X regions, each TILE_DIM*TILE_DIM ----
    size_t dst_elements = (size_t)CLUSTER_DIM_X * TILE_DIM * TILE_DIM;
    size_t dst_bytes    = dst_elements * sizeof(int);
    int *h_dst = (int *)malloc(dst_bytes);
    int *d_dst;
    cudaMalloc(&d_dst, dst_bytes);

    for (size_t i = 0; i < dst_elements; i++)
        h_dst[i] = (int)POISON_VALUE;
    cudaMemcpy(d_dst, h_dst, dst_bytes, cudaMemcpyHostToDevice);

    // ---- Launch: 1 cluster = CLUSTER_DIM_X blocks ----
    dim3 grid(CLUSTER_DIM_X, 1);
    dim3 block(TILE_DIM, TILE_DIM);

    printf("Grid: (%u, %u), Block: (%u, %u)\n", grid.x, grid.y, block.x, block.y);
    printf("Launching 1 cluster of %d CTAs...\n\n", CLUSTER_DIM_X);

    CUDA_SAFECALL((test_tma_multicast_mask_kernel<<<grid, block>>>(
        tensor_map, d_dst, mcast_mask)));
    CUDA_SAFECALL(cudaDeviceSynchronize());
    CUDA_SAFECALL(cudaMemcpy(h_dst, d_dst, dst_bytes, cudaMemcpyDeviceToHost));

    // ---- Verify each rank ----
    int all_pass = 1;

    for (int r = 0; r < CLUSTER_DIM_X; r++) {
        bool in_mask = (mcast_mask >> r) & 1;
        int base = r * TILE_DIM * TILE_DIM;
        int poison_count = 0, mismatch_count = 0, correct_count = 0;

        for (int i = 0; i < TILE_DIM * TILE_DIM; i++) {
            int got      = h_dst[base + i];
            int expected = in_mask ? mat[i] : (int)POISON_VALUE;

            if (got == expected) {
                correct_count++;
            } else if (got == (int)POISON_VALUE && in_mask) {
                if (poison_count == 0)
                    printf("  Rank %d: elem %d still POISON (got 0x%08X, expected 0x%08X)\n",
                           r, i, (unsigned)got, (unsigned)expected);
                poison_count++;
            } else {
                if (mismatch_count == 0)
                    printf("  Rank %d: elem %d MISMATCH (got 0x%08X, expected 0x%08X)\n",
                           r, i, (unsigned)got, (unsigned)expected);
                mismatch_count++;
            }
        }

        int total = TILE_DIM * TILE_DIM;
        if (correct_count == total) {
            if (in_mask)
                printf("  Rank %d: PASS - received correct tile data (%d/%d)\n",
                       r, correct_count, total);
            else
                printf("  Rank %d: PASS - correctly still has poison (%d/%d)\n",
                       r, correct_count, total);
        } else {
            all_pass = 0;
            if (in_mask && poison_count > 0)
                printf("  Rank %d: FAIL - NOP detected, %d/%d elements still poison\n",
                       r, poison_count, total);
            else
                printf("  Rank %d: FAIL - %d/%d elements wrong (%d poison, %d mismatch)\n",
                       r, poison_count + mismatch_count, total, poison_count, mismatch_count);
        }
    }

    printf("\n");
    printf("================================================================================\n");
    printf("TEST: TMA Tensor Multicast Mask (mask=0x%X)\n", mcast_mask);
    printf("  Instruction: cp.async.bulk.tensor.2d.shared::cluster.global\n");
    printf("               .mbarrier::complete_tx::bytes.multicast::cluster\n");
    printf("================================================================================\n");
    printf("RESULT: %s\n", all_pass ? "**PASS**" : "**FAIL**");
    printf("================================================================================\n\n");

    // ---- Dump for debugging ----
    char filename[128];
    sprintf(filename, "tma_multicast_mask_0x%X.txt", mcast_mask);
    FILE *f = fopen(filename, "w");
    fprintf(f, "# Mask=0x%X  Cluster=%d CTAs  Tile=%dx%d\n\n", mcast_mask, CLUSTER_DIM_X, TILE_DIM, TILE_DIM);
    for (int r = 0; r < CLUSTER_DIM_X; r++) {
        bool in_mask = (mcast_mask >> r) & 1;
        fprintf(f, "--- Rank %d (%s) ---\n", r,
                in_mask ? "in mask" : "NOT in mask");
        int base = r * TILE_DIM * TILE_DIM;
        for (int row = 0; row < TILE_DIM; row++) {
            for (int col = 0; col < TILE_DIM; col++)
                fprintf(f, "0x%08x ", (unsigned)h_dst[base + row * TILE_DIM + col]);
            fprintf(f, "\n");
        }
        fprintf(f, "\n");
    }
    fclose(f);
    printf("Values dumped to %s\n", filename);

    free(mat);
    free(h_dst);
    cudaFree(d_mat);
    cudaFree(d_dst);

    return all_pass ? 0 : 1;
}
