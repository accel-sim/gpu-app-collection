/**
 * UTMALDG_3D_TX: TMA Tensor Load rank-3 with mbarrier::complete_tx + L2::cache_hint
 * ====================================================================================
 * Exercises:
 *   cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint
 *
 * This is the exact instruction emitted by CUTLASS's SM90_TMA_LOAD_3D::copy()
 * (cute/arch/copy_sm90_tma.hpp) and used by example 48_hopper_warp_specialized_gemm.
 *
 * Unlike utmaldg_3d.cu (which avoids mbarrier by using __syncthreads), this ubench
 * exercises the full mbarrier completion path:
 *   1. mbarrier.init.shared::cta.b64        — initialize mbarrier, thread_count=1
 *   2. mbarrier.arrive.expect_tx.shared::cta.b64 — thread 0 arrives + declares expected bytes
 *   3. TMA load with .mbarrier::complete_tx::bytes.L2::cache_hint — hw signals mbar on done
 *   4. mbarrier.try_wait.parity.shared::cta.b64 — spin until mbarrier complete
 *
 * Single-CTA launch: avoids the GPGPU-Sim assertion
 *   "A mbarrier should only exist once in local/remote shmem"
 * which fires when multiple CTAs each call mbarrier.init at the same relative smem offset.
 *
 * PASS:    smem_buffer was correctly populated by TMA; scalar readback matches source.
 * FAIL:    smem_buffer contains poison or wrong values.
 * TIMEOUT: mbarrier wait never returns (TMA completion signaling not implemented in sim).
 *
 * Usage: ./utmaldg_3d_tx
 */

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cudaTypedefs.h>
#include <cuda.h>

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
__global__ void test_UTMALDG_3D_TX(const __grid_constant__ CUtensorMap tensor_map,
                                    int32_t *d_dst)
{
    __shared__ alignas(128) int32_t smem_buffer[TILE_D][TILE_H][TILE_W];
    // Raw uint64_t mbarrier — same as CUTLASS SM90_TMA_LOAD_3D uses.
    // Not cuda::barrier: we replicate the exact CUTLASS pattern with raw PTX.
    __shared__ uint64_t mbar;

    const int x = 0, y = 0, z = 0;  // single tile at origin (single-CTA launch)

    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        uint32_t smem_buf_addr  = static_cast<uint32_t>(__cvta_generic_to_shared(&smem_buffer));
        uint32_t smem_mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&mbar));

        // 1. Init mbarrier: 1 expected arrival (thread 0 only, producer-warp pattern)
        asm volatile(
            "mbarrier.init.shared::cta.b64 [%0], %1;"
            :: "r"(smem_mbar_addr), "r"(1)
            : "memory");

        // 2. arrive.expect_tx: thread 0 provides its 1 arrival + declares tx byte count.
        //    mbarrier is now complete when TMA delivers sizeof(smem_buffer) bytes.
        asm volatile(
            "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
            :: "r"(smem_mbar_addr), "r"((uint32_t)sizeof(smem_buffer))
            : "memory");

        // 3. TMA load: 3D, shared::cluster dst, global src, mbarrier::complete_tx, L2::cache_hint.
        //    cache_hint=1 is a valid evict-first hint; treated as NOP by GPGPU-Sim.
        //    Argument order matches CUTLASS SM90_TMA_LOAD_3D (copy_sm90_tma.hpp):
        //      [smem_dst], [tmap, {crd0, crd1, crd2}], [mbar], cache_hint
        uint64_t cache_hint = 1ULL;
        asm volatile(
            "cp.async.bulk.tensor.3d.shared::cluster.global"
            ".mbarrier::complete_tx::bytes.L2::cache_hint"
            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
            :: "r"(smem_buf_addr),
               "l"((uint64_t)&tensor_map),
               "r"(x), "r"(y), "r"(z),
               "r"(smem_mbar_addr),
               "l"(cache_hint)
            : "memory");

        // 4. Spin-wait on mbarrier phase 0 (matches CUTLASS wait_barrier pattern).
        asm volatile(
            "{\n"
            ".reg .pred P1;\n"
            "LAB_WAIT:\n"
            "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
            "@P1 bra DONE;\n"
            "bra LAB_WAIT;\n"
            "DONE:\n"
            "}\n"
            :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&mbar))),
               "r"(0)   // phase 0 (initial phase)
            : "memory");
    }
    __syncthreads();

    // Scalar readback: verify TMA populated smem_buffer correctly
    d_dst[threadIdx.z * TILE_H * TILE_W + threadIdx.y * TILE_W + threadIdx.x] =
        smem_buffer[threadIdx.z][threadIdx.y][threadIdx.x];
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
static bool verify(const int32_t *h_dst, const int32_t *h_src, uint32_t n)
{
    int poison_count = 0, mismatch_count = 0;
    for (uint32_t i = 0; i < n; i++) {
        int32_t got = h_dst[i];
        int32_t exp = h_src[i];
        if (got == POISON) {
            if (poison_count == 0)
                printf("  NOP at [%u]: got 0x%08X, expected %d\n",
                       i, (unsigned)got, exp);
            poison_count++;
        } else if (got != exp) {
            if (mismatch_count == 0)
                printf("  Mismatch at [%u]: got %d, expected %d\n", i, got, exp);
            mismatch_count++;
        }
    }
    if (poison_count > 0) {
        printf("  FAIL: %d/%u elements contain poison (TMA NOP or mbarrier wait skipped)\n",
               poison_count, n);
        return false;
    }
    if (mismatch_count > 0) {
        printf("  FAIL: %d/%u elements have wrong values\n", mismatch_count, n);
        return false;
    }
    return true;
}

// ============================================================================
// Main
// ============================================================================
int main(int, char **)
{
    constexpr uint32_t N = TILE_W * TILE_H * TILE_D;  // 256 elements, one tile
    size_t bytes = N * sizeof(int32_t);

    int32_t *h_src = (int32_t *)malloc(bytes);
    int32_t *h_dst = (int32_t *)malloc(bytes);

    for (uint32_t i = 0; i < N; i++) h_src[i] = (int32_t)(i + 1);
    for (uint32_t i = 0; i < N; i++) h_dst[i] = POISON;

    int32_t *d_src, *d_dst;
    cudaMalloc(&d_src, bytes);
    cudaMalloc(&d_dst, bytes);
    cudaMemcpy(d_src, h_src, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst, h_dst, bytes, cudaMemcpyHostToDevice);

    // Rank-3 descriptor: tensor exactly one tile in size.
    // globalDim  = {TILE_W, TILE_H, TILE_D}
    // globalStrides (bytes between rows): stride[0]=TILE_W*4, stride[1]=TILE_W*TILE_H*4
    CUtensorMap tensor_map{};
    constexpr uint32_t rank = 3;
    uint64_t size[rank]        = {TILE_W, TILE_H, TILE_D};
    uint64_t stride[rank - 1]  = {TILE_W * sizeof(int32_t),
                                   TILE_W * TILE_H * sizeof(int32_t)};
    uint32_t box_size[rank]    = {TILE_W, TILE_H, TILE_D};
    uint32_t elem_stride[rank] = {1, 1, 1};
    auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();
    cuTensorMapEncodeTiled(
        &tensor_map, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_INT32,
        rank, d_src, size, stride, box_size, elem_stride,
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    // Single CTA: grid=(1,1,1), block=(TILE_W, TILE_H, TILE_D) = 256 threads.
    // This avoids the GPGPU-Sim mbarrier duplicate-registration assertion
    // that fires when multiple CTAs each init a mbarrier at the same relative smem offset.
    dim3 grid(1, 1, 1);
    dim3 block(TILE_W, TILE_H, TILE_D);
    CUDA_SAFECALL((test_UTMALDG_3D_TX<<<grid, block>>>(tensor_map, d_dst)));
    cudaMemcpy(h_dst, d_dst, bytes, cudaMemcpyDeviceToHost);

    printf("=== UTMALDG_3D_TX (TMA rank-3 load: mbarrier::complete_tx + L2::cache_hint) ===\n");
    bool pass = verify(h_dst, h_src, N);
    printf("RESULT: %s\n", pass ? "PASS" : "FAIL");

    free(h_src); free(h_dst);
    cudaFree(d_src); cudaFree(d_dst);
    return pass ? 0 : 1;
}
