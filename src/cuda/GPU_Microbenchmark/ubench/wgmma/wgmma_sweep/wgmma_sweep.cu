// =============================================================================
// wgmma_sweep.cu  —  Combined WGMMA timing + D-fragment / smem layout sweep
//
// Fills ALL measurement gaps left by lat_gmma and wgmma_multik benchmarks.
//
// PHASE 0 — Shape-independent overhead (run once)
//   kernel_fence_overhead:        256 FENCEs             → cyc_per_fence
//   kernel_commit_wait_overhead:  256 empty COMMIT+WAITs → cyc_per_barrier
//
// PHASE 1 — SS timing sweep (smem-A, smem-B)
//   kernel_mma_throughput<N>:  1024 back-to-back SS MMAs → cyc_per_mma
//   Derived true latency:
//     lat = cyc_per_mma*N_MMA - 2*cyc_fence - (N_MMA-1)*II - cyc_barrier
//   Covers all 6 dtypes × 9 N values (N=8,16,32,64,96,128,192,224,256):
//     F32F16, F32TF32, F32BF16, F32E4M3, F16F16, S32S8
//   Output: wgmma_sweep_timing.csv
//
// PHASE 1b — Pipelined 2-group overlap (SS, F32F16, all N)
//   Issues N_MMA/2 MMAs + COMMIT + N_MMA/2 MMAs + COMMIT + WAIT<1> + WAIT<0>.
//   gap_after_wait1 = clock(after WAIT<0>) - clock(after WAIT<1>).
//   If gap ≈ 0: group 2 completed during group 1's latency tail → overlap confirmed.
//   Output: wgmma_sweep_pipeline.csv
//
// PHASE 2 — D-fragment layout verification (all N)
//   A=1, B[k][n]=n → D[m][n]=K*n. Checks d_frag_pos column formula.
//   F32F16 (F32 accum) : N=8..256 — primary
//   F16F16 (F16 accum) : N=8..256 — verifies F16 accumulator column mapping
//   S32S8  (S32 accum) : N=8..256 — verifies int accumulator column mapping
//                        B[k][n] = n%64 (int8 overflow guard), expected K*(col%64)
//   Output: wgmma_sweep_layout_col<dtype>N<N>.txt per shape
//
// PHASE 3 — smem_A layout verification (selected N and K values)
//   A[m][k]=m, B[k][n]=1 → D[m][n]=K*m. Verifies smem_A row layout
//   AND d_frag_pos row formula jointly.
//   F16  K=16: N=16,32,64,128,256 (K-interleaved for 2-byte elements)
//   TF32 K=8 : N=16,32,64,128    (K-interleaved for 4-byte elements)
//   S8   K=32: N=16,32,64        (K-interleaved for 1-byte elements)
//   Output: wgmma_sweep_layout_row<dtype>N<N>.txt per shape
//
// PHASE 4 — RS timing sweep (register-A, smem-B)
//   kernel_mma_throughput_rs<N>: 1024 back-to-back RS MMAs → cyc_per_mma_rs
//   A register fragment filled with 1s (values immaterial for timing).
//   RS is supported for F32{F16,BF16,TF32} and F16F16 dtypes.
//   Covers all 4 RS dtypes × 9 N values.
//   Output: appended to wgmma_sweep_timing.csv (section marked RS)
//
// Build: make  (see Makefile)
// =============================================================================

#include <cuda.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>

#include "cute/arch/util.hpp"
#include <cutlass/cutlass.h>
#include "cutlass/numeric_types.h"
#include <cute/tensor.hpp>
#include <cute/arch/mma_sm90.hpp>
#include <cute/arch/mma_sm90_gmma.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>

using namespace cute;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static constexpr int WGSIZE    = 128;   // one warpgroup
static constexpr int N_FENCE   = 256;
static constexpr int N_BARRIER = 256;
static constexpr int N_MMA     = 1024;
static constexpr int N_HALF    = N_MMA / 2;

#define WGMMA_FENCE  asm volatile("wgmma.fence.sync.aligned;\n"        ::: "memory")
#define WGMMA_COMMIT asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory")
#define WGMMA_WAIT0  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory")
#define WGMMA_WAIT1  asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory")

// ---------------------------------------------------------------------------
// Kernel 1: Fence overhead (shape-independent)
// ---------------------------------------------------------------------------
__global__ void kernel_fence_overhead(uint32_t* clk_out) {
    const int tid = threadIdx.x;
    uint32_t t0 = 0, t1 = 0;
    __syncthreads();
    if (tid == 0) asm volatile("mov.u32 %0, %%clock;" : "=r"(t0) :: "memory");
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < N_FENCE; i++) WGMMA_FENCE;
    __syncthreads();
    if (tid == 0) asm volatile("mov.u32 %0, %%clock;" : "=r"(t1) :: "memory");
    if (tid == 0) { clk_out[0] = t0; clk_out[1] = t1; }
}

// ---------------------------------------------------------------------------
// Kernel 2: Commit+wait overhead with EMPTY group (shape-independent)
// ---------------------------------------------------------------------------
__global__ void kernel_commit_wait_overhead(uint32_t* clk_out) {
    const int tid = threadIdx.x;
    uint32_t t0 = 0, t1 = 0;
    WGMMA_FENCE;
    __syncthreads();
    if (tid == 0) asm volatile("mov.u32 %0, %%clock;" : "=r"(t0) :: "memory");
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < N_BARRIER; i++) { WGMMA_COMMIT; WGMMA_WAIT0; }
    __syncthreads();
    if (tid == 0) asm volatile("mov.u32 %0, %%clock;" : "=r"(t1) :: "memory");
    if (tid == 0) { clk_out[0] = t0; clk_out[1] = t1; }
}

// ---------------------------------------------------------------------------
// Kernel 3: SS MMA throughput timing
// FENCE + N_MMA*MMA + COMMIT + WAIT + FENCE
// cyc_per_mma = total / N_MMA
// ---------------------------------------------------------------------------
template<class EA, class EB, class EC, class TileShape>
__global__ void kernel_mma_throughput(uint32_t* clk_out) {
    const int tid = threadIdx.x;
    const int wg  = tid / cutlass::NumThreadsPerWarpGroup;

    auto gmma_op  = GMMA::ss_op_selector<EA, EB, EC, TileShape,
                                          GMMA::Major::K, GMMA::Major::K>();
    using TiledMma = decltype(make_tiled_mma(gmma_op));
    TiledMma tiled_mma;

    static constexpr int PIPE = 1;
    using SmemLayoutA = decltype(tile_to_shape(
        GMMA::Layout_K_INTER_Atom<EA>{},
        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<PIPE>{})));
    using SmemLayoutB = decltype(tile_to_shape(
        GMMA::Layout_K_INTER_Atom<EB>{},
        make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<PIPE>{})));

    __shared__ EA smem_A[cosize_v<SmemLayoutA>];
    __shared__ EB smem_B[cosize_v<SmemLayoutB>];

    Tensor sA = make_tensor(make_smem_ptr(smem_A), SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(smem_B), SmemLayoutB{});
    for (int i = tid; i < (int)cosize_v<SmemLayoutA>; i += WGSIZE)
        smem_A[i] = EA(1.0f);
    for (int i = tid; i < (int)cosize_v<SmemLayoutB>; i += WGSIZE)
        smem_B[i] = EB(1.0f);
    __syncthreads();

    Layout wg_layout = make_layout(Int<1>{}, Int<cutlass::NumThreadsPerWarpGroup>{});
    auto thread_mma = tiled_mma.get_slice(wg_layout(wg));
    auto tCsA = thread_mma.partition_A(sA);
    auto tCsB = thread_mma.partition_B(sB);
    auto tCrA = thread_mma.make_fragment_A(tCsA);
    auto tCrB = thread_mma.make_fragment_B(tCsB);
    auto accum = partition_fragment_C(tiled_mma, take<0,2>(TileShape{}));
    clear(accum);

    uint32_t t0 = 0, t1 = 0;
    __syncthreads();
    if (tid == 0) asm volatile("mov.u32 %0, %%clock;" : "=r"(t0) :: "memory");
    __syncthreads();

    warpgroup_fence_operand(accum);
    warpgroup_arrive();
    #pragma unroll 8
    for (int j = 0; j < N_MMA; j++)
        cute::gemm(tiled_mma, tCrA(_,_,_,0), tCrB(_,_,_,0), accum);
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    warpgroup_fence_operand(accum);

    __syncthreads();
    if (tid == 0) asm volatile("mov.u32 %0, %%clock;" : "=r"(t1) :: "memory");
    if (tid == 0) { clk_out[0] = t0; clk_out[1] = t1; }
    if (tid == 0 && accum(0) == -9999.f) printf("x");  // prevent DCE
}

// ---------------------------------------------------------------------------
// Kernel 4: RS MMA throughput timing (register-A, smem-B)
// Same structure as kernel_mma_throughput but uses rs_op_selector.
// A register fragment is pre-filled with zeros (values immaterial for timing).
// ---------------------------------------------------------------------------
template<class EA, class EB, class EC, class TileShape>
__global__ void kernel_mma_throughput_rs(uint32_t* clk_out) {
    const int tid = threadIdx.x;
    const int wg  = tid / cutlass::NumThreadsPerWarpGroup;

    auto gmma_op  = GMMA::rs_op_selector<EA, EB, EC, TileShape,
                                          GMMA::Major::K, GMMA::Major::K>();
    using TiledMma = decltype(make_tiled_mma(gmma_op));
    TiledMma tiled_mma;

    static constexpr int PIPE = 1;
    using SmemLayoutA = decltype(tile_to_shape(
        GMMA::Layout_K_INTER_Atom<EA>{},
        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<PIPE>{})));
    using SmemLayoutB = decltype(tile_to_shape(
        GMMA::Layout_K_INTER_Atom<EB>{},
        make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<PIPE>{})));

    __shared__ EA smem_A[cosize_v<SmemLayoutA>];
    __shared__ EB smem_B[cosize_v<SmemLayoutB>];

    Tensor sA = make_tensor(make_smem_ptr(smem_A), SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(smem_B), SmemLayoutB{});
    for (int i = tid; i < (int)cosize_v<SmemLayoutA>; i += WGSIZE)
        smem_A[i] = EA(1.0f);
    for (int i = tid; i < (int)cosize_v<SmemLayoutB>; i += WGSIZE)
        smem_B[i] = EB(1.0f);
    __syncthreads();

    Layout wg_layout = make_layout(Int<1>{}, Int<cutlass::NumThreadsPerWarpGroup>{});
    auto thread_mma = tiled_mma.get_slice(wg_layout(wg));

    // For RS: partition_A gives smem shape; make_fragment_A returns a REGISTER
    // tensor (FrgTypeA is a value type for RS atoms, not a smem-pointer wrapper).
    // clear() zeros the registers — values don't affect timing.
    auto tCsA = thread_mma.partition_A(sA);
    auto tCrA = thread_mma.make_fragment_A(tCsA);
    clear(tCrA);

    auto tCsB = thread_mma.partition_B(sB);
    auto tCrB = thread_mma.make_fragment_B(tCsB);
    auto accum = partition_fragment_C(tiled_mma, take<0,2>(TileShape{}));
    clear(accum);

    uint32_t t0 = 0, t1 = 0;
    __syncthreads();
    if (tid == 0) asm volatile("mov.u32 %0, %%clock;" : "=r"(t0) :: "memory");
    __syncthreads();

    warpgroup_fence_operand(accum);
    warpgroup_arrive();
    #pragma unroll 8
    for (int j = 0; j < N_MMA; j++)
        cute::gemm(tiled_mma, tCrA(_,_,_,0), tCrB(_,_,_,0), accum);
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    warpgroup_fence_operand(accum);

    __syncthreads();
    if (tid == 0) asm volatile("mov.u32 %0, %%clock;" : "=r"(t1) :: "memory");
    if (tid == 0) { clk_out[0] = t0; clk_out[1] = t1; }
    if (tid == 0 && accum(0) == -9999.f) printf("x");
}

// ---------------------------------------------------------------------------
// Kernel 5: Pipelined 2-group SS timing
// Issues two committed groups of N_HALF MMAs each:
//   FENCE + N_HALF*MMA + COMMIT + FENCE + N_HALF*MMA + COMMIT
//   + WAIT<1> [wait for group 1, group 2 still pending]
//   + WAIT<0> [wait for group 2]
// Outputs 3 clock values: [t_before, t_after_wait1, t_after_wait0].
// gap_after_wait1 = clk[2]-clk[1]: if small → group 2 completed during
// group 1's latency tail (true overlap).
// ---------------------------------------------------------------------------
template<class EA, class EB, class EC, class TileShape>
__global__ void kernel_mma_pipelined(uint32_t* clk_out) {
    const int tid = threadIdx.x;
    const int wg  = tid / cutlass::NumThreadsPerWarpGroup;

    auto gmma_op  = GMMA::ss_op_selector<EA, EB, EC, TileShape,
                                          GMMA::Major::K, GMMA::Major::K>();
    using TiledMma = decltype(make_tiled_mma(gmma_op));
    TiledMma tiled_mma;

    static constexpr int PIPE = 1;
    using SmemLayoutA = decltype(tile_to_shape(
        GMMA::Layout_K_INTER_Atom<EA>{},
        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<PIPE>{})));
    using SmemLayoutB = decltype(tile_to_shape(
        GMMA::Layout_K_INTER_Atom<EB>{},
        make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<PIPE>{})));

    __shared__ EA smem_A[cosize_v<SmemLayoutA>];
    __shared__ EB smem_B[cosize_v<SmemLayoutB>];

    Tensor sA = make_tensor(make_smem_ptr(smem_A), SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(smem_B), SmemLayoutB{});
    for (int i = tid; i < (int)cosize_v<SmemLayoutA>; i += WGSIZE)
        smem_A[i] = EA(1.0f);
    for (int i = tid; i < (int)cosize_v<SmemLayoutB>; i += WGSIZE)
        smem_B[i] = EB(1.0f);
    __syncthreads();

    Layout wg_layout = make_layout(Int<1>{}, Int<cutlass::NumThreadsPerWarpGroup>{});
    auto thread_mma = tiled_mma.get_slice(wg_layout(wg));
    auto tCsA = thread_mma.partition_A(sA);
    auto tCsB = thread_mma.partition_B(sB);
    auto tCrA = thread_mma.make_fragment_A(tCsA);
    auto tCrB = thread_mma.make_fragment_B(tCsB);
    auto accum = partition_fragment_C(tiled_mma, take<0,2>(TileShape{}));
    clear(accum);

    uint32_t t0 = 0, t1 = 0, t2 = 0;
    __syncthreads();
    if (tid == 0) asm volatile("mov.u32 %0, %%clock;" : "=r"(t0) :: "memory");
    __syncthreads();

    warpgroup_fence_operand(accum);

    // Group 1: N_HALF MMAs
    warpgroup_arrive();
    #pragma unroll 8
    for (int j = 0; j < N_HALF; j++)
        cute::gemm(tiled_mma, tCrA(_,_,_,0), tCrB(_,_,_,0), accum);
    warpgroup_commit_batch();

    // Group 2: N_HALF more MMAs (overlaps with group 1's latency tail)
    warpgroup_arrive();
    #pragma unroll 8
    for (int j = 0; j < N_HALF; j++)
        cute::gemm(tiled_mma, tCrA(_,_,_,0), tCrB(_,_,_,0), accum);
    warpgroup_commit_batch();

    // Wait for group 1 only (1 group still pending); group 2 may be done already
    warpgroup_wait<1>();
    if (tid == 0) asm volatile("mov.u32 %0, %%clock;" : "=r"(t1) :: "memory");

    // Wait for group 2
    warpgroup_wait<0>();
    if (tid == 0) asm volatile("mov.u32 %0, %%clock;" : "=r"(t2) :: "memory");

    warpgroup_fence_operand(accum);

    if (tid == 0) { clk_out[0] = t0; clk_out[1] = t1; clk_out[2] = t2; }
    if (tid == 0 && accum(0) == -9999.f) printf("x");
}

// ---------------------------------------------------------------------------
// Kernel 6: D-fragment column layout verification
// A[m][k]=1, B[k][n]=(n%FILL_MOD if FILL_MOD>0 else n) => D[m][n]=K*(n or n%FILL_MOD)
// Verifies d_frag_pos column formula. FILL_MOD=64 for int8 B to avoid overflow.
// ---------------------------------------------------------------------------
template<class EA, class EB, class EC, class TileShape, int FILL_MOD = 0>
__global__ void kernel_mma_layout_col(EC* D_out) {
    const int tid = threadIdx.x;
    const int wg  = tid / cutlass::NumThreadsPerWarpGroup;

    static constexpr int N_val = (int)get<1>(TileShape{});
    static constexpr int K_val = (int)get<2>(TileShape{});

    auto gmma_op  = GMMA::ss_op_selector<EA, EB, EC, TileShape,
                                          GMMA::Major::K, GMMA::Major::K>();
    using TiledMma = decltype(make_tiled_mma(gmma_op));
    TiledMma tiled_mma;

    static constexpr int PIPE = 1;
    using SmemLayoutA = decltype(tile_to_shape(
        GMMA::Layout_K_INTER_Atom<EA>{},
        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<PIPE>{})));
    using SmemLayoutB = decltype(tile_to_shape(
        GMMA::Layout_K_INTER_Atom<EB>{},
        make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<PIPE>{})));

    __shared__ EA smem_A[cosize_v<SmemLayoutA>];
    __shared__ EB smem_B[cosize_v<SmemLayoutB>];

    Tensor sA = make_tensor(make_smem_ptr(smem_A), SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(smem_B), SmemLayoutB{});

    for (int i = tid; i < (int)cosize_v<SmemLayoutA>; i += WGSIZE)
        smem_A[i] = EA(1.0f);
    for (int i = tid; i < N_val * K_val; i += WGSIZE) {
        int n = i % N_val, k = i / N_val;
        int n_fill = (FILL_MOD > 0) ? (n % FILL_MOD) : n;
        sB(n, k, 0) = EB(float(n_fill));
    }
    __syncthreads();

    Layout wg_layout = make_layout(Int<1>{}, Int<cutlass::NumThreadsPerWarpGroup>{});
    auto thread_mma = tiled_mma.get_slice(wg_layout(wg));
    auto tCsA = thread_mma.partition_A(sA);
    auto tCsB = thread_mma.partition_B(sB);
    auto tCrA = thread_mma.make_fragment_A(tCsA);
    auto tCrB = thread_mma.make_fragment_B(tCsB);
    auto accum = partition_fragment_C(tiled_mma, take<0,2>(TileShape{}));
    clear(accum);

    warpgroup_fence_operand(accum);
    warpgroup_arrive();
    cute::gemm(tiled_mma, tCrA(_,_,_,0), tCrB(_,_,_,0), accum);
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    warpgroup_fence_operand(accum);

    const int D_ELEMS = (int)size(accum);
    EC* base = D_out + tid * D_ELEMS;
    for (int e = 0; e < D_ELEMS; e++)
        base[e] = accum(e);
}

// ---------------------------------------------------------------------------
// Kernel 7: smem_A row layout verification
// A[m][k]=m, B[k][n]=1 => D[m][n]=K*m
// Verifies d_frag_pos row formula AND smem_A K-interleaved swizzled layout.
// ---------------------------------------------------------------------------
template<class EA, class EB, class EC, class TileShape>
__global__ void kernel_mma_layout_row(EC* D_out) {
    const int tid = threadIdx.x;
    const int wg  = tid / cutlass::NumThreadsPerWarpGroup;

    static constexpr int M_val = (int)get<0>(TileShape{});
    static constexpr int N_val = (int)get<1>(TileShape{});
    static constexpr int K_val = (int)get<2>(TileShape{});

    auto gmma_op  = GMMA::ss_op_selector<EA, EB, EC, TileShape,
                                          GMMA::Major::K, GMMA::Major::K>();
    using TiledMma = decltype(make_tiled_mma(gmma_op));
    TiledMma tiled_mma;

    static constexpr int PIPE = 1;
    using SmemLayoutA = decltype(tile_to_shape(
        GMMA::Layout_K_INTER_Atom<EA>{},
        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<PIPE>{})));
    using SmemLayoutB = decltype(tile_to_shape(
        GMMA::Layout_K_INTER_Atom<EB>{},
        make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<PIPE>{})));

    __shared__ EA smem_A[cosize_v<SmemLayoutA>];
    __shared__ EB smem_B[cosize_v<SmemLayoutB>];

    Tensor sA = make_tensor(make_smem_ptr(smem_A), SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(smem_B), SmemLayoutB{});

    for (int i = tid; i < M_val * K_val; i += WGSIZE) {
        int m = i / K_val, k = i % K_val;
        sA(m, k, 0) = EA(float(m));
    }
    for (int i = tid; i < (int)cosize_v<SmemLayoutB>; i += WGSIZE)
        smem_B[i] = EB(1.0f);
    __syncthreads();

    Layout wg_layout = make_layout(Int<1>{}, Int<cutlass::NumThreadsPerWarpGroup>{});
    auto thread_mma = tiled_mma.get_slice(wg_layout(wg));
    auto tCsA = thread_mma.partition_A(sA);
    auto tCsB = thread_mma.partition_B(sB);
    auto tCrA = thread_mma.make_fragment_A(tCsA);
    auto tCrB = thread_mma.make_fragment_B(tCsB);
    auto accum = partition_fragment_C(tiled_mma, take<0,2>(TileShape{}));
    clear(accum);

    warpgroup_fence_operand(accum);
    warpgroup_arrive();
    cute::gemm(tiled_mma, tCrA(_,_,_,0), tCrB(_,_,_,0), accum);
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    warpgroup_fence_operand(accum);

    const int D_ELEMS = (int)size(accum);
    EC* base = D_out + tid * D_ELEMS;
    for (int e = 0; e < D_ELEMS; e++)
        base[e] = accum(e);
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------

// d_frag_pos: maps (thread T, accumulator element e, tile width N) → (row, col)
// Valid for f32/s32 accumulator at all N. For f16 accumulator, the spatial
// mapping is identical (same warpgroup thread layout), just different precision.
// elems_per_group=4 for f32/s32 and for f16 (each f16 element is 1 per slot).
static void d_frag_pos(int T, int e, int /*N*/, int* row, int* col) {
    int warp = T / 32, lane = T % 32;
    int g = e / 4, k = e % 4;
    *row = (lane / 4) * 2 + (k / 2) + warp * 16;
    *col = (lane % 4) * 2 + (k % 2) + g * 8;
}

static float measure_fence_overhead() {
    uint32_t *d_clk;
    cudaMalloc(&d_clk, 2 * sizeof(uint32_t));
    kernel_fence_overhead<<<1, WGSIZE>>>(d_clk);
    cudaDeviceSynchronize();
    uint32_t clk[2];
    cudaMemcpy(clk, d_clk, 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_clk);
    return (float)(clk[1] - clk[0]) / N_FENCE;
}

static float measure_barrier_overhead() {
    uint32_t *d_clk;
    cudaMalloc(&d_clk, 2 * sizeof(uint32_t));
    kernel_commit_wait_overhead<<<1, WGSIZE>>>(d_clk);
    cudaDeviceSynchronize();
    uint32_t clk[2];
    cudaMemcpy(clk, d_clk, 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_clk);
    return (float)(clk[1] - clk[0]) / N_BARRIER;
}

// Known II values from lat_gmma silicon run (H100, 2026-05-08).
// N=224 is interpolated (II=112); verify with this benchmark.
static int table_II(int N) {
    switch (N) {
        case   8: return  18;  case  16: return  20;  case  32: return  24;
        case  64: return  32;  case  96: return  48;  case 128: return  64;
        case 192: return  96;  case 224: return 112;  case 256: return 128;
        default:  return  N / 2;
    }
}

// SS timing: calls kernel_mma_throughput, derives latency
template<class EA, class EB, class EC, class TileShape>
static void run_timing(const char* name, int N, float cyc_fence,
                       float cyc_barrier, FILE* csv) {
    uint32_t *d_clk;
    cudaMalloc(&d_clk, 2 * sizeof(uint32_t));
    kernel_mma_throughput<EA, EB, EC, TileShape><<<1, WGSIZE>>>(d_clk);
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        printf("  %-60s KERNEL ERROR: %s\n", name, cudaGetErrorString(err));
        cudaFree(d_clk); return;
    }
    cudaDeviceSynchronize();
    uint32_t clk[2];
    cudaMemcpy(clk, d_clk, 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_clk);

    float cyc_mma = (float)(clk[1] - clk[0]) / N_MMA;
    int   II      = table_II(N);
    float lat     = cyc_mma * N_MMA - 2.0f * cyc_fence
                    - (float)(N_MMA - 1) * II - cyc_barrier;
    printf("  %-60s cyc/mma=%7.3f  II=%3d  lat=%7.3f\n", name, cyc_mma, II, lat);
    if (csv)
        fprintf(csv, "%s,SS,%.3f,%.3f,%.3f,%d,%.3f\n",
                name, cyc_fence, cyc_barrier, cyc_mma, II, lat);
}

// RS timing: calls kernel_mma_throughput_rs, derives latency
template<class EA, class EB, class EC, class TileShape>
static void run_timing_rs(const char* name, int N, float cyc_fence,
                          float cyc_barrier, FILE* csv) {
    uint32_t *d_clk;
    cudaMalloc(&d_clk, 2 * sizeof(uint32_t));
    kernel_mma_throughput_rs<EA, EB, EC, TileShape><<<1, WGSIZE>>>(d_clk);
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        printf("  RS %-57s KERNEL ERROR: %s\n", name, cudaGetErrorString(err));
        cudaFree(d_clk); return;
    }
    cudaDeviceSynchronize();
    uint32_t clk[2];
    cudaMemcpy(clk, d_clk, 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_clk);

    float cyc_mma = (float)(clk[1] - clk[0]) / N_MMA;
    int   II      = table_II(N);
    float lat     = cyc_mma * N_MMA - 2.0f * cyc_fence
                    - (float)(N_MMA - 1) * II - cyc_barrier;
    printf("  RS %-57s cyc/mma=%7.3f  II=%3d  lat=%7.3f\n", name, cyc_mma, II, lat);
    if (csv)
        fprintf(csv, "%s,RS,%.3f,%.3f,%.3f,%d,%.3f\n",
                name, cyc_fence, cyc_barrier, cyc_mma, II, lat);
}

// Pipelined 2-group timing: reports total cyc/mma and gap between WAIT<1> and WAIT<0>
template<class EA, class EB, class EC, class TileShape>
static void run_timing_pipelined(const char* name, int N, FILE* csv) {
    uint32_t *d_clk;
    cudaMalloc(&d_clk, 3 * sizeof(uint32_t));
    kernel_mma_pipelined<EA, EB, EC, TileShape><<<1, WGSIZE>>>(d_clk);
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        printf("  Pipe %-55s KERNEL ERROR: %s\n", name, cudaGetErrorString(err));
        cudaFree(d_clk); return;
    }
    cudaDeviceSynchronize();
    uint32_t clk[3];
    cudaMemcpy(clk, d_clk, 3 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_clk);

    float cyc_total  = (float)(clk[2] - clk[0]) / N_MMA;
    float gap_wait1  = (float)(clk[2] - clk[1]);    // cycles from WAIT<1> to WAIT<0>
    printf("  Pipe %-55s cyc/mma=%7.3f  gap_after_wait1=%7.1f%s\n",
           name, cyc_total, gap_wait1, gap_wait1 < 5.0f ? "  [OVERLAP]" : "");
    if (csv)
        fprintf(csv, "%s,%.3f,%.1f\n", name, cyc_total, gap_wait1);
}

// Layout column verification: A=1, B[k][n]=n (or n%FILL_MOD) => D=K*n (or K*(n%FILL_MOD))
template<class EA, class EB, class EC, class TileShape, int FILL_MOD = 0>
static void run_layout_col(const char* tag, const char* name, int N, int K) {
    const int D_ELEMS = N / 2;   // universal: f32, f16, s32 accumulators all = N/2
    const int total   = WGSIZE * D_ELEMS;
    std::vector<EC> h_D(total, EC(0));
    EC* d_D;
    cudaMalloc(&d_D, total * sizeof(EC));
    cudaMemset(d_D, 0, total * sizeof(EC));

    kernel_mma_layout_col<EA, EB, EC, TileShape, FILL_MOD><<<1, WGSIZE>>>(d_D);
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        printf("  ColLayout %-50s KERNEL ERROR: %s\n", name, cudaGetErrorString(err));
        cudaFree(d_D); return;
    }
    cudaDeviceSynchronize();
    cudaMemcpy(h_D.data(), d_D, total * sizeof(EC), cudaMemcpyDeviceToHost);
    cudaFree(d_D);

    char fname[128];
    snprintf(fname, sizeof(fname), "wgmma_sweep_layout_col%sN%d.txt", tag, N);
    FILE* f = fopen(fname, "w");
    if (f) {
        if (FILL_MOD > 0)
            fprintf(f, "# %s  A=1  B[k][n]=n%%64  expected D[m][n]=K*(n%%64) (K=%d)\n"
                       "# thread elem row col expected actual match\n", name, K);
        else
            fprintf(f, "# %s  A=1  B[k][n]=n  expected D[m][n]=K*n (K=%d)\n"
                       "# thread elem row col expected actual match\n", name, K);
    }

    int errors = 0, checked = 0;
    for (int T = 0; T < WGSIZE; T++) {
        for (int e = 0; e < D_ELEMS; e++) {
            int row, col;
            d_frag_pos(T, e, N, &row, &col);
            if (row >= 64 || col >= N) continue;
            int col_fill = (FILL_MOD > 0) ? (col % FILL_MOD) : col;
            float expected = (float)(K * col_fill);
            float actual   = (float)h_D[T * D_ELEMS + e];
            bool  ok       = (fabsf(actual - expected) < 0.5f);
            if (!ok) errors++;
            checked++;
            if (f)
                fprintf(f, "%d %d %d %d %.1f %.1f %s\n",
                        T, e, row, col, expected, actual, ok ? "OK" : "FAIL");
        }
    }
    if (f) { fprintf(f, "# errors=%d / %d checked\n", errors, checked); fclose(f); }
    printf("  ColLayout%-6s %-48s errors=%d/%d  -> %s\n",
           tag, name, errors, checked, errors == 0 ? "PASS" : "FAIL");
}

// Layout row verification: A[m][k]=m, B=1 => D=K*m
template<class EA, class EB, class EC, class TileShape>
static void run_layout_row(const char* tag, const char* name, int N, int K) {
    const int D_ELEMS = N / 2;
    const int total   = WGSIZE * D_ELEMS;
    std::vector<EC> h_D(total, EC(0));
    EC* d_D;
    cudaMalloc(&d_D, total * sizeof(EC));
    cudaMemset(d_D, 0, total * sizeof(EC));

    kernel_mma_layout_row<EA, EB, EC, TileShape><<<1, WGSIZE>>>(d_D);
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        printf("  RowLayout %-50s KERNEL ERROR: %s\n", name, cudaGetErrorString(err));
        cudaFree(d_D); return;
    }
    cudaDeviceSynchronize();
    cudaMemcpy(h_D.data(), d_D, total * sizeof(EC), cudaMemcpyDeviceToHost);
    cudaFree(d_D);

    char fname[128];
    snprintf(fname, sizeof(fname), "wgmma_sweep_layout_row%sN%d.txt", tag, N);
    FILE* f = fopen(fname, "w");
    if (f)
        fprintf(f, "# %s  A[m][k]=m  B=1  expected D[m][n]=K*m (K=%d)\n"
                   "# thread elem row col expected actual match\n", name, K);

    int errors = 0, checked = 0;
    for (int T = 0; T < WGSIZE; T++) {
        for (int e = 0; e < D_ELEMS; e++) {
            int row, col;
            d_frag_pos(T, e, N, &row, &col);
            if (row >= 64 || col >= N) continue;
            float expected = (float)(K * row);
            float actual   = (float)h_D[T * D_ELEMS + e];
            bool  ok       = (fabsf(actual - expected) < 0.5f);
            if (!ok) errors++;
            checked++;
            if (f)
                fprintf(f, "%d %d %d %d %.1f %.1f %s\n",
                        T, e, row, col, expected, actual, ok ? "OK" : "FAIL");
        }
    }
    if (f) { fprintf(f, "# errors=%d / %d checked\n", errors, checked); fclose(f); }
    printf("  RowLayout%-6s %-48s errors=%d/%d  -> %s\n",
           tag, name, errors, checked, errors == 0 ? "PASS" : "FAIL");
}

// ---------------------------------------------------------------------------
// Convenience macros
// ---------------------------------------------------------------------------
#define SHAPE(M,N,K) decltype(make_shape(Int<M>{}, Int<N>{}, Int<K>{}))

#define RUN_TIMING(EA, EB, EC, M, N, K, LABEL) \
    run_timing<EA, EB, EC, SHAPE(M,N,K)>(LABEL, N, cyc_fence, cyc_barrier, timing_csv)

#define RUN_TIMING_RS(EA, EB, EC, M, N, K, LABEL) \
    run_timing_rs<EA, EB, EC, SHAPE(M,N,K)>(LABEL, N, cyc_fence, cyc_barrier, timing_csv)

#define RUN_PIPELINED(EA, EB, EC, M, N, K, LABEL) \
    run_timing_pipelined<EA, EB, EC, SHAPE(M,N,K)>(LABEL, N, pipeline_csv)

#define RUN_COL_LAYOUT(EA, EB, EC, M, N, K, TAG, LABEL) \
    run_layout_col<EA, EB, EC, SHAPE(M,N,K), 0>(TAG, LABEL, N, K)

#define RUN_COL_LAYOUT_MOD(EA, EB, EC, M, N, K, MOD, TAG, LABEL) \
    run_layout_col<EA, EB, EC, SHAPE(M,N,K), MOD>(TAG, LABEL, N, K)

#define RUN_ROW_LAYOUT(EA, EB, EC, M, N, K, TAG, LABEL) \
    run_layout_row<EA, EB, EC, SHAPE(M,N,K)>(TAG, LABEL, N, K)

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("wgmma_sweep  -  %s (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);

    // -----------------------------------------------------------------------
    // Phase 0: Shape-independent overhead
    // -----------------------------------------------------------------------
    printf("=== Phase 0: Overhead kernels (shape-independent) ===\n");
    float cyc_fence   = measure_fence_overhead();
    float cyc_barrier = measure_barrier_overhead();
    printf("  cyc_per_fence   = %.3f  (256 x wgmma.fence.sync.aligned)\n", cyc_fence);
    printf("  cyc_per_barrier = %.3f  (256 x COMMIT+WAIT empty group)\n\n", cyc_barrier);

    FILE* timing_csv = fopen("wgmma_sweep_timing.csv", "w");
    if (timing_csv)
        fprintf(timing_csv, "shape,mode,cyc_fence,cyc_barrier,cyc_mma,II_table,derived_lat\n");

    // -----------------------------------------------------------------------
    // Phase 1: SS timing sweep — all dtypes × all N
    // -----------------------------------------------------------------------
    printf("=== Phase 1: SS timing sweep ===\n");
    printf("  %-60s  cyc/mma   II   derived_lat\n", "shape");

    printf("\n  -- F32 accumulator, F16 inputs (K=16) --\n");
    RUN_TIMING(cutlass::half_t, cutlass::half_t, float, 64,   8, 16, "F32F16_m64n8k16");
    RUN_TIMING(cutlass::half_t, cutlass::half_t, float, 64,  16, 16, "F32F16_m64n16k16");
    RUN_TIMING(cutlass::half_t, cutlass::half_t, float, 64,  32, 16, "F32F16_m64n32k16");
    RUN_TIMING(cutlass::half_t, cutlass::half_t, float, 64,  64, 16, "F32F16_m64n64k16");
    RUN_TIMING(cutlass::half_t, cutlass::half_t, float, 64,  96, 16, "F32F16_m64n96k16");
    RUN_TIMING(cutlass::half_t, cutlass::half_t, float, 64, 128, 16, "F32F16_m64n128k16");
    RUN_TIMING(cutlass::half_t, cutlass::half_t, float, 64, 192, 16, "F32F16_m64n192k16");
    RUN_TIMING(cutlass::half_t, cutlass::half_t, float, 64, 224, 16, "F32F16_m64n224k16");
    RUN_TIMING(cutlass::half_t, cutlass::half_t, float, 64, 256, 16, "F32F16_m64n256k16");

    printf("\n  -- F32 accumulator, TF32 inputs (K=8) --\n");
    RUN_TIMING(cutlass::tfloat32_t, cutlass::tfloat32_t, float, 64,   8,  8, "F32TF32_m64n8k8");
    RUN_TIMING(cutlass::tfloat32_t, cutlass::tfloat32_t, float, 64,  16,  8, "F32TF32_m64n16k8");
    RUN_TIMING(cutlass::tfloat32_t, cutlass::tfloat32_t, float, 64,  32,  8, "F32TF32_m64n32k8");
    RUN_TIMING(cutlass::tfloat32_t, cutlass::tfloat32_t, float, 64,  64,  8, "F32TF32_m64n64k8");
    RUN_TIMING(cutlass::tfloat32_t, cutlass::tfloat32_t, float, 64,  96,  8, "F32TF32_m64n96k8");
    RUN_TIMING(cutlass::tfloat32_t, cutlass::tfloat32_t, float, 64, 128,  8, "F32TF32_m64n128k8");
    RUN_TIMING(cutlass::tfloat32_t, cutlass::tfloat32_t, float, 64, 192,  8, "F32TF32_m64n192k8");
    RUN_TIMING(cutlass::tfloat32_t, cutlass::tfloat32_t, float, 64, 224,  8, "F32TF32_m64n224k8");
    RUN_TIMING(cutlass::tfloat32_t, cutlass::tfloat32_t, float, 64, 256,  8, "F32TF32_m64n256k8");

    printf("\n  -- F32 accumulator, BF16 inputs (K=16) --\n");
    RUN_TIMING(cutlass::bfloat16_t, cutlass::bfloat16_t, float, 64,   8, 16, "F32BF16_m64n8k16");
    RUN_TIMING(cutlass::bfloat16_t, cutlass::bfloat16_t, float, 64,  16, 16, "F32BF16_m64n16k16");
    RUN_TIMING(cutlass::bfloat16_t, cutlass::bfloat16_t, float, 64,  32, 16, "F32BF16_m64n32k16");
    RUN_TIMING(cutlass::bfloat16_t, cutlass::bfloat16_t, float, 64,  64, 16, "F32BF16_m64n64k16");
    RUN_TIMING(cutlass::bfloat16_t, cutlass::bfloat16_t, float, 64,  96, 16, "F32BF16_m64n96k16");
    RUN_TIMING(cutlass::bfloat16_t, cutlass::bfloat16_t, float, 64, 128, 16, "F32BF16_m64n128k16");
    RUN_TIMING(cutlass::bfloat16_t, cutlass::bfloat16_t, float, 64, 192, 16, "F32BF16_m64n192k16");
    RUN_TIMING(cutlass::bfloat16_t, cutlass::bfloat16_t, float, 64, 224, 16, "F32BF16_m64n224k16");
    RUN_TIMING(cutlass::bfloat16_t, cutlass::bfloat16_t, float, 64, 256, 16, "F32BF16_m64n256k16");

    printf("\n  -- F32 accumulator, FP8-E4M3 inputs (K=32) --\n");
    RUN_TIMING(cutlass::float_e4m3_t, cutlass::float_e4m3_t, float, 64,   8, 32, "F32E4M3_m64n8k32");
    RUN_TIMING(cutlass::float_e4m3_t, cutlass::float_e4m3_t, float, 64,  16, 32, "F32E4M3_m64n16k32");
    RUN_TIMING(cutlass::float_e4m3_t, cutlass::float_e4m3_t, float, 64,  32, 32, "F32E4M3_m64n32k32");
    RUN_TIMING(cutlass::float_e4m3_t, cutlass::float_e4m3_t, float, 64,  64, 32, "F32E4M3_m64n64k32");
    RUN_TIMING(cutlass::float_e4m3_t, cutlass::float_e4m3_t, float, 64,  96, 32, "F32E4M3_m64n96k32");
    RUN_TIMING(cutlass::float_e4m3_t, cutlass::float_e4m3_t, float, 64, 128, 32, "F32E4M3_m64n128k32");
    RUN_TIMING(cutlass::float_e4m3_t, cutlass::float_e4m3_t, float, 64, 192, 32, "F32E4M3_m64n192k32");
    RUN_TIMING(cutlass::float_e4m3_t, cutlass::float_e4m3_t, float, 64, 224, 32, "F32E4M3_m64n224k32");
    RUN_TIMING(cutlass::float_e4m3_t, cutlass::float_e4m3_t, float, 64, 256, 32, "F32E4M3_m64n256k32");

    printf("\n  -- F16 accumulator, F16 inputs (K=16) --\n");
    RUN_TIMING(cutlass::half_t, cutlass::half_t, cutlass::half_t, 64,   8, 16, "F16F16_m64n8k16");
    RUN_TIMING(cutlass::half_t, cutlass::half_t, cutlass::half_t, 64,  16, 16, "F16F16_m64n16k16");
    RUN_TIMING(cutlass::half_t, cutlass::half_t, cutlass::half_t, 64,  32, 16, "F16F16_m64n32k16");
    RUN_TIMING(cutlass::half_t, cutlass::half_t, cutlass::half_t, 64,  64, 16, "F16F16_m64n64k16");
    RUN_TIMING(cutlass::half_t, cutlass::half_t, cutlass::half_t, 64,  96, 16, "F16F16_m64n96k16");
    RUN_TIMING(cutlass::half_t, cutlass::half_t, cutlass::half_t, 64, 128, 16, "F16F16_m64n128k16");
    RUN_TIMING(cutlass::half_t, cutlass::half_t, cutlass::half_t, 64, 192, 16, "F16F16_m64n192k16");
    RUN_TIMING(cutlass::half_t, cutlass::half_t, cutlass::half_t, 64, 224, 16, "F16F16_m64n224k16");
    RUN_TIMING(cutlass::half_t, cutlass::half_t, cutlass::half_t, 64, 256, 16, "F16F16_m64n256k16");

    printf("\n  -- S32 accumulator, S8 inputs (K=32) --\n");
    RUN_TIMING(int8_t, int8_t, int32_t, 64,   8, 32, "S32S8_m64n8k32");
    RUN_TIMING(int8_t, int8_t, int32_t, 64,  16, 32, "S32S8_m64n16k32");
    RUN_TIMING(int8_t, int8_t, int32_t, 64,  32, 32, "S32S8_m64n32k32");
    RUN_TIMING(int8_t, int8_t, int32_t, 64,  64, 32, "S32S8_m64n64k32");
    RUN_TIMING(int8_t, int8_t, int32_t, 64,  96, 32, "S32S8_m64n96k32");
    RUN_TIMING(int8_t, int8_t, int32_t, 64, 128, 32, "S32S8_m64n128k32");
    RUN_TIMING(int8_t, int8_t, int32_t, 64, 192, 32, "S32S8_m64n192k32");
    RUN_TIMING(int8_t, int8_t, int32_t, 64, 224, 32, "S32S8_m64n224k32");
    RUN_TIMING(int8_t, int8_t, int32_t, 64, 256, 32, "S32S8_m64n256k32");

    // -----------------------------------------------------------------------
    // Phase 1b: Pipelined 2-group overlap (F32F16, all N)
    // -----------------------------------------------------------------------
    printf("\n=== Phase 1b: Pipelined 2-group overlap (F32F16) ===\n");
    printf("  gap_after_wait1 ≈ 0 means group 2 completed during group 1's latency tail.\n");
    printf("  %-60s  cyc/mma   gap_after_wait1\n", "shape");

    FILE* pipeline_csv = fopen("wgmma_sweep_pipeline.csv", "w");
    if (pipeline_csv)
        fprintf(pipeline_csv, "shape,cyc_per_mma,gap_after_wait1_cycles\n");

    RUN_PIPELINED(cutlass::half_t, cutlass::half_t, float, 64,   8, 16, "F32F16_m64n8k16");
    RUN_PIPELINED(cutlass::half_t, cutlass::half_t, float, 64,  16, 16, "F32F16_m64n16k16");
    RUN_PIPELINED(cutlass::half_t, cutlass::half_t, float, 64,  32, 16, "F32F16_m64n32k16");
    RUN_PIPELINED(cutlass::half_t, cutlass::half_t, float, 64,  64, 16, "F32F16_m64n64k16");
    RUN_PIPELINED(cutlass::half_t, cutlass::half_t, float, 64,  96, 16, "F32F16_m64n96k16");
    RUN_PIPELINED(cutlass::half_t, cutlass::half_t, float, 64, 128, 16, "F32F16_m64n128k16");
    RUN_PIPELINED(cutlass::half_t, cutlass::half_t, float, 64, 192, 16, "F32F16_m64n192k16");
    RUN_PIPELINED(cutlass::half_t, cutlass::half_t, float, 64, 224, 16, "F32F16_m64n224k16");
    RUN_PIPELINED(cutlass::half_t, cutlass::half_t, float, 64, 256, 16, "F32F16_m64n256k16");

    if (pipeline_csv) fclose(pipeline_csv);
    printf("  Pipelined results -> wgmma_sweep_pipeline.csv\n");

    // -----------------------------------------------------------------------
    // Phase 4: RS timing sweep (before layout phases to group timing output)
    // -----------------------------------------------------------------------
    printf("\n=== Phase 4: RS timing sweep (register-A, smem-B) ===\n");
    printf("  %-60s  cyc/mma   II   derived_lat\n", "shape");

    printf("\n  -- F32 accumulator, F16 inputs RS (K=16) --\n");
    RUN_TIMING_RS(cutlass::half_t, cutlass::half_t, float, 64,   8, 16, "F32F16_m64n8k16");
    RUN_TIMING_RS(cutlass::half_t, cutlass::half_t, float, 64,  16, 16, "F32F16_m64n16k16");
    RUN_TIMING_RS(cutlass::half_t, cutlass::half_t, float, 64,  32, 16, "F32F16_m64n32k16");
    RUN_TIMING_RS(cutlass::half_t, cutlass::half_t, float, 64,  64, 16, "F32F16_m64n64k16");
    RUN_TIMING_RS(cutlass::half_t, cutlass::half_t, float, 64,  96, 16, "F32F16_m64n96k16");
    RUN_TIMING_RS(cutlass::half_t, cutlass::half_t, float, 64, 128, 16, "F32F16_m64n128k16");
    RUN_TIMING_RS(cutlass::half_t, cutlass::half_t, float, 64, 192, 16, "F32F16_m64n192k16");
    RUN_TIMING_RS(cutlass::half_t, cutlass::half_t, float, 64, 224, 16, "F32F16_m64n224k16");
    RUN_TIMING_RS(cutlass::half_t, cutlass::half_t, float, 64, 256, 16, "F32F16_m64n256k16");

    printf("\n  -- F32 accumulator, TF32 inputs RS (K=8) --\n");
    RUN_TIMING_RS(cutlass::tfloat32_t, cutlass::tfloat32_t, float, 64,   8,  8, "F32TF32_m64n8k8");
    RUN_TIMING_RS(cutlass::tfloat32_t, cutlass::tfloat32_t, float, 64,  16,  8, "F32TF32_m64n16k8");
    RUN_TIMING_RS(cutlass::tfloat32_t, cutlass::tfloat32_t, float, 64,  32,  8, "F32TF32_m64n32k8");
    RUN_TIMING_RS(cutlass::tfloat32_t, cutlass::tfloat32_t, float, 64,  64,  8, "F32TF32_m64n64k8");
    RUN_TIMING_RS(cutlass::tfloat32_t, cutlass::tfloat32_t, float, 64,  96,  8, "F32TF32_m64n96k8");
    RUN_TIMING_RS(cutlass::tfloat32_t, cutlass::tfloat32_t, float, 64, 128,  8, "F32TF32_m64n128k8");
    RUN_TIMING_RS(cutlass::tfloat32_t, cutlass::tfloat32_t, float, 64, 192,  8, "F32TF32_m64n192k8");
    RUN_TIMING_RS(cutlass::tfloat32_t, cutlass::tfloat32_t, float, 64, 224,  8, "F32TF32_m64n224k8");
    RUN_TIMING_RS(cutlass::tfloat32_t, cutlass::tfloat32_t, float, 64, 256,  8, "F32TF32_m64n256k8");

    printf("\n  -- F32 accumulator, BF16 inputs RS (K=16) --\n");
    RUN_TIMING_RS(cutlass::bfloat16_t, cutlass::bfloat16_t, float, 64,   8, 16, "F32BF16_m64n8k16");
    RUN_TIMING_RS(cutlass::bfloat16_t, cutlass::bfloat16_t, float, 64,  16, 16, "F32BF16_m64n16k16");
    RUN_TIMING_RS(cutlass::bfloat16_t, cutlass::bfloat16_t, float, 64,  32, 16, "F32BF16_m64n32k16");
    RUN_TIMING_RS(cutlass::bfloat16_t, cutlass::bfloat16_t, float, 64,  64, 16, "F32BF16_m64n64k16");
    RUN_TIMING_RS(cutlass::bfloat16_t, cutlass::bfloat16_t, float, 64,  96, 16, "F32BF16_m64n96k16");
    RUN_TIMING_RS(cutlass::bfloat16_t, cutlass::bfloat16_t, float, 64, 128, 16, "F32BF16_m64n128k16");
    RUN_TIMING_RS(cutlass::bfloat16_t, cutlass::bfloat16_t, float, 64, 192, 16, "F32BF16_m64n192k16");
    RUN_TIMING_RS(cutlass::bfloat16_t, cutlass::bfloat16_t, float, 64, 224, 16, "F32BF16_m64n224k16");
    RUN_TIMING_RS(cutlass::bfloat16_t, cutlass::bfloat16_t, float, 64, 256, 16, "F32BF16_m64n256k16");

    printf("\n  -- F16 accumulator, F16 inputs RS (K=16) --\n");
    RUN_TIMING_RS(cutlass::half_t, cutlass::half_t, cutlass::half_t, 64,   8, 16, "F16F16_m64n8k16");
    RUN_TIMING_RS(cutlass::half_t, cutlass::half_t, cutlass::half_t, 64,  16, 16, "F16F16_m64n16k16");
    RUN_TIMING_RS(cutlass::half_t, cutlass::half_t, cutlass::half_t, 64,  32, 16, "F16F16_m64n32k16");
    RUN_TIMING_RS(cutlass::half_t, cutlass::half_t, cutlass::half_t, 64,  64, 16, "F16F16_m64n64k16");
    RUN_TIMING_RS(cutlass::half_t, cutlass::half_t, cutlass::half_t, 64,  96, 16, "F16F16_m64n96k16");
    RUN_TIMING_RS(cutlass::half_t, cutlass::half_t, cutlass::half_t, 64, 128, 16, "F16F16_m64n128k16");
    RUN_TIMING_RS(cutlass::half_t, cutlass::half_t, cutlass::half_t, 64, 192, 16, "F16F16_m64n192k16");
    RUN_TIMING_RS(cutlass::half_t, cutlass::half_t, cutlass::half_t, 64, 224, 16, "F16F16_m64n224k16");
    RUN_TIMING_RS(cutlass::half_t, cutlass::half_t, cutlass::half_t, 64, 256, 16, "F16F16_m64n256k16");

    if (timing_csv) fclose(timing_csv);
    printf("\n  Timing results -> wgmma_sweep_timing.csv\n");

    // -----------------------------------------------------------------------
    // Phase 2: D-fragment column layout — all N, three accumulators
    // -----------------------------------------------------------------------
    printf("\n=== Phase 2: D-fragment column layout (B[k][n]=n => D=K*n) ===\n");
    printf("  Verifies d_frag_pos column formula for all N and three accumulator types.\n");
    printf("  S32S8 uses B[k][n]=n%%64 (int8 overflow guard), expected K*(col%%64).\n");

    printf("\n  -- F32 accumulator (F16 inputs) --\n");
    RUN_COL_LAYOUT(cutlass::half_t, cutlass::half_t, float, 64,   8, 16, "F32", "F32F16_m64n8k16");
    RUN_COL_LAYOUT(cutlass::half_t, cutlass::half_t, float, 64,  16, 16, "F32", "F32F16_m64n16k16");
    RUN_COL_LAYOUT(cutlass::half_t, cutlass::half_t, float, 64,  32, 16, "F32", "F32F16_m64n32k16");
    RUN_COL_LAYOUT(cutlass::half_t, cutlass::half_t, float, 64,  64, 16, "F32", "F32F16_m64n64k16");
    RUN_COL_LAYOUT(cutlass::half_t, cutlass::half_t, float, 64,  96, 16, "F32", "F32F16_m64n96k16");
    RUN_COL_LAYOUT(cutlass::half_t, cutlass::half_t, float, 64, 128, 16, "F32", "F32F16_m64n128k16");
    RUN_COL_LAYOUT(cutlass::half_t, cutlass::half_t, float, 64, 192, 16, "F32", "F32F16_m64n192k16");
    RUN_COL_LAYOUT(cutlass::half_t, cutlass::half_t, float, 64, 224, 16, "F32", "F32F16_m64n224k16");
    RUN_COL_LAYOUT(cutlass::half_t, cutlass::half_t, float, 64, 256, 16, "F32", "F32F16_m64n256k16");

    printf("\n  -- F16 accumulator (F16 inputs) — D_ELEMS=N/2, same d_frag_pos --\n");
    RUN_COL_LAYOUT(cutlass::half_t, cutlass::half_t, cutlass::half_t, 64,   8, 16, "F16", "F16F16_m64n8k16");
    RUN_COL_LAYOUT(cutlass::half_t, cutlass::half_t, cutlass::half_t, 64,  16, 16, "F16", "F16F16_m64n16k16");
    RUN_COL_LAYOUT(cutlass::half_t, cutlass::half_t, cutlass::half_t, 64,  32, 16, "F16", "F16F16_m64n32k16");
    RUN_COL_LAYOUT(cutlass::half_t, cutlass::half_t, cutlass::half_t, 64,  64, 16, "F16", "F16F16_m64n64k16");
    RUN_COL_LAYOUT(cutlass::half_t, cutlass::half_t, cutlass::half_t, 64,  96, 16, "F16", "F16F16_m64n96k16");
    RUN_COL_LAYOUT(cutlass::half_t, cutlass::half_t, cutlass::half_t, 64, 128, 16, "F16", "F16F16_m64n128k16");
    RUN_COL_LAYOUT(cutlass::half_t, cutlass::half_t, cutlass::half_t, 64, 192, 16, "F16", "F16F16_m64n192k16");
    RUN_COL_LAYOUT(cutlass::half_t, cutlass::half_t, cutlass::half_t, 64, 224, 16, "F16", "F16F16_m64n224k16");
    RUN_COL_LAYOUT(cutlass::half_t, cutlass::half_t, cutlass::half_t, 64, 256, 16, "F16", "F16F16_m64n256k16");

    printf("\n  -- S32 accumulator (S8 inputs) — B[k][n]=n%%64, expected K*(col%%64) --\n");
    RUN_COL_LAYOUT_MOD(int8_t, int8_t, int32_t, 64,   8, 32, 64, "S32", "S32S8_m64n8k32");
    RUN_COL_LAYOUT_MOD(int8_t, int8_t, int32_t, 64,  16, 32, 64, "S32", "S32S8_m64n16k32");
    RUN_COL_LAYOUT_MOD(int8_t, int8_t, int32_t, 64,  32, 32, 64, "S32", "S32S8_m64n32k32");
    RUN_COL_LAYOUT_MOD(int8_t, int8_t, int32_t, 64,  64, 32, 64, "S32", "S32S8_m64n64k32");
    RUN_COL_LAYOUT_MOD(int8_t, int8_t, int32_t, 64,  96, 32, 64, "S32", "S32S8_m64n96k32");
    RUN_COL_LAYOUT_MOD(int8_t, int8_t, int32_t, 64, 128, 32, 64, "S32", "S32S8_m64n128k32");
    RUN_COL_LAYOUT_MOD(int8_t, int8_t, int32_t, 64, 192, 32, 64, "S32", "S32S8_m64n192k32");
    RUN_COL_LAYOUT_MOD(int8_t, int8_t, int32_t, 64, 224, 32, 64, "S32", "S32S8_m64n224k32");
    RUN_COL_LAYOUT_MOD(int8_t, int8_t, int32_t, 64, 256, 32, 64, "S32", "S32S8_m64n256k32");

    printf("\n  Column layout dumps -> wgmma_sweep_layout_col<F32|F16|S32>N<N>.txt\n");

    // -----------------------------------------------------------------------
    // Phase 3: smem_A row layout — three K values
    // A[m][k]=m, B=1 => D=K*m. Independently verifies both d_frag_pos row
    // formula AND smem_A K-interleaved physical layout.
    // -----------------------------------------------------------------------
    printf("\n=== Phase 3: smem_A row layout (A[m][k]=m, B=1 => D=K*m) ===\n");
    printf("  Verifies smem_A layout for different K (element size) values.\n");

    printf("\n  -- F16 inputs K=16 (2-byte elements, Swizzle<0,4,3>) --\n");
    RUN_ROW_LAYOUT(cutlass::half_t, cutlass::half_t, float, 64,  16, 16, "F16", "F32F16_m64n16k16");
    RUN_ROW_LAYOUT(cutlass::half_t, cutlass::half_t, float, 64,  32, 16, "F16", "F32F16_m64n32k16");
    RUN_ROW_LAYOUT(cutlass::half_t, cutlass::half_t, float, 64,  64, 16, "F16", "F32F16_m64n64k16");
    RUN_ROW_LAYOUT(cutlass::half_t, cutlass::half_t, float, 64, 128, 16, "F16", "F32F16_m64n128k16");
    RUN_ROW_LAYOUT(cutlass::half_t, cutlass::half_t, float, 64, 256, 16, "F16", "F32F16_m64n256k16");

    printf("\n  -- TF32 inputs K=8 (4-byte elements) — different K interleaving --\n");
    RUN_ROW_LAYOUT(cutlass::tfloat32_t, cutlass::tfloat32_t, float, 64,  16,  8, "TF32", "F32TF32_m64n16k8");
    RUN_ROW_LAYOUT(cutlass::tfloat32_t, cutlass::tfloat32_t, float, 64,  32,  8, "TF32", "F32TF32_m64n32k8");
    RUN_ROW_LAYOUT(cutlass::tfloat32_t, cutlass::tfloat32_t, float, 64,  64,  8, "TF32", "F32TF32_m64n64k8");
    RUN_ROW_LAYOUT(cutlass::tfloat32_t, cutlass::tfloat32_t, float, 64, 128,  8, "TF32", "F32TF32_m64n128k8");

    printf("\n  -- S8 inputs K=32 (1-byte elements) — different K interleaving --\n");
    RUN_ROW_LAYOUT(int8_t, int8_t, int32_t, 64,  16, 32, "S8", "S32S8_m64n16k32");
    RUN_ROW_LAYOUT(int8_t, int8_t, int32_t, 64,  32, 32, "S8", "S32S8_m64n32k32");
    RUN_ROW_LAYOUT(int8_t, int8_t, int32_t, 64,  64, 32, "S8", "S32S8_m64n64k32");

    printf("\n  Row layout dumps -> wgmma_sweep_layout_row<F16|TF32|S8>N<N>.txt\n");

    printf("\nDone.\n");
    return 0;
}
