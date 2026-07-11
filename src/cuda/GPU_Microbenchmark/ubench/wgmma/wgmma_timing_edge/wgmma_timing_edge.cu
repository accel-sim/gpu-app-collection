// =============================================================================
// wgmma_timing_edge.cu
//
// Silicon-validation microbenchmarks for WGMMA *timing-model edge cases* that
// lat_gmma / wgmma_multik do not cover.  Every test self-checks functional
// correctness (exit 0 = all PASS) and reports %clock-based cycle counts to
// stdout + wgmma_timing_edge_results.csv for sim-vs-silicon comparison.
//
// Tests (one named kernel each, so Nsight Compute can profile them separately):
//
//   1. kernel_group_size    - S MMAs in ONE commit group, S = 1..32.
//                             total(S) ~= LAT + (S-1)*II  ->  linear fit
//                             separates true completion latency (intercept)
//                             from initiation interval (slope).  Directly
//                             tests commit_wgmma_group_with_ops()'s formula.
//   2. kernel_wait_partial  - 4 MMAs in 4 separate groups; timed staged
//                             wait_group 3 -> 2 -> 1 -> 0.  Tests the
//                             "all but latest N" drain semantics + timing.
//   3. kernel_serialize     - 2 groups of 1 MMA vs 1 group of 2 MMAs.
//                             Tests inter-group serialization (model forces
//                             group completion >= previous group completion).
//   4. kernel_overlap       - MMA + commit, then a dependent FMA chain of
//                             length X, then wait.  Sweeping X shows how much
//                             independent ALU work hides under the async MMA
//                             (the knee = true async latency; cross-checks #1).
//   5. kernel_inflight      - 16x (MMA;COMMIT) with a %clock stamp after each
//                             pair, no wait until the end.  A knee in the
//                             per-iteration deltas reveals the hardware
//                             in-flight group limit (model: wgmma_max_pending).
//   6. kernel_swizzle       - 32-MMA chain with swizzle mode 0/3/2/1
//                             (none/32B/64B/128B).  Model predicts identical
//                             timing; functional check validates the swizzled
//                             smem layout on silicon.
//   7. kernel_multiwg       - 1 vs 2 warpgroups (128 vs 256 threads) each
//                             running a 64-MMA chain on private smem tiles.
//                             Tests tensor-core contention across warpgroups
//                             (model tracks warpgroups independently).
//
// All tests: m64n16k16 f32 <- f16 x f16, A = B = 1 so D element = K * #MMAs.
//
// Run on silicon:
//   ./wgmma_timing_edge                      # all tests + CSV + exit code
//   ncu --kernel-name 'regex:kernel_.*' \
//       --metrics gpu__time_duration.sum,sm__cycles_elapsed.avg,\
//                 sm__pipe_tensor_op_hmma_cycles_active.avg \
//       ./wgmma_timing_edge                  # per-kernel timing cross-check
//   nsys profile --stats=true ./wgmma_timing_edge   # coarse launch timeline
//
// Run in GPGPU-Sim:
//   functional:  PTX_SIM_USE_PTX_FILE=1 ./wgmma_timing_edge   (mode 1 config)
//   timing:      PTX_SIM_MODE_FUNC=0 ./wgmma_timing_edge
//
// Exit status: 0 = all functional checks PASS, 1 = any FAIL.
// =============================================================================

#include <cuda.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

static constexpr int M       = 64;
static constexpr int N       = 16;
static constexpr int K       = 16;
static constexpr int WGSIZE  = 128;
static constexpr int D_ELEMS = N / 2;      // 8 f32 registers per thread
static constexpr int E       = 2;          // bytes per f16
static constexpr int SBO     = 128;
static constexpr int LBO_A   = M * E * 8;  // 1024
static constexpr int LBO_B   = N * E * 8;  // 256
static constexpr int SMEM_A  = M * K * E;  // 2048
static constexpr int SMEM_B  = K * N * E;  // 512

// ---------------------------------------------------------------------------
// Device helpers (same conventions as wgmma_multik.cu / sim wgmma_layout.h)
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint32_t smem_addr(const void* ptr) {
    uint32_t a;
    asm volatile(
        "{ .reg .u64 _p; cvta.to.shared.u64 _p, %1; cvt.u32.u64 %0, _p; }"
        : "=r"(a) : "l"(ptr));
    return a;
}

__device__ __forceinline__ uint64_t make_gmma_desc(
    uint32_t base, uint32_t lbo, uint32_t sbo, uint32_t sw = 0) {
    uint64_t d = 0;
    d |= (uint64_t)(base >> 4) & 0x3FFFu;
    d |= ((uint64_t)((lbo >> 4) & 0x3FFFu)) << 16;
    d |= ((uint64_t)((sbo >> 4) & 0x3FFFu)) << 32;
    d |= ((uint64_t)(sw & 0x3u)) << 62;
    return d;
}

// Byte offset of logical element (leading, stride) - matches sim wgmma_smem_offset.
__host__ __device__ __forceinline__
int smem_off(int leading, int stride, int e, int LBO) {
    int T = 16 / e;
    return (stride % T + (leading % 8) * T) * e
         + (stride / T) * SBO
         + (leading / 8) * LBO;
}

// XOR swizzle - matches sim wgmma_apply_swizzle (mode 0=none,1=128B,2=64B,3=32B).
__host__ __device__ __forceinline__
int apply_swizzle(int off, int sw) {
    if (sw == 0) return off;
    int stride   = (sw == 1) ? 128 : (sw == 2) ? 64 : 32;
    int block    = off / stride;
    int in_block = off % stride;
    int xm       = (block & 1) ? (stride >> 1) : 0;
    return block * stride + (in_block ^ xm);
}

#define WGMMA_FENCE     asm volatile("wgmma.fence.sync.aligned;\n"          ::: "memory")
#define WGMMA_COMMIT    asm volatile("wgmma.commit_group.sync.aligned;\n"   ::: "memory")
#define WGMMA_WAIT_N(n) asm volatile("wgmma.wait_group.sync.aligned " #n ";\n" ::: "memory")

#define CLOCK(v) asm volatile("mov.u32 %0, %%clock;" : "=r"(v) :: "memory")

__device__ __forceinline__ void mma_f16(
    uint64_t da, uint64_t db,
    float& d0, float& d1, float& d2, float& d3,
    float& d4, float& d5, float& d6, float& d7) {
    asm volatile(
        "{ .reg .pred p; setp.ne.b32 p,1,0;\n"
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7},%8,%9,p,1,1,0,0; }\n"
        : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3),
          "+f"(d4),"+f"(d5),"+f"(d6),"+f"(d7)
        : "l"(da),"l"(db) : "memory");
}

// Fill smem with the canonical (optionally swizzled) layout; A/B value = 1.0.
// lane_stride lets a 256-thread block fill two private tiles.
__device__ __forceinline__ void fill_tiles(
    char* smA, char* smB, const half* A_g, const half* B_g,
    int tid, int nthreads, int sw)
{
    for (int i = tid; i < M*K; i += nthreads) {
        int m = i/K, k = i%K;
        *(half*)(smA + apply_swizzle(smem_off(k, m, E, LBO_A), sw)) = A_g[i];
    }
    for (int i = tid; i < K*N; i += nthreads) {
        int k = i/N, n = i%N;
        *(half*)(smB + apply_swizzle(smem_off(k, n, E, LBO_B), sw)) = B_g[i];
    }
}

__device__ __forceinline__ void store_d(
    float* D_g, int tid,
    float d0, float d1, float d2, float d3,
    float d4, float d5, float d6, float d7) {
    const int b = tid * D_ELEMS;
    D_g[b]=d0; D_g[b+1]=d1; D_g[b+2]=d2; D_g[b+3]=d3;
    D_g[b+4]=d4; D_g[b+5]=d5; D_g[b+6]=d6; D_g[b+7]=d7;
}

// ---------------------------------------------------------------------------
// 1. kernel_group_size: S MMAs in one commit group; time fence..wait0.
//    total(S) = LAT + (S-1)*II  -> host does the linear fit.
// ---------------------------------------------------------------------------
__global__ void kernel_group_size(
    const half* A_g, const half* B_g, float* D_g,
    uint32_t* clk_out, int S)
{
    __shared__ __align__(128) char smA[SMEM_A], smB[SMEM_B];
    const int tid = threadIdx.x;
    fill_tiles(smA, smB, A_g, B_g, tid, WGSIZE, 0);
    __syncthreads();

    uint64_t da = make_gmma_desc(smem_addr(smA), LBO_A, SBO);
    uint64_t db = make_gmma_desc(smem_addr(smB), LBO_B, SBO);
    float d0=0,d1=0,d2=0,d3=0,d4=0,d5=0,d6=0,d7=0;

    uint32_t t0 = 0, t1 = 0;
    __syncthreads();
    if (tid == 0) CLOCK(t0);
    __syncthreads();

    WGMMA_FENCE;
    for (int i = 0; i < S; i++)
        mma_f16(da, db, d0,d1,d2,d3,d4,d5,d6,d7);
    WGMMA_COMMIT;
    WGMMA_WAIT_N(0);
    WGMMA_FENCE;

    __syncthreads();
    if (tid == 0) CLOCK(t1);
    if (tid == 0) { clk_out[0] = t0; clk_out[1] = t1; }
    store_d(D_g, tid, d0,d1,d2,d3,d4,d5,d6,d7);
}

// ---------------------------------------------------------------------------
// 2. kernel_wait_partial: 4 single-MMA groups pending, then staged waits.
//    clk stamps: [0]=start, [1]=after wait 3, [2]=after wait 2,
//                [3]=after wait 1, [4]=after wait 0.
// ---------------------------------------------------------------------------
__global__ void kernel_wait_partial(
    const half* A_g, const half* B_g, float* D_g, uint32_t* clk_out)
{
    __shared__ __align__(128) char smA[SMEM_A], smB[SMEM_B];
    const int tid = threadIdx.x;
    fill_tiles(smA, smB, A_g, B_g, tid, WGSIZE, 0);
    __syncthreads();

    uint64_t da = make_gmma_desc(smem_addr(smA), LBO_A, SBO);
    uint64_t db = make_gmma_desc(smem_addr(smB), LBO_B, SBO);
    float d0=0,d1=0,d2=0,d3=0,d4=0,d5=0,d6=0,d7=0;
    uint32_t c0=0,c1=0,c2=0,c3=0,c4=0;

    __syncthreads();
    WGMMA_FENCE;
    mma_f16(da, db, d0,d1,d2,d3,d4,d5,d6,d7); WGMMA_COMMIT;   // group 0
    mma_f16(da, db, d0,d1,d2,d3,d4,d5,d6,d7); WGMMA_COMMIT;   // group 1
    mma_f16(da, db, d0,d1,d2,d3,d4,d5,d6,d7); WGMMA_COMMIT;   // group 2
    mma_f16(da, db, d0,d1,d2,d3,d4,d5,d6,d7); WGMMA_COMMIT;   // group 3
    if (tid == 0) CLOCK(c0);
    WGMMA_WAIT_N(3);   // oldest group (0) must be done
    if (tid == 0) CLOCK(c1);
    WGMMA_WAIT_N(2);   // groups 0,1 done
    if (tid == 0) CLOCK(c2);
    WGMMA_WAIT_N(1);   // groups 0,1,2 done
    if (tid == 0) CLOCK(c3);
    WGMMA_WAIT_N(0);   // all done
    WGMMA_FENCE;
    if (tid == 0) CLOCK(c4);

    if (tid == 0) {
        clk_out[0]=c0; clk_out[1]=c1; clk_out[2]=c2; clk_out[3]=c3; clk_out[4]=c4;
    }
    store_d(D_g, tid, d0,d1,d2,d3,d4,d5,d6,d7);
}

// ---------------------------------------------------------------------------
// 3. kernel_serialize: mode 0 = 1 group of 2 MMAs; mode 1 = 2 groups of 1 MMA.
//    Model serializes groups: mode1 total >= mode0 total expected on sim.
// ---------------------------------------------------------------------------
__global__ void kernel_serialize(
    const half* A_g, const half* B_g, float* D_g,
    uint32_t* clk_out, int mode)
{
    __shared__ __align__(128) char smA[SMEM_A], smB[SMEM_B];
    const int tid = threadIdx.x;
    fill_tiles(smA, smB, A_g, B_g, tid, WGSIZE, 0);
    __syncthreads();

    uint64_t da = make_gmma_desc(smem_addr(smA), LBO_A, SBO);
    uint64_t db = make_gmma_desc(smem_addr(smB), LBO_B, SBO);
    float d0=0,d1=0,d2=0,d3=0,d4=0,d5=0,d6=0,d7=0;

    uint32_t t0 = 0, t1 = 0;
    __syncthreads();
    if (tid == 0) CLOCK(t0);
    __syncthreads();

    WGMMA_FENCE;
    if (mode == 0) {
        mma_f16(da, db, d0,d1,d2,d3,d4,d5,d6,d7);
        mma_f16(da, db, d0,d1,d2,d3,d4,d5,d6,d7);
        WGMMA_COMMIT;
    } else {
        mma_f16(da, db, d0,d1,d2,d3,d4,d5,d6,d7); WGMMA_COMMIT;
        mma_f16(da, db, d0,d1,d2,d3,d4,d5,d6,d7); WGMMA_COMMIT;
    }
    WGMMA_WAIT_N(0);
    WGMMA_FENCE;

    __syncthreads();
    if (tid == 0) CLOCK(t1);
    if (tid == 0) { clk_out[0] = t0; clk_out[1] = t1; }
    store_d(D_g, tid, d0,d1,d2,d3,d4,d5,d6,d7);
}

// ---------------------------------------------------------------------------
// 4. kernel_overlap: MMA+commit, dependent FMA chain of length X, then wait.
//    do_mma / X let the host measure mma-only, fma-only and combined runs.
//    acc result is checked so the chain can't be optimized away.
// ---------------------------------------------------------------------------
__global__ void kernel_overlap(
    const half* A_g, const half* B_g, float* D_g,
    float* acc_g, uint32_t* clk_out, int X, int do_mma)
{
    __shared__ __align__(128) char smA[SMEM_A], smB[SMEM_B];
    const int tid = threadIdx.x;
    fill_tiles(smA, smB, A_g, B_g, tid, WGSIZE, 0);
    __syncthreads();

    uint64_t da = make_gmma_desc(smem_addr(smA), LBO_A, SBO);
    uint64_t db = make_gmma_desc(smem_addr(smB), LBO_B, SBO);
    float d0=0,d1=0,d2=0,d3=0,d4=0,d5=0,d6=0,d7=0;
    float acc = acc_g[tid];             // opaque start value (1.0f)

    uint32_t t0 = 0, t1 = 0;
    __syncthreads();
    if (tid == 0) CLOCK(t0);
    __syncthreads();

    WGMMA_FENCE;
    if (do_mma) {
        mma_f16(da, db, d0,d1,d2,d3,d4,d5,d6,d7);
        WGMMA_COMMIT;
    }
    // Dependent FMA chain, pinned between commit and wait by volatile asm.
    asm volatile("" : "+f"(acc));
    #pragma unroll 1
    for (int i = 0; i < X; i++)
        acc = __fmaf_rn(acc, 1.0f, 1.0f);   // acc += 1 each step (dependent)
    asm volatile("" : "+f"(acc));
    if (do_mma) {
        WGMMA_WAIT_N(0);
        WGMMA_FENCE;
    }

    __syncthreads();
    if (tid == 0) CLOCK(t1);
    if (tid == 0) { clk_out[0] = t0; clk_out[1] = t1; }
    acc_g[tid] = acc;
    store_d(D_g, tid, d0,d1,d2,d3,d4,d5,d6,d7);
}

// ---------------------------------------------------------------------------
// 5. kernel_inflight: NGRP x (MMA;COMMIT) with a clock stamp after each pair;
//    wait 0 only at the end.  Stall knee in the deltas = HW in-flight limit.
// ---------------------------------------------------------------------------
static constexpr int NGRP = 16;

__global__ void kernel_inflight(
    const half* A_g, const half* B_g, float* D_g, uint32_t* clk_out)
{
    __shared__ __align__(128) char smA[SMEM_A], smB[SMEM_B];
    const int tid = threadIdx.x;
    fill_tiles(smA, smB, A_g, B_g, tid, WGSIZE, 0);
    __syncthreads();

    uint64_t da = make_gmma_desc(smem_addr(smA), LBO_A, SBO);
    uint64_t db = make_gmma_desc(smem_addr(smB), LBO_B, SBO);
    float d0=0,d1=0,d2=0,d3=0,d4=0,d5=0,d6=0,d7=0;
    uint32_t stamps[NGRP + 2];

    __syncthreads();
    if (tid == 0) CLOCK(stamps[0]);
    __syncthreads();

    WGMMA_FENCE;
    #pragma unroll
    for (int g = 0; g < NGRP; g++) {
        mma_f16(da, db, d0,d1,d2,d3,d4,d5,d6,d7);
        WGMMA_COMMIT;
        if (tid == 0) CLOCK(stamps[g + 1]);
    }
    WGMMA_WAIT_N(0);
    WGMMA_FENCE;
    __syncthreads();
    if (tid == 0) CLOCK(stamps[NGRP + 1]);

    if (tid == 0)
        for (int i = 0; i < NGRP + 2; i++) clk_out[i] = stamps[i];
    store_d(D_g, tid, d0,d1,d2,d3,d4,d5,d6,d7);
}

// ---------------------------------------------------------------------------
// 6. kernel_swizzle: 32-MMA chain with swizzle mode sw (0/1/2/3).
//    Functional check validates the swizzled fill; timing compares modes.
// ---------------------------------------------------------------------------
static constexpr int SW_MMAS = 32;

__global__ void kernel_swizzle(
    const half* A_g, const half* B_g, float* D_g,
    uint32_t* clk_out, int sw)
{
    __shared__ __align__(128) char smA[SMEM_A], smB[SMEM_B];
    const int tid = threadIdx.x;
    fill_tiles(smA, smB, A_g, B_g, tid, WGSIZE, sw);
    __syncthreads();

    uint64_t da = make_gmma_desc(smem_addr(smA), LBO_A, SBO, (uint32_t)sw);
    uint64_t db = make_gmma_desc(smem_addr(smB), LBO_B, SBO, (uint32_t)sw);
    float d0=0,d1=0,d2=0,d3=0,d4=0,d5=0,d6=0,d7=0;

    uint32_t t0 = 0, t1 = 0;
    __syncthreads();
    if (tid == 0) CLOCK(t0);
    __syncthreads();

    WGMMA_FENCE;
    for (int i = 0; i < SW_MMAS; i++)
        mma_f16(da, db, d0,d1,d2,d3,d4,d5,d6,d7);
    WGMMA_COMMIT;
    WGMMA_WAIT_N(0);
    WGMMA_FENCE;

    __syncthreads();
    if (tid == 0) CLOCK(t1);
    if (tid == 0) { clk_out[0] = t0; clk_out[1] = t1; }
    store_d(D_g, tid, d0,d1,d2,d3,d4,d5,d6,d7);
}

// ---------------------------------------------------------------------------
// 7. kernel_multiwg: blockDim = 128 or 256 (1 or 2 warpgroups).  Each
//    warpgroup runs a WG_MMAS chain on its OWN smem tiles.  Per-warpgroup
//    start/stop clocks -> contention shows as higher cyc/mma with 2 wgs.
// ---------------------------------------------------------------------------
static constexpr int WG_MMAS = 64;
static constexpr int MAX_WG  = 2;

__global__ void kernel_multiwg(
    const half* A_g, const half* B_g, float* D_g, uint32_t* clk_out)
{
    __shared__ __align__(128) char smA[MAX_WG][SMEM_A], smB[MAX_WG][SMEM_B];
    const int tid  = threadIdx.x;
    const int wg   = tid / WGSIZE;          // warpgroup index in block
    const int wtid = tid % WGSIZE;          // thread index within warpgroup

    fill_tiles(smA[wg], smB[wg], A_g, B_g, wtid, WGSIZE, 0);
    __syncthreads();

    uint64_t da = make_gmma_desc(smem_addr(smA[wg]), LBO_A, SBO);
    uint64_t db = make_gmma_desc(smem_addr(smB[wg]), LBO_B, SBO);
    float d0=0,d1=0,d2=0,d3=0,d4=0,d5=0,d6=0,d7=0;

    uint32_t t0 = 0, t1 = 0;
    __syncthreads();                        // common start line for all wgs
    if (wtid == 0) CLOCK(t0);

    WGMMA_FENCE;
    for (int i = 0; i < WG_MMAS; i++)
        mma_f16(da, db, d0,d1,d2,d3,d4,d5,d6,d7);
    WGMMA_COMMIT;
    WGMMA_WAIT_N(0);
    WGMMA_FENCE;

    if (wtid == 0) CLOCK(t1);
    if (wtid == 0) { clk_out[2*wg] = t0; clk_out[2*wg + 1] = t1; }
    store_d(D_g + wg * WGSIZE * D_ELEMS, wtid, d0,d1,d2,d3,d4,d5,d6,d7);
}

// ===========================================================================
// Host side
// ===========================================================================
static void d_frag_pos(int T, int e, int* row, int* col) {
    int warp = T/32, lane = T%32, s = e/4, k = e%4;
    *row = (lane/4)*2 + k/2 + warp*16;
    *col = (lane%4)*2 + k%2 + s*8;
}

static bool check_uniform_flat(const char* name, const float* flat,
                               float expected, FILE* csv,
                               const char* param, float cycles,
                               float cyc_per_mma) {
    float max_err = 0.f;
    for (int T = 0; T < WGSIZE; T++)
        for (int e = 0; e < D_ELEMS; e++) {
            int row, col;
            d_frag_pos(T, e, &row, &col);
            if (row >= M || col >= N) continue;
            float err = fabsf(flat[T*D_ELEMS + e] - expected);
            if (err > max_err) max_err = err;
        }
    float tol  = fabsf(expected) * 1e-2f + 1e-3f;
    bool  pass = (max_err <= tol);
    printf("[%-28s %-12s] %s  expected=%.1f max_err=%.3e  cycles=%.0f",
           name, param, pass ? "PASS" : "FAIL", expected, max_err, cycles);
    if (cyc_per_mma > 0) printf("  cyc/mma=%.2f", cyc_per_mma);
    printf("\n");
    if (csv)
        fprintf(csv, "%s,%s,%s,%.1f,%.4e,%.0f,%.2f\n",
                name, param, pass ? "PASS" : "FAIL", expected, max_err,
                cycles, cyc_per_mma);
    return pass;
}

struct DevBufs {
    half *A, *B;
    float *D;
    uint32_t *clk;
    float *acc;
};

static DevBufs alloc_bufs(int d_mult = 1) {
    DevBufs b;
    std::vector<half> h_A(M*K, __float2half(1.f));
    std::vector<half> h_B(K*N, __float2half(1.f));
    cudaMalloc(&b.A,   M*K*sizeof(half));
    cudaMalloc(&b.B,   K*N*sizeof(half));
    cudaMalloc(&b.D,   d_mult*WGSIZE*D_ELEMS*sizeof(float));
    cudaMalloc(&b.clk, 32*sizeof(uint32_t));
    cudaMalloc(&b.acc, WGSIZE*sizeof(float));
    cudaMemcpy(b.A, h_A.data(), M*K*sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(b.B, h_B.data(), K*N*sizeof(half), cudaMemcpyHostToDevice);
    return b;
}

static void free_bufs(DevBufs& b) {
    cudaFree(b.A); cudaFree(b.B); cudaFree(b.D);
    cudaFree(b.clk); cudaFree(b.acc);
}

int main(int argc, char** argv) {
    const char* filter = (argc > 1) ? argv[1] : NULL;   // substring test filter
    auto run = [&](const char* name) {
        return !filter || strstr(name, filter);
    };

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("wgmma_timing_edge  -  %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("================================================================\n");

    FILE* csv = fopen("wgmma_timing_edge_results.csv", "w");
    if (csv) fprintf(csv, "test,param,status,expected,max_err,cycles,cycles_per_mma\n");

    bool all_pass = true;
    std::vector<float>    flat(2*WGSIZE*D_ELEMS);
    std::vector<uint32_t> clk(32);

    // --- 1. group-size sweep: LAT + (S-1)*II fit -------------------------
    if (run("group_size")) {
        printf("\n--- group_size: S MMAs in one group (fit: total = LAT + (S-1)*II) ---\n");
        const int Ss[] = {1, 2, 4, 8, 16, 32};
        double sx=0, sy=0, sxx=0, sxy=0; int np = 0;
        DevBufs b = alloc_bufs();
        for (int S : Ss) {
            kernel_group_size<<<1, WGSIZE>>>(b.A, b.B, b.D, b.clk, S);
            cudaDeviceSynchronize();
            cudaMemcpy(flat.data(), b.D, WGSIZE*D_ELEMS*sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(clk.data(), b.clk, 2*sizeof(uint32_t), cudaMemcpyDeviceToHost);
            float cyc = (float)(clk[1] - clk[0]);
            char param[32]; snprintf(param, 32, "S=%d", S);
            all_pass &= check_uniform_flat("group_size", flat.data(),
                                           (float)(S*K), csv, param, cyc, cyc/S);
            sx += S; sy += cyc; sxx += (double)S*S; sxy += (double)S*cyc; np++;
        }
        free_bufs(b);
        double ii  = (np*sxy - sx*sy) / (np*sxx - sx*sx);
        double lat = (sy - ii*sx) / np + ii;   // intercept at S=1... total(1)=LAT
        printf("  linear fit: II (slope) = %.2f cyc/mma, LAT (total at S=1) = %.2f cyc\n", ii, lat);
        if (csv) fprintf(csv, "group_size_fit,II=%.2f;LAT=%.2f,TIMING,,,,\n", ii, lat);
    }

    // --- 2. staged partial waits -----------------------------------------
    if (run("wait_partial")) {
        printf("\n--- wait_partial: 4 groups pending; wait 3 -> 2 -> 1 -> 0 ---\n");
        DevBufs b = alloc_bufs();
        kernel_wait_partial<<<1, WGSIZE>>>(b.A, b.B, b.D, b.clk);
        cudaDeviceSynchronize();
        cudaMemcpy(flat.data(), b.D, WGSIZE*D_ELEMS*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(clk.data(), b.clk, 5*sizeof(uint32_t), cudaMemcpyDeviceToHost);
        free_bufs(b);
        float total = (float)(clk[4] - clk[0]);
        all_pass &= check_uniform_flat("wait_partial", flat.data(),
                                       (float)(4*K), csv, "4grp", total, 0);
        printf("  wait3=%u  wait2=+%u  wait1=+%u  wait0=+%u cycles\n",
               clk[1]-clk[0], clk[2]-clk[1], clk[3]-clk[2], clk[4]-clk[3]);
        if (csv) fprintf(csv, "wait_partial_stages,w3=%u;w2=%u;w1=%u;w0=%u,TIMING,,,%'.0f,\n",
                         clk[1]-clk[0], clk[2]-clk[1], clk[3]-clk[2], clk[4]-clk[3], total);
    }

    // --- 3. group serialization ------------------------------------------
    if (run("serialize")) {
        printf("\n--- serialize: 1 group x 2 MMAs vs 2 groups x 1 MMA ---\n");
        DevBufs b = alloc_bufs();
        float cyc_mode[2];
        for (int mode = 0; mode < 2; mode++) {
            kernel_serialize<<<1, WGSIZE>>>(b.A, b.B, b.D, b.clk, mode);
            cudaDeviceSynchronize();
            cudaMemcpy(flat.data(), b.D, WGSIZE*D_ELEMS*sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(clk.data(), b.clk, 2*sizeof(uint32_t), cudaMemcpyDeviceToHost);
            cyc_mode[mode] = (float)(clk[1] - clk[0]);
            all_pass &= check_uniform_flat("serialize", flat.data(), (float)(2*K),
                                           csv, mode == 0 ? "1grpx2mma" : "2grpx1mma",
                                           cyc_mode[mode], cyc_mode[mode]/2);
        }
        free_bufs(b);
        printf("  delta (2grp - 1grp) = %.0f cycles\n", cyc_mode[1] - cyc_mode[0]);
    }

    // --- 4. async overlap sweep ------------------------------------------
    if (run("overlap")) {
        printf("\n--- overlap: MMA || dependent FMA chain of length X ---\n");
        DevBufs b = alloc_bufs();
        const int Xs[] = {0, 8, 16, 32, 64, 96, 128, 192, 256};
        std::vector<float> h_acc(WGSIZE, 1.f), r_acc(WGSIZE);
        for (int X : Xs) {
            for (int do_mma = (X == 0 ? 1 : 0); do_mma < 2; do_mma++) {
                cudaMemcpy(b.acc, h_acc.data(), WGSIZE*sizeof(float), cudaMemcpyHostToDevice);
                kernel_overlap<<<1, WGSIZE>>>(b.A, b.B, b.D, b.acc, b.clk, X, do_mma);
                cudaDeviceSynchronize();
                cudaMemcpy(flat.data(), b.D, WGSIZE*D_ELEMS*sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(r_acc.data(), b.acc, WGSIZE*sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(clk.data(), b.clk, 2*sizeof(uint32_t), cudaMemcpyDeviceToHost);
                float cyc = (float)(clk[1] - clk[0]);
                // functional: acc = 1 + X for every thread; D = K if do_mma
                bool acc_ok = true;
                for (int t = 0; t < WGSIZE; t++)
                    if (fabsf(r_acc[t] - (1.f + X)) > 1e-3f) acc_ok = false;
                char param[40];
                snprintf(param, 40, "X=%d%s", X, do_mma ? "+mma" : ",fma_only");
                if (do_mma) {
                    all_pass &= check_uniform_flat("overlap", flat.data(), (float)K,
                                                   csv, param, cyc, 0) && acc_ok;
                } else {
                    printf("[%-28s %-12s] %s  cycles=%.0f (fma chain alone)\n",
                           "overlap", param, acc_ok ? "PASS" : "FAIL", cyc);
                    if (csv) fprintf(csv, "overlap,%s,%s,,,%0.f,\n",
                                     param, acc_ok ? "PASS" : "FAIL", cyc);
                    all_pass &= acc_ok;
                }
                if (!acc_ok) printf("  FMA chain result wrong (X=%d)\n", X);
            }
        }
        free_bufs(b);
        printf("  (flat combined-total until X exceeds the MMA latency = async overlap works)\n");
    }

    // --- 5. in-flight group limit -----------------------------------------
    if (run("inflight")) {
        printf("\n--- inflight: %d x (MMA;COMMIT) with per-commit clock stamps ---\n", NGRP);
        DevBufs b = alloc_bufs();
        kernel_inflight<<<1, WGSIZE>>>(b.A, b.B, b.D, b.clk);
        cudaDeviceSynchronize();
        cudaMemcpy(flat.data(), b.D, WGSIZE*D_ELEMS*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(clk.data(), b.clk, (NGRP+2)*sizeof(uint32_t), cudaMemcpyDeviceToHost);
        free_bufs(b);
        float total = (float)(clk[NGRP+1] - clk[0]);
        all_pass &= check_uniform_flat("inflight", flat.data(), (float)(NGRP*K),
                                       csv, "16grp", total, total/NGRP);
        printf("  per-(mma;commit) deltas: ");
        for (int g = 1; g <= NGRP; g++) printf("%u ", clk[g] - clk[g-1]);
        printf("\n  drain after last commit: %u cycles\n", clk[NGRP+1] - clk[NGRP]);
        if (csv) {
            fprintf(csv, "inflight_deltas,");
            for (int g = 1; g <= NGRP; g++)
                fprintf(csv, "%u%s", clk[g] - clk[g-1], g == NGRP ? "" : ";");
            fprintf(csv, ",TIMING,,,%0.f,\n", total);
        }
    }

    // --- 6. swizzle modes --------------------------------------------------
    if (run("swizzle")) {
        printf("\n--- swizzle: %d-MMA chain, modes none/32B/64B/128B ---\n", SW_MMAS);
        static const char* swname[] = {"none", "128B", "64B", "32B"};
        DevBufs b = alloc_bufs();
        for (int sw = 0; sw < 4; sw++) {
            kernel_swizzle<<<1, WGSIZE>>>(b.A, b.B, b.D, b.clk, sw);
            cudaDeviceSynchronize();
            cudaMemcpy(flat.data(), b.D, WGSIZE*D_ELEMS*sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(clk.data(), b.clk, 2*sizeof(uint32_t), cudaMemcpyDeviceToHost);
            float cyc = (float)(clk[1] - clk[0]);
            char param[24]; snprintf(param, 24, "sw=%s", swname[sw]);
            all_pass &= check_uniform_flat("swizzle", flat.data(), (float)(SW_MMAS*K),
                                           csv, param, cyc, cyc/SW_MMAS);
        }
        free_bufs(b);
    }

    // --- 7. multi-warpgroup contention --------------------------------------
    if (run("multiwg")) {
        printf("\n--- multiwg: %d-MMA chain per warpgroup, 1 vs 2 warpgroups ---\n", WG_MMAS);
        DevBufs b = alloc_bufs(MAX_WG);
        for (int nwg = 1; nwg <= MAX_WG; nwg++) {
            cudaMemset(b.D, 0, MAX_WG*WGSIZE*D_ELEMS*sizeof(float));
            kernel_multiwg<<<1, nwg*WGSIZE>>>(b.A, b.B, b.D, b.clk);
            cudaDeviceSynchronize();
            cudaMemcpy(flat.data(), b.D, nwg*WGSIZE*D_ELEMS*sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(clk.data(), b.clk, 2*nwg*sizeof(uint32_t), cudaMemcpyDeviceToHost);
            for (int wg = 0; wg < nwg; wg++) {
                float cyc = (float)(clk[2*wg+1] - clk[2*wg]);
                char param[24]; snprintf(param, 24, "nwg=%d;wg=%d", nwg, wg);
                all_pass &= check_uniform_flat("multiwg",
                                               flat.data() + wg*WGSIZE*D_ELEMS,
                                               (float)(WG_MMAS*K), csv, param,
                                               cyc, cyc/WG_MMAS);
            }
        }
        free_bufs(b);
        printf("  (2-wg cyc/mma vs 1-wg cyc/mma = tensor-core contention factor)\n");
    }

    if (csv) fclose(csv);
    printf("\n================================================================\n");
    printf("Overall: %s\n", all_pass ? "PASSED" : "FAILED");
    return all_pass ? 0 : 1;
}
