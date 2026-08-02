// =============================================================================
// wgmma_multik.cu
//
// Tests WGMMA D-register accumulation across multiple K-tiles.
//
// The critical property being tested: after each wgmma.mma_async with scaleD=1,
// the D registers hold the running sum.  The simulator must preserve the D
// register file across wgmma commits and not reset it between iterations.
//
// Two tests using f16->f32 m64n16k16:
//
//   static_4tiles  - same smem (A=B=1) used for 4 back-to-back MMAs in one group.
//                    All 4 contribute K=16 each -> D[m][n] = 4*K = 64.
//                    No smem reload; one commit/wait covers all 4 ops.
//
//   varying_4tiles - smem reloaded each tile with A[tile] = (tile+1), B = 1.
//                    tile 0: A=1 -> K*1 = 16
//                    tile 1: A=2 -> K*2 = 32
//                    tile 2: A=3 -> K*3 = 48
//                    tile 3: A=4 -> K*4 = 64
//                    Total: K*(1+2+3+4) = 160
//                    Each tile is a separate commit/wait group with smem reload.
//
// Dumps for silicon-vs-sim comparison:
//   smem_A / smem_B  : raw smem content before WGMMA (verifies descriptor layout)
//   D registers      : final D values (verifies WGMMA computation)
//   Per-tile D snap  : intermediate D after each tile in varying_4tiles
//   Clock timing     : %clock delta around WGMMA ops
//
// Exit status: 0 = all PASS, 1 = any FAIL.
// =============================================================================

#include <cuda.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

static constexpr int M       = 64;
static constexpr int N       = 16;
static constexpr int K       = 16;
static constexpr int WGSIZE  = 128;
static constexpr int D_ELEMS = N / 2;    // 8 f32 registers per thread
static constexpr int E       = 2;        // bytes per f16
static constexpr int SBO     = 128;
static constexpr int LBO_A   = M * 16;  // 1024
static constexpr int LBO_B   = N * 16;  // 256
static constexpr int SMEM_A  = M * K * E;  // 2048
static constexpr int SMEM_B  = K * N * E;  // 512

// ---------------------------------------------------------------------------
// Device helpers
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

__host__ __device__ __forceinline__
int smem_off(int leading, int stride, int e, int LBO) {
    // Canonical CuTe GMMA Major-K layout (SW=0), matching the simulator's
    // wgmma_smem_offset(): u128 = (leading/T)*(LBO/16) + (stride%8) + (stride/8)*(SBO/16).
    int T = 16 / e;
    int u128 = (leading / T) * (LBO / 16)
             + (stride % 8)
             + (stride / 8) * (SBO / 16);
    return u128 * 16 + (leading % T) * e;
}

#define WGMMA_FENCE  asm volatile("wgmma.fence.sync.aligned;\n"        ::: "memory")
#define WGMMA_COMMIT asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory")
#define WGMMA_WAIT   asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory")

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

// ---------------------------------------------------------------------------
// Kernel 1: 4 MMAs in one group, constant smem (A=B=1).
// D accumulates across all 4 ops -> expected = 4*K = 64.
// Dumps: smem A/B (before WGMMA), D registers (after), clock timing.
// ---------------------------------------------------------------------------
__global__ void kernel_static4(
    const half* A_g, const half* B_g,
    float* D_g, uint32_t* clk_out,
    half* smA_raw, half* smB_raw)
{
    __shared__ __align__(128) char smA[SMEM_A], smB[SMEM_B];
    const int tid = threadIdx.x;
    for (int i = tid; i < M*K; i += WGSIZE) {
        int m = i/K, k = i%K;
        *(half*)(smA + smem_off(k, m, E, LBO_A)) = A_g[i];
    }
    for (int i = tid; i < K*N; i += WGSIZE) {
        int k = i/N, n = i%N;
        *(half*)(smB + smem_off(k, n, E, LBO_B)) = B_g[i];
    }
    __syncthreads();

    // Dump raw smem content (physical byte order) before WGMMA
    for (int i = tid; i < M*K; i += WGSIZE)
        smA_raw[i] = *(const half*)(smA + i * E);
    for (int i = tid; i < K*N; i += WGSIZE)
        smB_raw[i] = *(const half*)(smB + i * E);

    uint64_t da = make_gmma_desc(smem_addr(smA), LBO_A, SBO);
    uint64_t db = make_gmma_desc(smem_addr(smB), LBO_B, SBO);
    float d0=0,d1=0,d2=0,d3=0,d4=0,d5=0,d6=0,d7=0;

    uint32_t t_start = 0, t_stop = 0;
    __syncthreads();
    if (tid == 0) asm volatile("mov.u32 %0, %%clock;" : "=r"(t_start) :: "memory");
    __syncthreads();

    WGMMA_FENCE;
    mma_f16(da, db, d0,d1,d2,d3,d4,d5,d6,d7);  // D += K  (D was 0)
    mma_f16(da, db, d0,d1,d2,d3,d4,d5,d6,d7);  // D += K
    mma_f16(da, db, d0,d1,d2,d3,d4,d5,d6,d7);  // D += K
    mma_f16(da, db, d0,d1,d2,d3,d4,d5,d6,d7);  // D += K
    WGMMA_COMMIT;
    WGMMA_WAIT;
    WGMMA_FENCE;

    __syncthreads();
    if (tid == 0) asm volatile("mov.u32 %0, %%clock;" : "=r"(t_stop) :: "memory");
    if (tid == 0) { clk_out[0] = t_start; clk_out[1] = t_stop; }

    const int b = tid * D_ELEMS;
    D_g[b]=d0; D_g[b+1]=d1; D_g[b+2]=d2; D_g[b+3]=d3;
    D_g[b+4]=d4; D_g[b+5]=d5; D_g[b+6]=d6; D_g[b+7]=d7;
}

// ---------------------------------------------------------------------------
// Kernel 2: 4 tiles with different A values, smem reloaded each tile.
// A_g layout: [4 * M * K], where A_g[tile*M*K + m*K + k] = (half)(tile+1).
// B_g: [K * N] all ones (shared across tiles).
// Expected D[m][n] = K*(1+2+3+4) = 160.
//
// Dumps:
//   smA_raw/smB_raw : physical smem content of tile 0 (before first WGMMA)
//   D_tiles         : D register snapshot after each of the 4 tiles
//                     (4 * WGSIZE * D_ELEMS floats, tile-major)
//   clk_out         : clock_start/stop bracketing all 4 tiles
// ---------------------------------------------------------------------------
__global__ void kernel_varying4(
    const half* A_g, const half* B_g,
    float* D_g, uint32_t* clk_out,
    half* smA_raw, half* smB_raw,
    float* D_tiles)
{
    __shared__ __align__(128) char smA[SMEM_A], smB[SMEM_B];
    const int tid = threadIdx.x;

    // Load B once - same for all tiles.
    for (int i = tid; i < K*N; i += WGSIZE) {
        int k = i/N, n = i%N;
        *(half*)(smB + smem_off(k, n, E, LBO_B)) = B_g[i];
    }

    // Tile 0: A = 1.0
    for (int i = tid; i < M*K; i += WGSIZE) {
        int m = i/K, k = i%K;
        *(half*)(smA + smem_off(k, m, E, LBO_A)) = A_g[0*M*K + m*K + k];
    }
    __syncthreads();

    // Dump tile 0 smem (physical byte order) before any WGMMA
    for (int i = tid; i < M*K; i += WGSIZE)
        smA_raw[i] = *(const half*)(smA + i * E);
    for (int i = tid; i < K*N; i += WGSIZE)
        smB_raw[i] = *(const half*)(smB + i * E);

    uint64_t da = make_gmma_desc(smem_addr(smA), LBO_A, SBO);
    uint64_t db = make_gmma_desc(smem_addr(smB), LBO_B, SBO);
    float d0=0,d1=0,d2=0,d3=0,d4=0,d5=0,d6=0,d7=0;

    uint32_t t_start = 0, t_stop = 0;
    __syncthreads();
    if (tid == 0) asm volatile("mov.u32 %0, %%clock;" : "=r"(t_start) :: "memory");
    __syncthreads();

    WGMMA_FENCE;
    mma_f16(da, db, d0,d1,d2,d3,d4,d5,d6,d7);
    WGMMA_COMMIT;
    WGMMA_WAIT;

    // Snapshot D after tile 0 (expected each element = K*1 = 16)
    { const int b = tid * D_ELEMS, off = 0 * WGSIZE * D_ELEMS;
      D_tiles[off+b]=d0; D_tiles[off+b+1]=d1; D_tiles[off+b+2]=d2; D_tiles[off+b+3]=d3;
      D_tiles[off+b+4]=d4; D_tiles[off+b+5]=d5; D_tiles[off+b+6]=d6; D_tiles[off+b+7]=d7; }

    // Tile 1: A = 2.0
    WGMMA_FENCE;  // post-wait: releases D, signals smem can be updated
    for (int i = tid; i < M*K; i += WGSIZE) {
        int m = i/K, k = i%K;
        *(half*)(smA + smem_off(k, m, E, LBO_A)) = A_g[1*M*K + m*K + k];
    }
    __syncthreads();
    WGMMA_FENCE;  // pre-MMA: signals new smem content visible to wgmma
    mma_f16(da, db, d0,d1,d2,d3,d4,d5,d6,d7);
    WGMMA_COMMIT;
    WGMMA_WAIT;

    // Snapshot D after tile 1 (expected each element = K*(1+2) = 48)
    { const int b = tid * D_ELEMS, off = 1 * WGSIZE * D_ELEMS;
      D_tiles[off+b]=d0; D_tiles[off+b+1]=d1; D_tiles[off+b+2]=d2; D_tiles[off+b+3]=d3;
      D_tiles[off+b+4]=d4; D_tiles[off+b+5]=d5; D_tiles[off+b+6]=d6; D_tiles[off+b+7]=d7; }

    // Tile 2: A = 3.0
    WGMMA_FENCE;
    for (int i = tid; i < M*K; i += WGSIZE) {
        int m = i/K, k = i%K;
        *(half*)(smA + smem_off(k, m, E, LBO_A)) = A_g[2*M*K + m*K + k];
    }
    __syncthreads();
    WGMMA_FENCE;
    mma_f16(da, db, d0,d1,d2,d3,d4,d5,d6,d7);
    WGMMA_COMMIT;
    WGMMA_WAIT;

    // Snapshot D after tile 2 (expected each element = K*(1+2+3) = 96)
    { const int b = tid * D_ELEMS, off = 2 * WGSIZE * D_ELEMS;
      D_tiles[off+b]=d0; D_tiles[off+b+1]=d1; D_tiles[off+b+2]=d2; D_tiles[off+b+3]=d3;
      D_tiles[off+b+4]=d4; D_tiles[off+b+5]=d5; D_tiles[off+b+6]=d6; D_tiles[off+b+7]=d7; }

    // Tile 3: A = 4.0
    WGMMA_FENCE;
    for (int i = tid; i < M*K; i += WGSIZE) {
        int m = i/K, k = i%K;
        *(half*)(smA + smem_off(k, m, E, LBO_A)) = A_g[3*M*K + m*K + k];
    }
    __syncthreads();
    WGMMA_FENCE;
    mma_f16(da, db, d0,d1,d2,d3,d4,d5,d6,d7);
    WGMMA_COMMIT;
    WGMMA_WAIT;
    WGMMA_FENCE;  // final fence

    __syncthreads();
    if (tid == 0) asm volatile("mov.u32 %0, %%clock;" : "=r"(t_stop) :: "memory");
    if (tid == 0) { clk_out[0] = t_start; clk_out[1] = t_stop; }

    // Snapshot D after tile 3 (final, expected each element = K*(1+2+3+4) = 160)
    { const int b = tid * D_ELEMS, off = 3 * WGSIZE * D_ELEMS;
      D_tiles[off+b]=d0; D_tiles[off+b+1]=d1; D_tiles[off+b+2]=d2; D_tiles[off+b+3]=d3;
      D_tiles[off+b+4]=d4; D_tiles[off+b+5]=d5; D_tiles[off+b+6]=d6; D_tiles[off+b+7]=d7; }

    const int b = tid * D_ELEMS;
    D_g[b]=d0; D_g[b+1]=d1; D_g[b+2]=d2; D_g[b+3]=d3;
    D_g[b+4]=d4; D_g[b+5]=d5; D_g[b+6]=d6; D_g[b+7]=d7;
}

// ---------------------------------------------------------------------------
// Kernel 3: Fence overhead.
// Measures the cost of wgmma.fence.sync.aligned in isolation.
// N_FENCE consecutive FENCEs; no WGMMA ops.
// cycles_per_fence = (stop - start) / N_FENCE
// ---------------------------------------------------------------------------
static constexpr int N_FENCE = 256;

__global__ void kernel_fence_overhead(uint32_t* clk_out) {
    const int tid = threadIdx.x;
    uint32_t t_start = 0, t_stop = 0;
    __syncthreads();
    if (tid == 0) asm volatile("mov.u32 %0, %%clock;" : "=r"(t_start) :: "memory");
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < N_FENCE; i++)
        WGMMA_FENCE;
    __syncthreads();
    if (tid == 0) asm volatile("mov.u32 %0, %%clock;" : "=r"(t_stop) :: "memory");
    if (tid == 0) { clk_out[0] = t_start; clk_out[1] = t_stop; }
}

// ---------------------------------------------------------------------------
// Kernel 4: Commit+Wait barrier overhead (no MMA ops).
// Commits an empty wgmma group (no ops issued) and waits on it.
// This measures the pure synchronization barrier cost of COMMIT+WAIT.
// N_BARRIER iterations; cycles_per_barrier = (stop - start) / N_BARRIER.
// ---------------------------------------------------------------------------
static constexpr int N_BARRIER = 256;

__global__ void kernel_commit_wait_overhead(uint32_t* clk_out) {
    const int tid = threadIdx.x;
    uint32_t t_start = 0, t_stop = 0;
    WGMMA_FENCE;
    __syncthreads();
    if (tid == 0) asm volatile("mov.u32 %0, %%clock;" : "=r"(t_start) :: "memory");
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < N_BARRIER; i++) {
        WGMMA_COMMIT;
        WGMMA_WAIT;
    }
    __syncthreads();
    if (tid == 0) asm volatile("mov.u32 %0, %%clock;" : "=r"(t_stop) :: "memory");
    if (tid == 0) { clk_out[0] = t_start; clk_out[1] = t_stop; }
}

// ---------------------------------------------------------------------------
// Kernel 5: Single-shape MMA throughput (m64n16k16 f32<-f16xf16).
// N_MMA back-to-back MMA ops in one commit group.
// Includes smem setup; clock brackets fence..commit_wait..fence.
// cycles_per_mma = (stop - start) / N_MMA (includes fence+commit+wait amortised).
// ---------------------------------------------------------------------------
static constexpr int N_MMA = 1024;

__global__ void kernel_mma_throughput(
    const half* A_g, const half* B_g, float* D_g, uint32_t* clk_out)
{
    __shared__ __align__(128) char smA[SMEM_A], smB[SMEM_B];
    const int tid = threadIdx.x;
    for (int i = tid; i < M*K; i += WGSIZE) {
        int m = i/K, k = i%K;
        *(half*)(smA + smem_off(k, m, E, LBO_A)) = A_g[i];
    }
    for (int i = tid; i < K*N; i += WGSIZE) {
        int k = i/N, n = i%N;
        *(half*)(smB + smem_off(k, n, E, LBO_B)) = B_g[i];
    }
    __syncthreads();
    uint64_t da = make_gmma_desc(smem_addr(smA), LBO_A, SBO);
    uint64_t db = make_gmma_desc(smem_addr(smB), LBO_B, SBO);
    float d0=0,d1=0,d2=0,d3=0,d4=0,d5=0,d6=0,d7=0;

    uint32_t t_start = 0, t_stop = 0;
    __syncthreads();
    if (tid == 0) asm volatile("mov.u32 %0, %%clock;" : "=r"(t_start) :: "memory");
    __syncthreads();

    WGMMA_FENCE;
    #pragma unroll 8
    for (int i = 0; i < N_MMA; i++)
        mma_f16(da, db, d0,d1,d2,d3,d4,d5,d6,d7);
    WGMMA_COMMIT;
    WGMMA_WAIT;
    WGMMA_FENCE;

    __syncthreads();
    if (tid == 0) asm volatile("mov.u32 %0, %%clock;" : "=r"(t_stop) :: "memory");
    if (tid == 0) { clk_out[0] = t_start; clk_out[1] = t_stop; }

    const int b = tid * D_ELEMS;
    D_g[b]=d0; D_g[b+1]=d1; D_g[b+2]=d2; D_g[b+3]=d3;
    D_g[b+4]=d4; D_g[b+5]=d5; D_g[b+6]=d6; D_g[b+7]=d7;
}

// ===========================================================================
// Host helpers
// ===========================================================================
static void d_frag_pos(int T, int e, int* row, int* col) {
    int warp = T/32, lane = T%32, s = e/4, k = e%4;
    *row = (lane/4)*2 + k/2 + warp*16;
    *col = (lane%4)*2 + k%2 + s*8;
}

static void reassemble(const float* flat, float D[M][N]) {
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) D[m][n] = 0.f;
    for (int T = 0; T < WGSIZE; T++)
        for (int e = 0; e < D_ELEMS; e++) {
            int row, col;
            d_frag_pos(T, e, &row, &col);
            if (row < M && col < N)
                D[row][col] = flat[T * D_ELEMS + e];
        }
}

static bool check_uniform(const char* name, float D[M][N], float expected, FILE* outfile = NULL) {
    float max_err = 0.f;
    int   max_m = 0, max_n = 0;
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float e = fabsf(D[m][n] - expected);
            if (e > max_err) { max_err = e; max_m = m; max_n = n; }
        }
    float tol  = fabsf(expected) * 1e-2f;
    bool  pass = (max_err <= tol);
    printf("[%-36s] %s  expected=%.1f  max_err=%.4e\n",
           name, pass ? "PASS" : "FAIL", expected, max_err);
    if (!pass)
        printf("  Worst: D[%d][%d] = %.6f  expected = %.6f\n",
               max_m, max_n, D[max_m][max_n], expected);
    if (outfile)
        fprintf(outfile, "%s,%s,%.1f,%.4e\n", name, pass ? "PASS" : "FAIL", expected, max_err);
    return pass;
}

// Write a comprehensive dump file for silicon-vs-sim diff comparison.
// smA_raw/smB_raw: raw physical smem content (M*K and K*N fp16 values).
// flat           : per-thread D registers (WGSIZE*D_ELEMS floats).
// D              : reassembled D[M][N] matrix.
// D_tiles        : per-tile D snapshots (num_tiles * WGSIZE*D_ELEMS), NULL if none.
// num_tiles      : number of tile snapshots in D_tiles.
// t_start/stop   : raw %clock values.
static void dump_results(
    const char* filename, const char* test_name,
    const half* smA_raw, const half* smB_raw,
    const float* flat, float D[M][N],
    const float* D_tiles, int num_tiles,
    uint32_t t_start, uint32_t t_stop,
    int num_mma_ops)
{
    FILE* f = fopen(filename, "w");
    if (!f) { fprintf(stderr, "cannot open %s\n", filename); return; }

    fprintf(f, "=== %s ===\n", test_name);
    fprintf(f, "[timing]\n");
    fprintf(f, "clock_start=%u\n", t_start);
    fprintf(f, "clock_stop=%u\n", t_stop);
    fprintf(f, "total_cycles=%u\n", t_stop - t_start);
    if (num_mma_ops > 0)
        fprintf(f, "cycles_per_mma=%.2f\n", (float)(t_stop - t_start) / num_mma_ops);

    // Physical smem layout — what the WGMMA descriptor sees
    fprintf(f, "\n[smem_A physical M=%d K=%d halfwords]\n", M, K);
    for (int i = 0; i < M*K; i++)
        fprintf(f, "%d %.6f\n", i, __half2float(smA_raw[i]));

    fprintf(f, "\n[smem_B physical K=%d N=%d halfwords]\n", K, N);
    for (int i = 0; i < K*N; i++)
        fprintf(f, "%d %.6f\n", i, __half2float(smB_raw[i]));

    // Logical smem layout via smem_off mapping (shows which (k,m)/(k,n) -> physical index)
    fprintf(f, "\n[smem_A logical k m phys_idx value]\n");
    for (int k = 0; k < K; k++)
        for (int m = 0; m < M; m++) {
            int off = smem_off(k, m, E, LBO_A);
            int idx = off / E;
            fprintf(f, "%d %d %d %.6f\n", k, m, idx, __half2float(smA_raw[idx]));
        }

    fprintf(f, "\n[smem_B logical k n phys_idx value]\n");
    for (int k = 0; k < K; k++)
        for (int n = 0; n < N; n++) {
            int off = smem_off(k, n, E, LBO_B);
            int idx = off / E;
            fprintf(f, "%d %d %d %.6f\n", k, n, idx, __half2float(smB_raw[idx]));
        }

    // Per-tile D snapshots (if provided)
    if (D_tiles && num_tiles > 0) {
        static const float tile_expected[] = {16.f, 48.f, 96.f, 160.f};
        for (int t = 0; t < num_tiles; t++) {
            fprintf(f, "\n[D_tile_%d thread elem value expected=%.1f]\n",
                    t, (t < 4) ? tile_expected[t] : 0.f);
            const float* snap = D_tiles + t * WGSIZE * D_ELEMS;
            for (int T = 0; T < WGSIZE; T++)
                for (int e = 0; e < D_ELEMS; e++)
                    fprintf(f, "%d %d %.6f\n", T, e, snap[T*D_ELEMS + e]);
        }
    }

    // Final D registers (per-thread, raw)
    fprintf(f, "\n[D_registers thread elem value]\n");
    for (int T = 0; T < WGSIZE; T++)
        for (int e = 0; e < D_ELEMS; e++)
            fprintf(f, "%d %d %.6f\n", T, e, flat[T*D_ELEMS + e]);

    // Thread-to-matrix mapping + D matrix
    fprintf(f, "\n[thread_to_matrix thread elem row col]\n");
    for (int T = 0; T < WGSIZE; T++)
        for (int e = 0; e < D_ELEMS; e++) {
            int row, col;
            d_frag_pos(T, e, &row, &col);
            fprintf(f, "%d %d %d %d\n", T, e, row, col);
        }

    fprintf(f, "\n[D_matrix row col value]\n");
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++)
            fprintf(f, "%d %d %.6f\n", m, n, D[m][n]);

    fclose(f);
    printf("  -> dumped to %s\n", filename);
}

// ===========================================================================
// main
// ===========================================================================
int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("wgmma_multik  -  %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("================================================================\n");

    FILE* outfile = fopen("wgmma_multik_results.csv", "w");
    if (outfile)
        fprintf(outfile, "test,status,expected,max_err\n");

    bool all_pass = true;
    static float D[M][N];

    // --- Test 1: static_4tiles ---
    printf("\n--- static_4tiles (4 MMAs, same smem A=B=1, expected D=64) ---\n");
    {
        std::vector<half> h_A(M*K, __float2half(1.f));
        std::vector<half> h_B(K*N, __float2half(1.f));
        half *d_A, *d_B; float *d_D; uint32_t *d_clk;
        half *d_smA_raw, *d_smB_raw;
        cudaMalloc(&d_A,       h_A.size()*sizeof(half));
        cudaMalloc(&d_B,       h_B.size()*sizeof(half));
        cudaMalloc(&d_D,       WGSIZE*D_ELEMS*sizeof(float));
        cudaMalloc(&d_clk,     2*sizeof(uint32_t));
        cudaMalloc(&d_smA_raw, M*K*sizeof(half));
        cudaMalloc(&d_smB_raw, K*N*sizeof(half));
        cudaMemcpy(d_A, h_A.data(), h_A.size()*sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B.data(), h_B.size()*sizeof(half), cudaMemcpyHostToDevice);

        kernel_static4<<<1, WGSIZE>>>(d_A, d_B, d_D, d_clk, d_smA_raw, d_smB_raw);
        cudaDeviceSynchronize();

        std::vector<float>    flat(WGSIZE*D_ELEMS);
        std::vector<uint32_t> clk(2);
        std::vector<half>     smA_raw(M*K), smB_raw(K*N);
        cudaMemcpy(flat.data(),    d_D,       flat.size()*sizeof(float),    cudaMemcpyDeviceToHost);
        cudaMemcpy(clk.data(),     d_clk,     2*sizeof(uint32_t),           cudaMemcpyDeviceToHost);
        cudaMemcpy(smA_raw.data(), d_smA_raw, M*K*sizeof(half),             cudaMemcpyDeviceToHost);
        cudaMemcpy(smB_raw.data(), d_smB_raw, K*N*sizeof(half),             cudaMemcpyDeviceToHost);
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_D);
        cudaFree(d_clk); cudaFree(d_smA_raw); cudaFree(d_smB_raw);

        reassemble(flat.data(), D);
        all_pass &= check_uniform("static_4tiles", D, (float)(4 * K), outfile);

        printf("  clock: start=%u stop=%u cycles=%u (%.2f cyc/mma)\n",
               clk[0], clk[1], clk[1]-clk[0], (float)(clk[1]-clk[0])/4.f);

        dump_results("wgmma_multik_static4.txt", "static_4tiles",
                     smA_raw.data(), smB_raw.data(),
                     flat.data(), D,
                     nullptr, 0,
                     clk[0], clk[1], 4);
    }

    // --- Test 2: varying_4tiles ---
    printf("\n--- varying_4tiles (A[tile]=tile+1, B=1, expected D=160) ---\n");
    {
        std::vector<half> h_A(4 * M * K);
        for (int t = 0; t < 4; t++)
            for (int i = 0; i < M*K; i++)
                h_A[t*M*K + i] = __float2half((float)(t + 1));
        std::vector<half> h_B(K*N, __float2half(1.f));
        half *d_A, *d_B; float *d_D; uint32_t *d_clk;
        half *d_smA_raw, *d_smB_raw;
        float *d_D_tiles;
        cudaMalloc(&d_A,       h_A.size()*sizeof(half));
        cudaMalloc(&d_B,       h_B.size()*sizeof(half));
        cudaMalloc(&d_D,       WGSIZE*D_ELEMS*sizeof(float));
        cudaMalloc(&d_clk,     2*sizeof(uint32_t));
        cudaMalloc(&d_smA_raw, M*K*sizeof(half));
        cudaMalloc(&d_smB_raw, K*N*sizeof(half));
        cudaMalloc(&d_D_tiles, 4*WGSIZE*D_ELEMS*sizeof(float));
        cudaMemcpy(d_A, h_A.data(), h_A.size()*sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B.data(), h_B.size()*sizeof(half), cudaMemcpyHostToDevice);

        kernel_varying4<<<1, WGSIZE>>>(d_A, d_B, d_D, d_clk, d_smA_raw, d_smB_raw, d_D_tiles);
        cudaDeviceSynchronize();

        std::vector<float>    flat(WGSIZE*D_ELEMS);
        std::vector<uint32_t> clk(2);
        std::vector<half>     smA_raw(M*K), smB_raw(K*N);
        std::vector<float>    D_tiles(4*WGSIZE*D_ELEMS);
        cudaMemcpy(flat.data(),     d_D,        flat.size()*sizeof(float),     cudaMemcpyDeviceToHost);
        cudaMemcpy(clk.data(),      d_clk,      2*sizeof(uint32_t),            cudaMemcpyDeviceToHost);
        cudaMemcpy(smA_raw.data(),  d_smA_raw,  M*K*sizeof(half),              cudaMemcpyDeviceToHost);
        cudaMemcpy(smB_raw.data(),  d_smB_raw,  K*N*sizeof(half),              cudaMemcpyDeviceToHost);
        cudaMemcpy(D_tiles.data(),  d_D_tiles,  D_tiles.size()*sizeof(float),  cudaMemcpyDeviceToHost);
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_D);
        cudaFree(d_clk); cudaFree(d_smA_raw); cudaFree(d_smB_raw); cudaFree(d_D_tiles);

        reassemble(flat.data(), D);
        all_pass &= check_uniform("varying_4tiles", D, (float)(K * 10), outfile);

        printf("  clock: start=%u stop=%u cycles=%u (%.2f cyc/mma)\n",
               clk[0], clk[1], clk[1]-clk[0], (float)(clk[1]-clk[0])/4.f);

        dump_results("wgmma_multik_varying4.txt", "varying_4tiles",
                     smA_raw.data(), smB_raw.data(),
                     flat.data(), D,
                     D_tiles.data(), 4,
                     clk[0], clk[1], 4);
    }

    // --- Timing breakdown: fence overhead ---
    printf("\n--- fence_overhead (N=%d WGMMA_FENCEs, no MMA) ---\n", N_FENCE);
    {
        uint32_t *d_clk;
        cudaMalloc(&d_clk, 2*sizeof(uint32_t));
        kernel_fence_overhead<<<1, WGSIZE>>>(d_clk);
        cudaDeviceSynchronize();
        std::vector<uint32_t> clk(2);
        cudaMemcpy(clk.data(), d_clk, 2*sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaFree(d_clk);
        float cyc_per_fence = (float)(clk[1]-clk[0]) / N_FENCE;
        printf("  clock: start=%u stop=%u total=%u cyc/fence=%.2f\n",
               clk[0], clk[1], clk[1]-clk[0], cyc_per_fence);
        if (outfile)
            fprintf(outfile, "fence_overhead,TIMING,N/A,cyc_per_fence=%.2f\n", cyc_per_fence);
    }

    // --- Timing breakdown: commit+wait barrier overhead (no MMA ops) ---
    printf("\n--- commit_wait_overhead (N=%d empty COMMIT+WAIT, no MMA) ---\n", N_BARRIER);
    {
        uint32_t *d_clk;
        cudaMalloc(&d_clk, 2*sizeof(uint32_t));
        kernel_commit_wait_overhead<<<1, WGSIZE>>>(d_clk);
        cudaDeviceSynchronize();
        std::vector<uint32_t> clk(2);
        cudaMemcpy(clk.data(), d_clk, 2*sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaFree(d_clk);
        float cyc_per_barrier = (float)(clk[1]-clk[0]) / N_BARRIER;
        printf("  clock: start=%u stop=%u total=%u cyc/barrier=%.2f\n",
               clk[0], clk[1], clk[1]-clk[0], cyc_per_barrier);
        if (outfile)
            fprintf(outfile, "commit_wait_overhead,TIMING,N/A,cyc_per_barrier=%.2f\n", cyc_per_barrier);
    }

    // --- Timing: MMA throughput (m64n16k16 f32<-f16xf16, N=%d ops) ---
    printf("\n--- mma_throughput_m64n16k16 (N=%d MMAs, A=B=1) ---\n", N_MMA);
    {
        std::vector<half> h_A(M*K, __float2half(1.f));
        std::vector<half> h_B(K*N, __float2half(1.f));
        half *d_A, *d_B; float *d_D; uint32_t *d_clk;
        cudaMalloc(&d_A,   h_A.size()*sizeof(half));
        cudaMalloc(&d_B,   h_B.size()*sizeof(half));
        cudaMalloc(&d_D,   WGSIZE*D_ELEMS*sizeof(float));
        cudaMalloc(&d_clk, 2*sizeof(uint32_t));
        cudaMemcpy(d_A, h_A.data(), h_A.size()*sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B.data(), h_B.size()*sizeof(half), cudaMemcpyHostToDevice);
        kernel_mma_throughput<<<1, WGSIZE>>>(d_A, d_B, d_D, d_clk);
        cudaDeviceSynchronize();
        std::vector<uint32_t> clk(2);
        cudaMemcpy(clk.data(), d_clk, 2*sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_D); cudaFree(d_clk);
        float cyc_per_mma = (float)(clk[1]-clk[0]) / N_MMA;
        printf("  clock: start=%u stop=%u total=%u cyc/mma=%.2f\n",
               clk[0], clk[1], clk[1]-clk[0], cyc_per_mma);
        if (outfile)
            fprintf(outfile, "mma_throughput_m64n16k16,TIMING,N/A,cyc_per_mma=%.2f\n", cyc_per_mma);
    }

    if (outfile)
        fclose(outfile);

    printf("\n================================================================\n");
    printf("Overall: %s\n", all_pass ? "PASSED" : "FAILED");
    return all_pass ? 0 : 1;
}
