// =============================================================================
// wgmma_multik.cu
//
// Tests WGMMA D-register accumulation across multiple K-tiles.
//
// The critical property being tested: after each wgmma.mma_async with scaleD=1,
// the D registers hold the running sum.  The simulator must preserve the D
// register file across wgmma commits and not reset it between iterations.
//
// Two tests using f16→f32 m64n16k16:
//
//   static_4tiles  – same smem (A=B=1) used for 4 back-to-back MMAs in one group.
//                    All 4 contribute K=16 each → D[m][n] = 4*K = 64.
//                    No smem reload; one commit/wait covers all 4 ops.
//
//   varying_4tiles – smem reloaded each tile with A[tile] = (tile+1), B = 1.
//                    tile 0: A=1 → K*1 = 16
//                    tile 1: A=2 → K*2 = 32
//                    tile 2: A=3 → K*3 = 48
//                    tile 3: A=4 → K*4 = 64
//                    Total: K*(1+2+3+4) = 160
//                    Each tile is a separate commit/wait group with smem reload.
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
static constexpr int LBO_A   = M * E * 8;  // 1024
static constexpr int LBO_B   = N * E * 8;  // 256
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
    int T = 16 / e;
    return (stride % T + (leading % 8) * T) * e
         + (stride / T) * SBO
         + (leading / 8) * LBO;
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
// D accumulates across all 4 ops → expected = 4*K = 64.
// ---------------------------------------------------------------------------
__global__ void kernel_static4(const half* A_g, const half* B_g, float* D_g) {
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
    WGMMA_FENCE;
    mma_f16(da, db, d0,d1,d2,d3,d4,d5,d6,d7);  // D += K  (D was 0)
    mma_f16(da, db, d0,d1,d2,d3,d4,d5,d6,d7);  // D += K
    mma_f16(da, db, d0,d1,d2,d3,d4,d5,d6,d7);  // D += K
    mma_f16(da, db, d0,d1,d2,d3,d4,d5,d6,d7);  // D += K
    WGMMA_COMMIT;
    WGMMA_WAIT;
    WGMMA_FENCE;
    const int b = tid * D_ELEMS;
    D_g[b]=d0; D_g[b+1]=d1; D_g[b+2]=d2; D_g[b+3]=d3;
    D_g[b+4]=d4; D_g[b+5]=d5; D_g[b+6]=d6; D_g[b+7]=d7;
}

// ---------------------------------------------------------------------------
// Kernel 2: 4 tiles with different A values, smem reloaded each tile.
// A_g layout: [4 * M * K], where A_g[tile*M*K + m*K + k] = (half)(tile+1).
// B_g: [K * N] all ones (shared across tiles).
// Expected D[m][n] = K*(1+2+3+4) = 160.
// ---------------------------------------------------------------------------
__global__ void kernel_varying4(const half* A_g, const half* B_g, float* D_g) {
    __shared__ __align__(128) char smA[SMEM_A], smB[SMEM_B];
    const int tid = threadIdx.x;

    // Load B once — same for all tiles.
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
    uint64_t da = make_gmma_desc(smem_addr(smA), LBO_A, SBO);
    uint64_t db = make_gmma_desc(smem_addr(smB), LBO_B, SBO);
    float d0=0,d1=0,d2=0,d3=0,d4=0,d5=0,d6=0,d7=0;
    WGMMA_FENCE;
    mma_f16(da, db, d0,d1,d2,d3,d4,d5,d6,d7);
    WGMMA_COMMIT;
    WGMMA_WAIT;

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

static bool check_uniform(const char* name, float D[M][N], float expected) {
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
    return pass;
}

// ===========================================================================
// main
// ===========================================================================
int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("wgmma_multik  —  %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("================================================================\n");

    bool all_pass = true;
    static float D[M][N];

    // --- Test 1: static_4tiles ---
    printf("\n--- static_4tiles (4 MMAs, same smem A=B=1, expected D=64) ---\n");
    {
        std::vector<half> h_A(M*K, __float2half(1.f));
        std::vector<half> h_B(K*N, __float2half(1.f));
        half *d_A, *d_B; float *d_D;
        cudaMalloc(&d_A, h_A.size()*sizeof(half));
        cudaMalloc(&d_B, h_B.size()*sizeof(half));
        cudaMalloc(&d_D, WGSIZE*D_ELEMS*sizeof(float));
        cudaMemcpy(d_A, h_A.data(), h_A.size()*sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B.data(), h_B.size()*sizeof(half), cudaMemcpyHostToDevice);
        kernel_static4<<<1, WGSIZE>>>(d_A, d_B, d_D);
        cudaDeviceSynchronize();
        std::vector<float> flat(WGSIZE*D_ELEMS);
        cudaMemcpy(flat.data(), d_D, flat.size()*sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_D);
        reassemble(flat.data(), D);
        all_pass &= check_uniform("static_4tiles", D, (float)(4 * K));
    }

    // --- Test 2: varying_4tiles ---
    printf("\n--- varying_4tiles (A[tile]=tile+1, B=1, expected D=160) ---\n");
    {
        // A_g: [4*M*K] where tile t has all elements = float(t+1)
        std::vector<half> h_A(4 * M * K);
        for (int t = 0; t < 4; t++)
            for (int i = 0; i < M*K; i++)
                h_A[t*M*K + i] = __float2half((float)(t + 1));
        std::vector<half> h_B(K*N, __float2half(1.f));
        half *d_A, *d_B; float *d_D;
        cudaMalloc(&d_A, h_A.size()*sizeof(half));
        cudaMalloc(&d_B, h_B.size()*sizeof(half));
        cudaMalloc(&d_D, WGSIZE*D_ELEMS*sizeof(float));
        cudaMemcpy(d_A, h_A.data(), h_A.size()*sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B.data(), h_B.size()*sizeof(half), cudaMemcpyHostToDevice);
        kernel_varying4<<<1, WGSIZE>>>(d_A, d_B, d_D);
        cudaDeviceSynchronize();
        std::vector<float> flat(WGSIZE*D_ELEMS);
        cudaMemcpy(flat.data(), d_D, flat.size()*sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_D);
        reassemble(flat.data(), D);
        // Expected: K*(1+2+3+4) = 16*10 = 160
        all_pass &= check_uniform("varying_4tiles", D, (float)(K * 10));
    }

    printf("\n================================================================\n");
    printf("Overall: %s\n", all_pass ? "PASS" : "FAIL");
    return all_pass ? 0 : 1;
}
