// =============================================================================
// wgmma_dtypes.cu
//
// Tests WGMMA with f16 accumulator: m64n16k16.f16.f16.f16.
//
// wgmma_verify already covers f16/bf16/tf32/fp8/int8 with f32/s32 accumulators.
// This file adds the f16←f16×f16 variant where D registers are f16x2 packed
// into .b32 (N/4 = 4 registers per thread for N=16, each holding 2 f16 values).
//
// Fragment layout for f16 accumulator is identical to f32 — same d_frag_pos
// formula — but each register slot holds 2 f16 values (low and high halfword).
//
// Two tests (all-ones style, exactly representable in f16):
//   A=1, B=1 → D[m][n] = K      = 16
//   A=2, B=3 → D[m][n] = 6*K    = 96
//
// Exit status: 0 = all PASS, 1 = any FAIL.
// =============================================================================

#include <cuda.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

static constexpr int M      = 64;
static constexpr int N      = 16;
static constexpr int K      = 16;
static constexpr int WGSIZE = 128;
static constexpr int E      = 2;          // bytes per f16
static constexpr int SBO    = 128;
static constexpr int LBO_A  = M * E * 8;  // 1024
static constexpr int LBO_B  = N * E * 8;  // 256
static constexpr int SMEM_A = M * K * E;  // 2048
static constexpr int SMEM_B = K * N * E;  // 512
static constexpr int D_REGS = N / 4;      // 4 f16x2 registers per thread (N=16)

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

// ---------------------------------------------------------------------------
// Kernel: m64n16k16.f16.f16.f16
// D is 4 f16x2 registers (uint32_t) per thread for N=16.
// ---------------------------------------------------------------------------
__global__ void kernel_f16f16(const half* A_g, const half* B_g, uint32_t* D_g) {
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
    uint32_t d0=0, d1=0, d2=0, d3=0;
    WGMMA_FENCE;
    asm volatile(
        "{ .reg .pred p; setp.ne.b32 p,1,0;\n"
        "wgmma.mma_async.sync.aligned.m64n16k16.f16.f16.f16 "
        "{%0,%1,%2,%3},%4,%5,p,1,1,0,0; }\n"
        : "+r"(d0),"+r"(d1),"+r"(d2),"+r"(d3)
        : "l"(da),"l"(db) : "memory");
    WGMMA_COMMIT; WGMMA_WAIT; WGMMA_FENCE;
    int b = tid * D_REGS;
    D_g[b]=d0; D_g[b+1]=d1; D_g[b+2]=d2; D_g[b+3]=d3;
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------
static void d_frag_pos(int T, int e, int* row, int* col) {
    int warp = T/32, lane = T%32, s = e/4, k = e%4;
    *row = (lane/4)*2 + k/2 + warp*16;
    *col = (lane%4)*2 + k%2 + s*8;
}

// Unpack WGSIZE*D_REGS f16x2 registers into M×N float matrix.
// Thread T, register r → scalar elements e=2r (low half) and e=2r+1 (high half).
static void reassemble(const uint32_t* flat, float* D) {
    for (int i = 0; i < M*N; i++) D[i] = 0.f;
    for (int T = 0; T < WGSIZE; T++) {
        for (int r = 0; r < D_REGS; r++) {
            uint32_t packed = flat[T * D_REGS + r];
            for (int h = 0; h < 2; h++) {
                int e = 2*r + h;
                uint16_t bits = (h == 0) ? (uint16_t)(packed & 0xFFFFu)
                                         : (uint16_t)(packed >> 16);
                half val;
                memcpy(&val, &bits, sizeof(val));
                int row, col;
                d_frag_pos(T, e, &row, &col);
                if (row < M && col < N)
                    D[row * N + col] = __half2float(val);
            }
        }
    }
}

static bool check_uniform(const char* name, const float* D, float expected) {
    float max_err = 0.f;
    int   worst   = 0;
    for (int i = 0; i < M*N; i++) {
        float e = fabsf(D[i] - expected);
        if (e > max_err) { max_err = e; worst = i; }
    }
    float tol  = fabsf(expected) * 5e-3f + 0.5f;  // generous for f16 precision
    bool  pass = (max_err <= tol);
    printf("[%-40s] %s  expected=%.1f  max_err=%.4e\n",
           name, pass ? "PASS" : "FAIL", expected, max_err);
    if (!pass)
        printf("  Worst: D[%d][%d] = %.6f  expected = %.6f\n",
               worst/N, worst%N, D[worst], expected);
    return pass;
}

static bool run(const char* name, float a_val, float b_val, float expected) {
    std::vector<half> h_A(M*K, __float2half(a_val));
    std::vector<half> h_B(K*N, __float2half(b_val));
    half*     d_A;
    half*     d_B;
    uint32_t* d_D;
    cudaMalloc(&d_A, h_A.size() * sizeof(half));
    cudaMalloc(&d_B, h_B.size() * sizeof(half));
    cudaMalloc(&d_D, WGSIZE * D_REGS * sizeof(uint32_t));
    cudaMemcpy(d_A, h_A.data(), h_A.size()*sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), h_B.size()*sizeof(half), cudaMemcpyHostToDevice);
    kernel_f16f16<<<1, WGSIZE>>>(d_A, d_B, d_D);
    cudaDeviceSynchronize();
    std::vector<uint32_t> flat(WGSIZE * D_REGS);
    cudaMemcpy(flat.data(), d_D, flat.size()*sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_D);
    std::vector<float> D(M * N);
    reassemble(flat.data(), D.data());
    return check_uniform(name, D.data(), expected);
}

// ===========================================================================
// main
// ===========================================================================
int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("wgmma_dtypes  —  %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("================================================================\n");

    bool all_pass = true;
    all_pass &= run("f16←f16×f16  m64n16k16  A=1 B=1", 1.f, 1.f, (float)K);
    all_pass &= run("f16←f16×f16  m64n16k16  A=2 B=3", 2.f, 3.f, (float)(2*3*K));

    printf("\n================================================================\n");
    printf("Overall: %s\n", all_pass ? "PASSED" : "FAILED");
    return all_pass ? 0 : 1;
}
