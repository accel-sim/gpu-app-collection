// =============================================================================
// wgmma_fence.cu
//
// Tests wgmma.fence.sync.aligned semantics on SM90a.
//
// Three subtests using f16 m64n16k16 → f32:
//
//   fence_single   – one fence→MMA→commit→wait_group_0→fence cycle.
//                    Baseline: verifies MMA sees smem data written before fence.
//                    Expected D[m][n] = 16.0 (K ones × 1×1).
//
//   fence_sequence – two back-to-back cycles in one kernel with different smem
//                    data.  The second fence gates a fresh smem write, so the
//                    second MMA must produce a different result from the first.
//                    Expected D1[m][n] = 16.0  (A=1, B=1)
//                    Expected D2[m][n] = 64.0  (A=2, B=2, K × 2×2)
//
//   fence_accum    – two cycles accumulating into the same D registers.
//                    Verifies that D is not accidentally reset between cycles
//                    and that the post-wait fence properly releases D.
//                    Expected D[m][n] = 32.0  (2 × 16.0)
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

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static constexpr int M       = 64;
static constexpr int N       = 16;
static constexpr int K       = 16;
static constexpr int WGSIZE  = 128;
static constexpr int D_ELEMS = N / 2;    // 8 D-regs per thread
static constexpr int E       = 2;        // bytes per f16
static constexpr int SBO     = 128;
static constexpr int LBO_A   = M * E * 8;   // 1024
static constexpr int LBO_B   = N * E * 8;   // 256
static constexpr int SMEM_A  = M * K * E;   // 2048
static constexpr int SMEM_B  = K * N * E;   // 512

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint32_t smem_addr(const void* ptr) {
    uint32_t a;
    asm volatile(
        "{ .reg .u64 _p; cvta.to.shared.u64 _p, %1; cvt.u32.u64 %0, _p; }\n"
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

// ---------------------------------------------------------------------------
// WGMMA protocol macros
// ---------------------------------------------------------------------------
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
// Kernel helpers: load A/B globals into K-major smem (no swizzle)
// ---------------------------------------------------------------------------
__device__ void load_smem(char* smA, char* smB,
                           const half* A_g, const half* B_g) {
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
}

// ===========================================================================
// Kernel 1: single fence→MMA→commit→wait_group_0→fence
// ===========================================================================
__global__ void kernel_fence_single(const half* A_g, const half* B_g, float* D_g) {
    __shared__ __align__(128) char smA[SMEM_A], smB[SMEM_B];
    load_smem(smA, smB, A_g, B_g);

    uint64_t da = make_gmma_desc(smem_addr(smA), LBO_A, SBO);
    uint64_t db = make_gmma_desc(smem_addr(smB), LBO_B, SBO);
    float d0=0,d1=0,d2=0,d3=0,d4=0,d5=0,d6=0,d7=0;

    WGMMA_FENCE;
    mma_f16(da, db, d0,d1,d2,d3,d4,d5,d6,d7);
    WGMMA_COMMIT; WGMMA_WAIT; WGMMA_FENCE;

    const int b = threadIdx.x * D_ELEMS;
    D_g[b]=d0; D_g[b+1]=d1; D_g[b+2]=d2; D_g[b+3]=d3;
    D_g[b+4]=d4; D_g[b+5]=d5; D_g[b+6]=d6; D_g[b+7]=d7;
}

// ===========================================================================
// Kernel 2: two back-to-back fence→MMA→commit→wait→fence sequences.
//   Sequence 1 uses A_g1, B_g1 (all 1.0).  Sequence 2 uses A_g2, B_g2 (all 2.0).
//   Outputs D_g1 and D_g2 separately for independent checking.
// ===========================================================================
__global__ void kernel_fence_sequence(const half* A_g1, const half* B_g1,
                                       const half* A_g2, const half* B_g2,
                                       float* D_g1, float* D_g2) {
    __shared__ __align__(128) char smA[SMEM_A], smB[SMEM_B];
    const int tid = threadIdx.x;

    // Sequence 1
    load_smem(smA, smB, A_g1, B_g1);
    uint64_t da = make_gmma_desc(smem_addr(smA), LBO_A, SBO);
    uint64_t db = make_gmma_desc(smem_addr(smB), LBO_B, SBO);
    float d0=0,d1=0,d2=0,d3=0,d4=0,d5=0,d6=0,d7=0;
    WGMMA_FENCE;
    mma_f16(da, db, d0,d1,d2,d3,d4,d5,d6,d7);
    WGMMA_COMMIT; WGMMA_WAIT; WGMMA_FENCE;
    const int b = tid * D_ELEMS;
    D_g1[b]=d0; D_g1[b+1]=d1; D_g1[b+2]=d2; D_g1[b+3]=d3;
    D_g1[b+4]=d4; D_g1[b+5]=d5; D_g1[b+6]=d6; D_g1[b+7]=d7;

    // Overwrite smem with sequence-2 data, then re-fence
    __syncthreads();
    load_smem(smA, smB, A_g2, B_g2);
    float e0=0,e1=0,e2=0,e3=0,e4=0,e5=0,e6=0,e7=0;
    WGMMA_FENCE;
    mma_f16(da, db, e0,e1,e2,e3,e4,e5,e6,e7);
    WGMMA_COMMIT; WGMMA_WAIT; WGMMA_FENCE;
    D_g2[b]=e0; D_g2[b+1]=e1; D_g2[b+2]=e2; D_g2[b+3]=e3;
    D_g2[b+4]=e4; D_g2[b+5]=e5; D_g2[b+6]=e6; D_g2[b+7]=e7;
}

// ===========================================================================
// Kernel 3: two accumulating cycles using the same smem.
//   The post-wait fence releases D so the second cycle can start immediately.
//   Expected: D[m][n] = 32.0  (2 × K × 1 × 1).
// ===========================================================================
__global__ void kernel_fence_accum(const half* A_g, const half* B_g, float* D_g) {
    __shared__ __align__(128) char smA[SMEM_A], smB[SMEM_B];
    load_smem(smA, smB, A_g, B_g);

    uint64_t da = make_gmma_desc(smem_addr(smA), LBO_A, SBO);
    uint64_t db = make_gmma_desc(smem_addr(smB), LBO_B, SBO);
    float d0=0,d1=0,d2=0,d3=0,d4=0,d5=0,d6=0,d7=0;

    // Cycle 1
    WGMMA_FENCE;
    mma_f16(da, db, d0,d1,d2,d3,d4,d5,d6,d7);
    WGMMA_COMMIT; WGMMA_WAIT;
    // Single fence serves as end-of-cycle-1 and start-of-cycle-2
    WGMMA_FENCE;
    // Cycle 2: accumulate into same D registers
    mma_f16(da, db, d0,d1,d2,d3,d4,d5,d6,d7);
    WGMMA_COMMIT; WGMMA_WAIT; WGMMA_FENCE;

    const int b = threadIdx.x * D_ELEMS;
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

static void reassemble_D(const float* flat, float D[M][N]) {
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) D[m][n] = 0.f;
    for (int T = 0; T < WGSIZE; T++)
        for (int e = 0; e < D_ELEMS; e++) {
            int row, col;
            d_frag_pos(T, e, &row, &col);
            if (row < M && col < N) D[row][col] = flat[T*D_ELEMS + e];
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
    float tol  = (fabsf(expected) < 1.f) ? 1e-3f : fabsf(expected) * 1e-2f;
    bool  pass = (max_err <= tol);
    printf("[%-40s] %s  expected=%.1f  max_err=%.4e\n",
           name, pass ? "PASS" : "FAIL", expected, max_err);
    if (!pass)
        printf("  Worst: D[%d][%d] = %.6f  expected = %.6f\n",
               max_m, max_n, D[max_m][max_n], expected);
    return pass;
}

// Allocate device buffers, copy h_A/h_B, run kernel, retrieve result.
// Returns a flat WGSIZE*D_ELEMS float vector.
static std::vector<float> run1(const std::vector<half>& h_A,
                                const std::vector<half>& h_B,
                                void (*kern)(const half*, const half*, float*)) {
    half  *d_A, *d_B;
    float *d_D;
    cudaMalloc(&d_A, h_A.size() * sizeof(half));
    cudaMalloc(&d_B, h_B.size() * sizeof(half));
    cudaMalloc(&d_D, WGSIZE * D_ELEMS * sizeof(float));
    cudaMemcpy(d_A, h_A.data(), h_A.size()*sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), h_B.size()*sizeof(half), cudaMemcpyHostToDevice);
    kern<<<1, WGSIZE>>>(d_A, d_B, d_D);
    cudaDeviceSynchronize();
    std::vector<float> flat(WGSIZE * D_ELEMS);
    cudaMemcpy(flat.data(), d_D, flat.size()*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_D);
    return flat;
}

// ===========================================================================
// main
// ===========================================================================
int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("wgmma_fence  —  %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("================================================================\n");

    bool all_pass = true;

    // Host matrices: all-ones (A1=B1=1.0), all-twos (A2=B2=2.0)
    std::vector<half> A1(M*K), B1(K*N), A2(M*K), B2(K*N);
    for (auto& v : A1) v = __float2half(1.f);
    for (auto& v : B1) v = __float2half(1.f);
    for (auto& v : A2) v = __float2half(2.f);
    for (auto& v : B2) v = __float2half(2.f);

    static float D[M][N];

    // -------------------------------------------------------------------------
    // Test 1: fence_single
    // -------------------------------------------------------------------------
    printf("\n--- fence_single ---\n");
    {
        auto flat = run1(A1, B1, kernel_fence_single);
        reassemble_D(flat.data(), D);
        all_pass &= check_uniform("fence_single", D, (float)K);
    }

    // -------------------------------------------------------------------------
    // Test 2: fence_sequence (two separate cycles, independent D)
    // -------------------------------------------------------------------------
    printf("\n--- fence_sequence ---\n");
    {
        half  *dA1, *dB1, *dA2, *dB2;
        float *dD1, *dD2;
        cudaMalloc(&dA1, M*K*sizeof(half)); cudaMalloc(&dB1, K*N*sizeof(half));
        cudaMalloc(&dA2, M*K*sizeof(half)); cudaMalloc(&dB2, K*N*sizeof(half));
        cudaMalloc(&dD1, WGSIZE*D_ELEMS*sizeof(float));
        cudaMalloc(&dD2, WGSIZE*D_ELEMS*sizeof(float));
        cudaMemcpy(dA1, A1.data(), M*K*sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(dB1, B1.data(), K*N*sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(dA2, A2.data(), M*K*sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(dB2, B2.data(), K*N*sizeof(half), cudaMemcpyHostToDevice);

        kernel_fence_sequence<<<1, WGSIZE>>>(dA1, dB1, dA2, dB2, dD1, dD2);
        cudaDeviceSynchronize();

        std::vector<float> flat1(WGSIZE*D_ELEMS), flat2(WGSIZE*D_ELEMS);
        cudaMemcpy(flat1.data(), dD1, flat1.size()*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(flat2.data(), dD2, flat2.size()*sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(dA1); cudaFree(dB1); cudaFree(dA2); cudaFree(dB2);
        cudaFree(dD1); cudaFree(dD2);

        static float D2[M][N];
        reassemble_D(flat1.data(), D);
        reassemble_D(flat2.data(), D2);
        all_pass &= check_uniform("fence_sequence seq1 (A=1,B=1)", D,  (float)K);
        all_pass &= check_uniform("fence_sequence seq2 (A=2,B=2)", D2, (float)(K*4));
    }

    // -------------------------------------------------------------------------
    // Test 3: fence_accum (two cycles accumulate into same D)
    // -------------------------------------------------------------------------
    printf("\n--- fence_accum ---\n");
    {
        auto flat = run1(A1, B1, kernel_fence_accum);
        reassemble_D(flat.data(), D);
        all_pass &= check_uniform("fence_accum (2 × K)", D, (float)(2*K));
    }

    printf("\n================================================================\n");
    printf("Overall: %s\n", all_pass ? "PASSED" : "FAILED");
    return all_pass ? 0 : 1;
}
