// =============================================================================
// wgmma_sync.cu
//
// Tests wgmma.commit_group / wgmma.wait_group semantics on SM90a.
//
// Three subtests using f16 m64n16k16 → f32, all-ones inputs:
//
//   one_group_two_mmas  – two MMA ops in the same committed group.
//                         Verifies D accumulates both before commit.
//                         Expected D[m][n] = 32.0
//
//   two_groups_serial   – two groups committed and waited one at a time.
//                         The inter-group fence ends group 0 and starts group 1.
//                         Expected D[m][n] = 32.0 (accumulated across both groups)
//
//   two_groups_pipeline – two groups committed before any wait (software pipeline).
//                         Tests the group queue: both groups in flight simultaneously
//                         before a single wait_group 0 drains them.
//                         Expected D[m][n] = 32.0
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
static constexpr int D_ELEMS = N / 2;
static constexpr int E       = 2;
static constexpr int SBO     = 128;
static constexpr int LBO_A   = M * 16;   // 1024
static constexpr int LBO_B   = N * 16;   // 256
static constexpr int SMEM_A  = M * K * E;   // 2048
static constexpr int SMEM_B  = K * N * E;   // 512

// ---------------------------------------------------------------------------
// Device helpers (identical to wgmma_verify / wgmma_fence)
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
// Kernel 1: two MMA ops in one committed group
//   fence → MMA_a → MMA_b → commit_group → wait_group_0 → fence
//   Both MMAs accumulate into the same D registers before commit.
// ===========================================================================
__global__ void kernel_one_group_two_mmas(const half* A_g, const half* B_g,
                                           float* D_g) {
    __shared__ __align__(128) char smA[SMEM_A], smB[SMEM_B];
    load_smem(smA, smB, A_g, B_g);

    uint64_t da = make_gmma_desc(smem_addr(smA), LBO_A, SBO);
    uint64_t db = make_gmma_desc(smem_addr(smB), LBO_B, SBO);
    float d0=0,d1=0,d2=0,d3=0,d4=0,d5=0,d6=0,d7=0;

    WGMMA_FENCE;
    mma_f16(da, db, d0,d1,d2,d3,d4,d5,d6,d7);   // contributes K = 16
    mma_f16(da, db, d0,d1,d2,d3,d4,d5,d6,d7);   // contributes K = 16 more
    WGMMA_COMMIT;
    WGMMA_WAIT; WGMMA_FENCE;

    const int b = threadIdx.x * D_ELEMS;
    D_g[b]=d0; D_g[b+1]=d1; D_g[b+2]=d2; D_g[b+3]=d3;
    D_g[b+4]=d4; D_g[b+5]=d5; D_g[b+6]=d6; D_g[b+7]=d7;
}

// ===========================================================================
// Kernel 2: two groups committed and waited serially
//   fence → MMA_a → commit → wait → fence → MMA_b → commit → wait → fence
//   The inter-group fence acts as both the post-wait fence (releasing D)
//   and the pre-MMA fence (signalling smem ready for the next MMA).
// ===========================================================================
__global__ void kernel_two_groups_serial(const half* A_g, const half* B_g,
                                          float* D_g) {
    __shared__ __align__(128) char smA[SMEM_A], smB[SMEM_B];
    load_smem(smA, smB, A_g, B_g);

    uint64_t da = make_gmma_desc(smem_addr(smA), LBO_A, SBO);
    uint64_t db = make_gmma_desc(smem_addr(smB), LBO_B, SBO);
    float d0=0,d1=0,d2=0,d3=0,d4=0,d5=0,d6=0,d7=0;

    // Group 0
    WGMMA_FENCE;
    mma_f16(da, db, d0,d1,d2,d3,d4,d5,d6,d7);
    WGMMA_COMMIT;
    WGMMA_WAIT;
    // Inter-group fence: ends group 0 (D readable) and starts group 1 (smem ready)
    WGMMA_FENCE;
    // Group 1: accumulates into same D
    mma_f16(da, db, d0,d1,d2,d3,d4,d5,d6,d7);
    WGMMA_COMMIT;
    WGMMA_WAIT; WGMMA_FENCE;

    const int b = threadIdx.x * D_ELEMS;
    D_g[b]=d0; D_g[b+1]=d1; D_g[b+2]=d2; D_g[b+3]=d3;
    D_g[b+4]=d4; D_g[b+5]=d5; D_g[b+6]=d6; D_g[b+7]=d7;
}

// ===========================================================================
// Kernel 3: two groups pipelined (both committed before any wait)
//   fence → MMA_a → commit_group_0 → MMA_b → commit_group_1 → wait_group_0 → fence
//   No fence between the two groups: smem is unchanged so no new fence is needed.
//   Tests the group queue: two in-flight groups drained by a single wait.
// ===========================================================================
__global__ void kernel_two_groups_pipeline(const half* A_g, const half* B_g,
                                            float* D_g) {
    __shared__ __align__(128) char smA[SMEM_A], smB[SMEM_B];
    load_smem(smA, smB, A_g, B_g);

    uint64_t da = make_gmma_desc(smem_addr(smA), LBO_A, SBO);
    uint64_t db = make_gmma_desc(smem_addr(smB), LBO_B, SBO);
    float d0=0,d1=0,d2=0,d3=0,d4=0,d5=0,d6=0,d7=0;

    WGMMA_FENCE;
    mma_f16(da, db, d0,d1,d2,d3,d4,d5,d6,d7);   // into group 0
    WGMMA_COMMIT;                                 // commit group 0
    mma_f16(da, db, d0,d1,d2,d3,d4,d5,d6,d7);   // into group 1 (no fence needed)
    WGMMA_COMMIT;                                 // commit group 1
    WGMMA_WAIT; WGMMA_FENCE;                     // drain both groups

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

static std::vector<float> run1(const std::vector<half>& h_A,
                                const std::vector<half>& h_B,
                                void (*kern)(const half*, const half*, float*)) {
    half  *d_A, *d_B;
    float *d_D;
    cudaMalloc(&d_A, h_A.size()*sizeof(half));
    cudaMalloc(&d_B, h_B.size()*sizeof(half));
    cudaMalloc(&d_D, WGSIZE*D_ELEMS*sizeof(float));
    cudaMemcpy(d_A, h_A.data(), h_A.size()*sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), h_B.size()*sizeof(half), cudaMemcpyHostToDevice);
    kern<<<1, WGSIZE>>>(d_A, d_B, d_D);
    cudaDeviceSynchronize();
    std::vector<float> flat(WGSIZE*D_ELEMS);
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
    printf("wgmma_sync  —  %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("================================================================\n");

    bool all_pass = true;

    std::vector<half> A(M*K), B(K*N);
    for (auto& v : A) v = __float2half(1.f);
    for (auto& v : B) v = __float2half(1.f);

    static float D[M][N];
    const float expected = (float)(2 * K);  // two accumulated MMAs × K ones

    printf("\n--- one_group_two_mmas ---\n");
    {
        auto flat = run1(A, B, kernel_one_group_two_mmas);
        reassemble_D(flat.data(), D);
        all_pass &= check_uniform("one_group_two_mmas", D, expected);
    }

    printf("\n--- two_groups_serial ---\n");
    {
        auto flat = run1(A, B, kernel_two_groups_serial);
        reassemble_D(flat.data(), D);
        all_pass &= check_uniform("two_groups_serial", D, expected);
    }

    printf("\n--- two_groups_pipeline ---\n");
    {
        auto flat = run1(A, B, kernel_two_groups_pipeline);
        reassemble_D(flat.data(), D);
        all_pass &= check_uniform("two_groups_pipeline", D, expected);
    }

    printf("\n================================================================\n");
    printf("Overall: %s\n", all_pass ? "PASSED" : "FAILED");
    return all_pass ? 0 : 1;
}
