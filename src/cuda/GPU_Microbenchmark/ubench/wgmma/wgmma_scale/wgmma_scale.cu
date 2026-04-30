// =============================================================================
// wgmma_scale.cu
//
// Tests WGMMA scale / accumulator-init parameters on SM90a.
//
// Two parameter axes:
//
//   scaleD (predicate) — controls whether D is used as accumulator:
//     scaleD=1 (true):  D_out = D_in + A*B
//     scaleD=0 (false): D_out = A*B  (D_in is ignored)
//   Applies to all WGMMA types.  Tested here with f16.
//
//   scaleA / scaleB — negate A or B operand (fp8 types only):
//     scaleA=-1: D_out = (-A)*B = -(A*B)
//     scaleB=-1: D_out = A*(-B) = -(A*B)
//     both=-1:   D_out = (-A)*(-B) = +(A*B)
//   Tested with e4m3 m64n16k32.
//
// Tests:
//   scaleD=0 (f16)        : D pre-init to 50, expect D = K    = 16
//   scaleD=1 (f16)        : D pre-init to K,  expect D = 2K   = 32
//   e4m3 scaleA=-1        : all-ones A,B,     expect D = -K32 = -32
//   e4m3 scaleB=-1        : all-ones A,B,     expect D = -K32 = -32
//   e4m3 scaleA=-1 scaleB=-1: all-ones,       expect D = +K32 = +32
//
// Exit status: 0 = all PASS, 1 = any FAIL.
// =============================================================================

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstring>

static constexpr int M      = 64;
static constexpr int N      = 16;
static constexpr int WGSIZE = 128;
static constexpr int D_ELEMS = N / 2;   // 8
static constexpr int SBO    = 128;
static constexpr int E_F16  = 2;
static constexpr int E_FP8  = 1;
static constexpr int K_F16  = 16;
static constexpr int K_FP8  = 32;

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

#define WGMMA_FENCE  asm volatile("wgmma.fence.sync.aligned;\n"        ::: "memory")
#define WGMMA_COMMIT asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory")
#define WGMMA_WAIT   asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory")

// ===========================================================================
// scaleD tests (f16)
// ===========================================================================

// D pre-initialized to init_val; scaleD selects whether D_in is used.
// Template parameter SCALE_D: 1 = accumulate, 0 = ignore D_in.
template<int SCALE_D>
__global__ void kernel_f16_scaled(const half* A_g, const half* B_g,
                                   float init_val, float* D_g) {
    constexpr int LBO_A = M * E_F16 * 8;
    constexpr int LBO_B = N * E_F16 * 8;
    __shared__ __align__(128) char smA[M*K_F16*E_F16], smB[K_F16*N*E_F16];
    const int tid = threadIdx.x;
    for (int i = tid; i < M*K_F16; i += WGSIZE) { int m=i/K_F16,k=i%K_F16; *(half*)(smA+smem_off(k,m,E_F16,LBO_A))=A_g[i]; }
    for (int i = tid; i < K_F16*N; i += WGSIZE) { int k=i/N,n=i%N;         *(half*)(smB+smem_off(k,n,E_F16,LBO_B))=B_g[i]; }
    __syncthreads();
    uint64_t da = make_gmma_desc(smem_addr(smA), LBO_A, SBO);
    uint64_t db = make_gmma_desc(smem_addr(smB), LBO_B, SBO);
    // Pre-initialize D to init_val to test whether scaleD=0 ignores it.
    float d0=init_val,d1=init_val,d2=init_val,d3=init_val;
    float d4=init_val,d5=init_val,d6=init_val,d7=init_val;
    WGMMA_FENCE;
    if constexpr (SCALE_D == 1) {
        asm volatile(
            "{ .reg .pred p; setp.ne.b32 p,1,0;\n"   // p = true  → accumulate
            "wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 "
            "{%0,%1,%2,%3,%4,%5,%6,%7},%8,%9,p,1,1,0,0; }\n"
            :"+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3),"+f"(d4),"+f"(d5),"+f"(d6),"+f"(d7)
            :"l"(da),"l"(db):"memory");
    } else {
        asm volatile(
            "{ .reg .pred p; setp.ne.b32 p,0,0;\n"   // p = false → D_in ignored
            "wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 "
            "{%0,%1,%2,%3,%4,%5,%6,%7},%8,%9,p,1,1,0,0; }\n"
            :"+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3),"+f"(d4),"+f"(d5),"+f"(d6),"+f"(d7)
            :"l"(da),"l"(db):"memory");
    }
    WGMMA_COMMIT; WGMMA_WAIT; WGMMA_FENCE;
    int b = tid * D_ELEMS;
    D_g[b]=d0;D_g[b+1]=d1;D_g[b+2]=d2;D_g[b+3]=d3;
    D_g[b+4]=d4;D_g[b+5]=d5;D_g[b+6]=d6;D_g[b+7]=d7;
}

// ===========================================================================
// scaleA / scaleB tests (fp8 e4m3)
// SA, SB: compile-time literal, must be 1 or -1
// ===========================================================================
template<int SA, int SB>
__global__ void kernel_e4m3_scale(const uint8_t* A_g, const uint8_t* B_g,
                                   float* D_g) {
    constexpr int LBO_A = M * E_FP8 * 8;
    constexpr int LBO_B = N * E_FP8 * 8;
    __shared__ __align__(128) char smA[M*K_FP8*E_FP8], smB[K_FP8*N*E_FP8];
    const int tid = threadIdx.x;
    for (int i = tid; i < M*K_FP8; i += WGSIZE) { int m=i/K_FP8,k=i%K_FP8; *(uint8_t*)(smA+smem_off(k,m,E_FP8,LBO_A))=A_g[i]; }
    for (int i = tid; i < K_FP8*N; i += WGSIZE) { int k=i/N,n=i%N;          *(uint8_t*)(smB+smem_off(k,n,E_FP8,LBO_B))=B_g[i]; }
    __syncthreads();
    uint64_t da = make_gmma_desc(smem_addr(smA), LBO_A, SBO);
    uint64_t db = make_gmma_desc(smem_addr(smB), LBO_B, SBO);
    float d0=0,d1=0,d2=0,d3=0,d4=0,d5=0,d6=0,d7=0;
    WGMMA_FENCE;
    // SA and SB are template parameters, so the compiler emits a constant
    // literal in the PTX instruction (required by the ISA).
    if constexpr (SA == 1 && SB == 1) {
        asm volatile(
            "{ .reg .pred p; setp.ne.b32 p,1,0;\n"
            "wgmma.mma_async.sync.aligned.m64n16k32.f32.e4m3.e4m3 "
            "{%0,%1,%2,%3,%4,%5,%6,%7},%8,%9,p,1,1; }\n"
            :"+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3),"+f"(d4),"+f"(d5),"+f"(d6),"+f"(d7)
            :"l"(da),"l"(db):"memory");
    } else if constexpr (SA == -1 && SB == 1) {
        asm volatile(
            "{ .reg .pred p; setp.ne.b32 p,1,0;\n"
            "wgmma.mma_async.sync.aligned.m64n16k32.f32.e4m3.e4m3 "
            "{%0,%1,%2,%3,%4,%5,%6,%7},%8,%9,p,-1,1; }\n"
            :"+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3),"+f"(d4),"+f"(d5),"+f"(d6),"+f"(d7)
            :"l"(da),"l"(db):"memory");
    } else if constexpr (SA == 1 && SB == -1) {
        asm volatile(
            "{ .reg .pred p; setp.ne.b32 p,1,0;\n"
            "wgmma.mma_async.sync.aligned.m64n16k32.f32.e4m3.e4m3 "
            "{%0,%1,%2,%3,%4,%5,%6,%7},%8,%9,p,1,-1; }\n"
            :"+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3),"+f"(d4),"+f"(d5),"+f"(d6),"+f"(d7)
            :"l"(da),"l"(db):"memory");
    } else {
        asm volatile(
            "{ .reg .pred p; setp.ne.b32 p,1,0;\n"
            "wgmma.mma_async.sync.aligned.m64n16k32.f32.e4m3.e4m3 "
            "{%0,%1,%2,%3,%4,%5,%6,%7},%8,%9,p,-1,-1; }\n"
            :"+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3),"+f"(d4),"+f"(d5),"+f"(d6),"+f"(d7)
            :"l"(da),"l"(db):"memory");
    }
    WGMMA_COMMIT; WGMMA_WAIT; WGMMA_FENCE;
    int b = tid * D_ELEMS;
    D_g[b]=d0;D_g[b+1]=d1;D_g[b+2]=d2;D_g[b+3]=d3;
    D_g[b+4]=d4;D_g[b+5]=d5;D_g[b+6]=d6;D_g[b+7]=d7;
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
            int row, col; d_frag_pos(T, e, &row, &col);
            if (row < M && col < N) D[row][col] = flat[T*D_ELEMS + e];
        }
}

static bool check_uniform(const char* name, float D[M][N], float expected) {
    float max_err = 0.f; int max_m = 0, max_n = 0;
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float e = fabsf(D[m][n] - expected);
            if (e > max_err) { max_err = e; max_m = m; max_n = n; }
        }
    float tol  = (fabsf(expected) < 1.f) ? 1e-3f : fabsf(expected) * 1e-2f;
    bool  pass = (max_err <= tol);
    printf("[%-42s] %s  expected=%.1f  max_err=%.4e\n",
           name, pass ? "PASS" : "FAIL", expected, max_err);
    if (!pass)
        printf("  Worst: D[%d][%d] = %.6f  expected = %.6f\n",
               max_m, max_n, D[max_m][max_n], expected);
    return pass;
}

// Convert float to raw fp8 e4m3 byte (same as wgmma_verify)
static uint8_t float_to_e4m3(float f) {
    __nv_fp8_e4m3 v(f); uint8_t b; memcpy(&b, &v, 1); return b;
}

// ===========================================================================
// main
// ===========================================================================
int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("wgmma_scale  —  %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("================================================================\n");

    bool all_pass = true;
    static float D[M][N];

    // -------------------------------------------------------------------------
    // scaleD tests (f16)
    // -------------------------------------------------------------------------
    printf("\n--- scaleD (f16 m64n16k16) ---\n");
    {
        std::vector<half> hA(M*K_F16, __float2half(1.f));
        std::vector<half> hB(K_F16*N, __float2half(1.f));
        half  *dA, *dB; float *dD;
        cudaMalloc(&dA, hA.size()*sizeof(half));
        cudaMalloc(&dB, hB.size()*sizeof(half));
        cudaMalloc(&dD, WGSIZE*D_ELEMS*sizeof(float));
        cudaMemcpy(dA, hA.data(), hA.size()*sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(dB, hB.data(), hB.size()*sizeof(half), cudaMemcpyHostToDevice);

        // scaleD=0: D_in=50 is ignored → result = A*B = K_F16 = 16
        kernel_f16_scaled<0><<<1, WGSIZE>>>(dA, dB, 50.f, dD);
        cudaDeviceSynchronize();
        std::vector<float> flat(WGSIZE*D_ELEMS);
        cudaMemcpy(flat.data(), dD, flat.size()*sizeof(float), cudaMemcpyDeviceToHost);
        reassemble(flat.data(), D);
        all_pass &= check_uniform("f16 scaleD=0 (D_in=50 ignored)", D, (float)K_F16);

        // scaleD=1: D_in=K_F16=16, accumulates → result = 16+16 = 32
        kernel_f16_scaled<1><<<1, WGSIZE>>>(dA, dB, (float)K_F16, dD);
        cudaDeviceSynchronize();
        cudaMemcpy(flat.data(), dD, flat.size()*sizeof(float), cudaMemcpyDeviceToHost);
        reassemble(flat.data(), D);
        all_pass &= check_uniform("f16 scaleD=1 (D_in=K accumulated)", D, (float)(2*K_F16));

        cudaFree(dA); cudaFree(dB); cudaFree(dD);
    }

    // -------------------------------------------------------------------------
    // scaleA / scaleB tests (fp8 e4m3, all-ones A and B)
    // K_FP8=32, all-ones: baseline D[m][n] = 32, negated = -32, double-neg = +32
    // -------------------------------------------------------------------------
    printf("\n--- scaleA/scaleB (e4m3 m64n16k32, A=B=1.0) ---\n");
    {
        uint8_t raw = float_to_e4m3(1.f);
        std::vector<uint8_t> hA(M*K_FP8, raw), hB(K_FP8*N, raw);
        uint8_t *dA, *dB; float *dD;
        cudaMalloc(&dA, hA.size());
        cudaMalloc(&dB, hB.size());
        cudaMalloc(&dD, WGSIZE*D_ELEMS*sizeof(float));
        cudaMemcpy(dA, hA.data(), hA.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(dB, hB.data(), hB.size(), cudaMemcpyHostToDevice);
        std::vector<float> flat(WGSIZE*D_ELEMS);

        auto run_fp8 = [&](auto kern, float expected, const char* name) {
            kern<<<1, WGSIZE>>>(dA, dB, dD);
            cudaDeviceSynchronize();
            cudaMemcpy(flat.data(), dD, flat.size()*sizeof(float), cudaMemcpyDeviceToHost);
            reassemble(flat.data(), D);
            return check_uniform(name, D, expected);
        };

        all_pass &= run_fp8(kernel_e4m3_scale<1,1>,   (float) K_FP8, "e4m3 scaleA=+1 scaleB=+1 (baseline)");
        all_pass &= run_fp8(kernel_e4m3_scale<-1,1>,  (float)-K_FP8, "e4m3 scaleA=-1 scaleB=+1");
        all_pass &= run_fp8(kernel_e4m3_scale<1,-1>,  (float)-K_FP8, "e4m3 scaleA=+1 scaleB=-1");
        all_pass &= run_fp8(kernel_e4m3_scale<-1,-1>, (float) K_FP8, "e4m3 scaleA=-1 scaleB=-1 (double-neg)");

        cudaFree(dA); cudaFree(dB); cudaFree(dD);
    }

    printf("\n================================================================\n");
    printf("Overall: %s\n", all_pass ? "PASS" : "FAIL");
    return all_pass ? 0 : 1;
}
