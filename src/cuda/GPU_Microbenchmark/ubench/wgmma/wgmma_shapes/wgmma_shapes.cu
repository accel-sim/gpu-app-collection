// =============================================================================
// wgmma_shapes.cu
//
// Tests f16 m64nNk16 → f32 for N = 16, 32, 64.
//
// Motivation: different N values produce different D-register counts per thread
// and different fragment layouts.  wgmma_verify only covers N=16; this file
// stresses the larger shapes that exercise different simulator code paths.
//
//   N=16:  8 D-regs/thread  (baseline — same as wgmma_verify)
//   N=32: 16 D-regs/thread
//   N=64: 32 D-regs/thread
//
// N=128/256 follow the same fragment formula as N=64 (more regs, same layout
// logic); they are covered implicitly by wgmma_no_tma via CUTLASS.
//
// All tests use all-ones A and B.
// Expected D[m][n] = K = 16 for every (m, n) regardless of N.
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

static constexpr int M      = 64;
static constexpr int K      = 16;
static constexpr int WGSIZE = 128;
static constexpr int SBO    = 128;
static constexpr int E      = 2;    // bytes per f16

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

// ---------------------------------------------------------------------------
// Kernel: f16 m64n16k16 (8 D-regs — baseline)
// ---------------------------------------------------------------------------
__global__ void kernel_n16(const half* A_g, const half* B_g, float* D_g) {
    constexpr int N = 16, D_ELEMS = N/2;
    constexpr int LBO_A = M*16, LBO_B = N*16;
    __shared__ __align__(128) char smA[M*K*E], smB[K*N*E];
    const int tid = threadIdx.x;
    for (int i = tid; i < M*K; i += WGSIZE) { int m=i/K,k=i%K; *(half*)(smA+smem_off(k,m,E,LBO_A))=A_g[i]; }
    for (int i = tid; i < K*N; i += WGSIZE) { int k=i/N,n=i%N; *(half*)(smB+smem_off(k,n,E,LBO_B))=B_g[i]; }
    __syncthreads();
    uint64_t da=make_gmma_desc(smem_addr(smA),LBO_A,SBO);
    uint64_t db=make_gmma_desc(smem_addr(smB),LBO_B,SBO);
    float d0=0,d1=0,d2=0,d3=0,d4=0,d5=0,d6=0,d7=0;
    WGMMA_FENCE;
    asm volatile(
        "{ .reg .pred p; setp.ne.b32 p,1,0;\n"
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7},%8,%9,p,1,1,0,0; }\n"
        :"+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3),"+f"(d4),"+f"(d5),"+f"(d6),"+f"(d7)
        :"l"(da),"l"(db):"memory");
    WGMMA_COMMIT; WGMMA_WAIT; WGMMA_FENCE;
    int b=tid*D_ELEMS;
    D_g[b]=d0;D_g[b+1]=d1;D_g[b+2]=d2;D_g[b+3]=d3;
    D_g[b+4]=d4;D_g[b+5]=d5;D_g[b+6]=d6;D_g[b+7]=d7;
}

// ---------------------------------------------------------------------------
// Kernel: f16 m64n32k16 (16 D-regs)
// ---------------------------------------------------------------------------
__global__ void kernel_n32(const half* A_g, const half* B_g, float* D_g) {
    constexpr int N = 32, D_ELEMS = N/2;
    constexpr int LBO_A = M*16, LBO_B = N*16;
    __shared__ __align__(128) char smA[M*K*E], smB[K*N*E];
    const int tid = threadIdx.x;
    for (int i = tid; i < M*K; i += WGSIZE) { int m=i/K,k=i%K; *(half*)(smA+smem_off(k,m,E,LBO_A))=A_g[i]; }
    for (int i = tid; i < K*N; i += WGSIZE) { int k=i/N,n=i%N; *(half*)(smB+smem_off(k,n,E,LBO_B))=B_g[i]; }
    __syncthreads();
    uint64_t da=make_gmma_desc(smem_addr(smA),LBO_A,SBO);
    uint64_t db=make_gmma_desc(smem_addr(smB),LBO_B,SBO);
    float d0=0,d1=0,d2=0,d3=0,d4=0,d5=0,d6=0,d7=0;
    float d8=0,d9=0,d10=0,d11=0,d12=0,d13=0,d14=0,d15=0;
    WGMMA_FENCE;
    asm volatile(
        "{ .reg .pred p; setp.ne.b32 p,1,0;\n"
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15},%16,%17,p,1,1,0,0; }\n"
        :"+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3),"+f"(d4),"+f"(d5),"+f"(d6),"+f"(d7),
         "+f"(d8),"+f"(d9),"+f"(d10),"+f"(d11),"+f"(d12),"+f"(d13),"+f"(d14),"+f"(d15)
        :"l"(da),"l"(db):"memory");
    WGMMA_COMMIT; WGMMA_WAIT; WGMMA_FENCE;
    int b=tid*D_ELEMS;
    D_g[b]=d0;D_g[b+1]=d1;D_g[b+2]=d2;D_g[b+3]=d3;
    D_g[b+4]=d4;D_g[b+5]=d5;D_g[b+6]=d6;D_g[b+7]=d7;
    D_g[b+8]=d8;D_g[b+9]=d9;D_g[b+10]=d10;D_g[b+11]=d11;
    D_g[b+12]=d12;D_g[b+13]=d13;D_g[b+14]=d14;D_g[b+15]=d15;
}

// ---------------------------------------------------------------------------
// Kernel: f16 m64n64k16 (32 D-regs)
// ---------------------------------------------------------------------------
__global__ void kernel_n64(const half* A_g, const half* B_g, float* D_g) {
    constexpr int N = 64, D_ELEMS = N/2;
    constexpr int LBO_A = M*16, LBO_B = N*16;
    __shared__ __align__(128) char smA[M*K*E], smB[K*N*E];
    const int tid = threadIdx.x;
    for (int i = tid; i < M*K; i += WGSIZE) { int m=i/K,k=i%K; *(half*)(smA+smem_off(k,m,E,LBO_A))=A_g[i]; }
    for (int i = tid; i < K*N; i += WGSIZE) { int k=i/N,n=i%N; *(half*)(smB+smem_off(k,n,E,LBO_B))=B_g[i]; }
    __syncthreads();
    uint64_t da=make_gmma_desc(smem_addr(smA),LBO_A,SBO);
    uint64_t db=make_gmma_desc(smem_addr(smB),LBO_B,SBO);
    float d0=0,d1=0,d2=0,d3=0,d4=0,d5=0,d6=0,d7=0;
    float d8=0,d9=0,d10=0,d11=0,d12=0,d13=0,d14=0,d15=0;
    float d16=0,d17=0,d18=0,d19=0,d20=0,d21=0,d22=0,d23=0;
    float d24=0,d25=0,d26=0,d27=0,d28=0,d29=0,d30=0,d31=0;
    WGMMA_FENCE;
    asm volatile(
        "{ .reg .pred p; setp.ne.b32 p,1,0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},%32,%33,p,1,1,0,0; }\n"
        :"+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3),"+f"(d4),"+f"(d5),"+f"(d6),"+f"(d7),
         "+f"(d8),"+f"(d9),"+f"(d10),"+f"(d11),"+f"(d12),"+f"(d13),"+f"(d14),"+f"(d15),
         "+f"(d16),"+f"(d17),"+f"(d18),"+f"(d19),"+f"(d20),"+f"(d21),"+f"(d22),"+f"(d23),
         "+f"(d24),"+f"(d25),"+f"(d26),"+f"(d27),"+f"(d28),"+f"(d29),"+f"(d30),"+f"(d31)
        :"l"(da),"l"(db):"memory");
    WGMMA_COMMIT; WGMMA_WAIT; WGMMA_FENCE;
    int b=tid*D_ELEMS;
    D_g[b]=d0;D_g[b+1]=d1;D_g[b+2]=d2;D_g[b+3]=d3;
    D_g[b+4]=d4;D_g[b+5]=d5;D_g[b+6]=d6;D_g[b+7]=d7;
    D_g[b+8]=d8;D_g[b+9]=d9;D_g[b+10]=d10;D_g[b+11]=d11;
    D_g[b+12]=d12;D_g[b+13]=d13;D_g[b+14]=d14;D_g[b+15]=d15;
    D_g[b+16]=d16;D_g[b+17]=d17;D_g[b+18]=d18;D_g[b+19]=d19;
    D_g[b+20]=d20;D_g[b+21]=d21;D_g[b+22]=d22;D_g[b+23]=d23;
    D_g[b+24]=d24;D_g[b+25]=d25;D_g[b+26]=d26;D_g[b+27]=d27;
    D_g[b+28]=d28;D_g[b+29]=d29;D_g[b+30]=d30;D_g[b+31]=d31;
}

// ===========================================================================
// Host helpers
// ===========================================================================

// Fragment layout formula is the same for all N.
// e ranges 0..(N/2-1); s=e/4 indexes groups of 8 columns.
static void reassemble(const float* flat, float* D, int N) {
    const int D_ELEMS = N / 2;
    for (int i = 0; i < M*N; i++) D[i] = 0.f;
    for (int T = 0; T < WGSIZE; T++) {
        int warp = T/32, lane = T%32;
        for (int e = 0; e < D_ELEMS; e++) {
            int s = e/4, k = e%4;
            int row = (lane/4)*2 + k/2 + warp*16;
            int col = (lane%4)*2 + k%2 + s*8;
            if (row < M && col < N)
                D[row*N + col] = flat[T*D_ELEMS + e];
        }
    }
}

static bool check_uniform(const char* name, const float* D, int N, float expected) {
    float max_err = 0.f;
    int   worst   = 0;
    for (int i = 0; i < M*N; i++) {
        float e = fabsf(D[i] - expected);
        if (e > max_err) { max_err = e; worst = i; }
    }
    float tol  = (fabsf(expected) < 1.f) ? 1e-3f : fabsf(expected) * 1e-2f;
    bool  pass = (max_err <= tol);
    printf("[%-38s] %s  expected=%.1f  max_err=%.4e\n",
           name, pass ? "PASS" : "FAIL", expected, max_err);
    if (!pass)
        printf("  Worst: D[%d][%d] = %.6f  expected = %.6f\n",
               worst/N, worst%N, D[worst], expected);
    return pass;
}

struct KernFn { void (*fn)(const half*, const half*, float*); int N; const char* name; };

static bool run(KernFn kf) {
    const int D_ELEMS = kf.N / 2;
    std::vector<half>  hA(M*K, __float2half(1.f));
    std::vector<half>  hB(K*kf.N, __float2half(1.f));
    half  *dA, *dB; float *dD;
    cudaMalloc(&dA, M*K*sizeof(half));
    cudaMalloc(&dB, K*kf.N*sizeof(half));
    cudaMalloc(&dD, WGSIZE*D_ELEMS*sizeof(float));
    cudaMemcpy(dA, hA.data(), hA.size()*sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB.data(), hB.size()*sizeof(half), cudaMemcpyHostToDevice);
    kf.fn<<<1, WGSIZE>>>(dA, dB, dD);
    cudaDeviceSynchronize();
    std::vector<float> flat(WGSIZE*D_ELEMS);
    cudaMemcpy(flat.data(), dD, flat.size()*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dA); cudaFree(dB); cudaFree(dD);
    std::vector<float> D(M*kf.N);
    reassemble(flat.data(), D.data(), kf.N);
    return check_uniform(kf.name, D.data(), kf.N, (float)K);
}

// ===========================================================================
// main
// ===========================================================================
int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("wgmma_shapes  —  %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("================================================================\n");

    bool all_pass = true;
    KernFn tests[] = {
        { kernel_n16, 16, "f16 m64n16k16 (8 D-regs/thread)"  },
        { kernel_n32, 32, "f16 m64n32k16 (16 D-regs/thread)" },
        { kernel_n64, 64, "f16 m64n64k16 (32 D-regs/thread)" },
    };
    for (auto& t : tests) all_pass &= run(t);

    printf("\n================================================================\n");
    printf("Overall: %s\n", all_pass ? "PASSED" : "FAILED");
    return all_pass ? 0 : 1;
}
