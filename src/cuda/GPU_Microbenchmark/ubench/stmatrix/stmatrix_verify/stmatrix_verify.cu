// =============================================================================
// stmatrix_verify.cu
//
// Verifies stmatrix.sync.aligned.m8n8.shared.b16 for all Hopper sm_90a variants:
//   x1, x2, x4  ×  non-transposed and .trans
//
// Matrix model:  M_m[r][c] = (uint16_t)(m*64 + r*8 + c)   (values 0..255)
//
// Non-transposed layout: smem[m*64 + r*8 + c] = M_m[r][c]  (row-major)
// Transposed layout:     smem[m*64 + c*8 + r] = M_m[r][c]  (column-major)
//
// For each test, one warp sets up registers, calls stmatrix, copies smem to
// global memory, then the host verifies every element.
//
// Exit status: 0 = all PASS, 1 = any FAIL.
// =============================================================================

#include <cuda.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------
// Device helper: generic-pointer → 32-bit shared-memory address
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint32_t smem_u32(const void *ptr) {
    uint32_t a;
    asm volatile("{ .reg .u64 _p; cvta.to.shared.u64 _p, %1; cvt.u32.u64 %0, _p; }"
                 : "=r"(a) : "l"(ptr));
    return a;
}

// ---------------------------------------------------------------------------
// stmatrix wrappers (inline asm, one per register count × .trans)
// ---------------------------------------------------------------------------
__device__ __forceinline__
void stm_x1(uint32_t addr, uint32_t r0) {
    asm volatile("stmatrix.sync.aligned.x1.m8n8.shared.b16 [%0], {%1};"
                 :: "r"(addr), "r"(r0) : "memory");
}
__device__ __forceinline__
void stm_x1t(uint32_t addr, uint32_t r0) {
    asm volatile("stmatrix.sync.aligned.x1.trans.m8n8.shared.b16 [%0], {%1};"
                 :: "r"(addr), "r"(r0) : "memory");
}
__device__ __forceinline__
void stm_x2(uint32_t addr, uint32_t r0, uint32_t r1) {
    asm volatile("stmatrix.sync.aligned.x2.m8n8.shared.b16 [%0], {%1,%2};"
                 :: "r"(addr), "r"(r0), "r"(r1) : "memory");
}
__device__ __forceinline__
void stm_x2t(uint32_t addr, uint32_t r0, uint32_t r1) {
    asm volatile("stmatrix.sync.aligned.x2.trans.m8n8.shared.b16 [%0], {%1,%2};"
                 :: "r"(addr), "r"(r0), "r"(r1) : "memory");
}
__device__ __forceinline__
void stm_x4(uint32_t addr, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
    asm volatile("stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], {%1,%2,%3,%4};"
                 :: "r"(addr), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
}
__device__ __forceinline__
void stm_x4t(uint32_t addr, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
    asm volatile("stmatrix.sync.aligned.x4.trans.m8n8.shared.b16 [%0], {%1,%2,%3,%4};"
                 :: "r"(addr), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
}

// ---------------------------------------------------------------------------
// Data helpers
// ---------------------------------------------------------------------------

// Value model: M_m[r][c] = m*64 + r*8 + c
__host__ __device__ __forceinline__
uint16_t mat_val(int m, int r, int c) { return (uint16_t)(m*64 + r*8 + c); }

// Non-transposed register for matrix m, thread T:
//   lo = M_m[T/4][(T%4)*2],  hi = M_m[T/4][(T%4)*2+1]
__device__ __forceinline__
uint32_t reg_nontrans(int m, int T) {
    uint16_t lo = mat_val(m, T/4, (T%4)*2);
    uint16_t hi = mat_val(m, T/4, (T%4)*2 + 1);
    return (uint32_t)lo | ((uint32_t)hi << 16);
}

// Transposed register for matrix m, thread T:
//   lo = M_m[(T%4)*2][T/4],  hi = M_m[(T%4)*2+1][T/4]
__device__ __forceinline__
uint32_t reg_trans(int m, int T) {
    uint16_t lo = mat_val(m, (T%4)*2,     T/4);
    uint16_t hi = mat_val(m, (T%4)*2 + 1, T/4);
    return (uint32_t)lo | ((uint32_t)hi << 16);
}

// Address register for thread T (for non-transposed and transposed alike):
//   Threads 0-7:  address provider for matrix 0, row/col (T%8).
//   Threads 8-15: address provider for matrix 1.
//   Threads 16-23, 24-31: matrices 2, 3.
//   Formula:  smem_base + (T/8)*128 + (T%8)*16
__device__ __forceinline__
uint32_t addr_reg(uint32_t smem_base, int T) {
    return smem_base + (T/8)*128 + (T%8)*16;
}

// ---------------------------------------------------------------------------
// Kernels
// (each launched with 1 block × 32 threads, dynamic shared memory)
// ---------------------------------------------------------------------------

__global__ void kernel_x1(uint16_t *out) {
    __shared__ __align__(128) uint8_t smem_bytes[128];
    int T = threadIdx.x;
    uint32_t base = smem_u32(smem_bytes);
    uint32_t addr = addr_reg(base, T);
    uint32_t r0   = reg_nontrans(0, T);
    stm_x1(addr, r0);
    __syncthreads();
    for (int i = T; i < 64; i += 32) out[i] = ((uint16_t *)smem_bytes)[i];
}

__global__ void kernel_x1t(uint16_t *out) {
    __shared__ __align__(128) uint8_t smem_bytes[128];
    int T = threadIdx.x;
    uint32_t base = smem_u32(smem_bytes);
    uint32_t addr = addr_reg(base, T);
    uint32_t r0   = reg_trans(0, T);
    stm_x1t(addr, r0);
    __syncthreads();
    for (int i = T; i < 64; i += 32) out[i] = ((uint16_t *)smem_bytes)[i];
}

__global__ void kernel_x2(uint16_t *out) {
    __shared__ __align__(128) uint8_t smem_bytes[256];
    int T = threadIdx.x;
    uint32_t base = smem_u32(smem_bytes);
    uint32_t addr = addr_reg(base, T);
    uint32_t r0   = reg_nontrans(0, T);
    uint32_t r1   = reg_nontrans(1, T);
    stm_x2(addr, r0, r1);
    __syncthreads();
    for (int i = T; i < 128; i += 32) out[i] = ((uint16_t *)smem_bytes)[i];
}

__global__ void kernel_x2t(uint16_t *out) {
    __shared__ __align__(128) uint8_t smem_bytes[256];
    int T = threadIdx.x;
    uint32_t base = smem_u32(smem_bytes);
    uint32_t addr = addr_reg(base, T);
    uint32_t r0   = reg_trans(0, T);
    uint32_t r1   = reg_trans(1, T);
    stm_x2t(addr, r0, r1);
    __syncthreads();
    for (int i = T; i < 128; i += 32) out[i] = ((uint16_t *)smem_bytes)[i];
}

__global__ void kernel_x4(uint16_t *out) {
    __shared__ __align__(128) uint8_t smem_bytes[512];
    int T = threadIdx.x;
    uint32_t base = smem_u32(smem_bytes);
    uint32_t addr = addr_reg(base, T);
    uint32_t r0   = reg_nontrans(0, T);
    uint32_t r1   = reg_nontrans(1, T);
    uint32_t r2   = reg_nontrans(2, T);
    uint32_t r3   = reg_nontrans(3, T);
    stm_x4(addr, r0, r1, r2, r3);
    __syncthreads();
    for (int i = T; i < 256; i += 32) out[i] = ((uint16_t *)smem_bytes)[i];
}

__global__ void kernel_x4t(uint16_t *out) {
    __shared__ __align__(128) uint8_t smem_bytes[512];
    int T = threadIdx.x;
    uint32_t base = smem_u32(smem_bytes);
    uint32_t addr = addr_reg(base, T);
    uint32_t r0   = reg_trans(0, T);
    uint32_t r1   = reg_trans(1, T);
    uint32_t r2   = reg_trans(2, T);
    uint32_t r3   = reg_trans(3, T);
    stm_x4t(addr, r0, r1, r2, r3);
    __syncthreads();
    for (int i = T; i < 256; i += 32) out[i] = ((uint16_t *)smem_bytes)[i];
}

// ---------------------------------------------------------------------------
// Host: expected-value helpers
// ---------------------------------------------------------------------------

// Non-transposed: smem[m*64 + r*8 + c] = M_m[r][c]
static uint16_t expected_nontrans(int num_matrices, int idx) {
    int m = idx / 64;
    int local = idx % 64;
    int r = local / 8;
    int c = local % 8;
    (void)num_matrices;
    return mat_val(m, r, c);
}

// Transposed: smem[m*64 + c*8 + r] = M_m[r][c]
//   → smem[m*64 + i] = M_m[i%8][i/8]
static uint16_t expected_trans(int num_matrices, int idx) {
    int m = idx / 64;
    int local = idx % 64;
    int r = local % 8;   // row index
    int c = local / 8;   // column index
    (void)num_matrices;
    return mat_val(m, r, c);
}

// ---------------------------------------------------------------------------
// Host: run one test variant
// ---------------------------------------------------------------------------
typedef void (*kernel_fn)(uint16_t *);

static bool run_test(const char *name,
                     kernel_fn kernel,
                     int num_matrices,
                     bool is_trans) {
    const int nelems = num_matrices * 64;
    uint16_t *d_out;
    cudaMalloc(&d_out, nelems * sizeof(uint16_t));
    cudaMemset(d_out, 0, nelems * sizeof(uint16_t));

    kernel<<<1, 32>>>(d_out);
    cudaDeviceSynchronize();

    std::vector<uint16_t> h_out(nelems);
    cudaMemcpy(h_out.data(), d_out, nelems * sizeof(uint16_t),
               cudaMemcpyDeviceToHost);
    cudaFree(d_out);

    int errors = 0;
    int first_bad = -1;
    uint16_t first_got = 0, first_exp = 0;
    for (int i = 0; i < nelems; i++) {
        uint16_t exp = is_trans ? expected_trans(num_matrices, i)
                                : expected_nontrans(num_matrices, i);
        if (h_out[i] != exp) {
            if (errors == 0) {
                first_bad = i;
                first_got = h_out[i];
                first_exp = exp;
            }
            errors++;
        }
    }

    bool pass = (errors == 0);
    printf("[%-44s] %s", name, pass ? "PASS" : "FAIL");
    if (!pass)
        printf("  first mismatch: smem[%d] = %u, expected %u  (%d/%d wrong)",
               first_bad, first_got, first_exp, errors, nelems);
    printf("\n");
    return pass;
}

// ===========================================================================
// main
// ===========================================================================
int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("stmatrix_verify  —  %s (SM %d.%d)\n",
           prop.name, prop.major, prop.minor);
    printf("============================================================\n");

    bool all_pass = true;

    all_pass &= run_test("x1  non-transposed  (1 matrix)",  kernel_x1,  1, false);
    all_pass &= run_test("x1  transposed      (1 matrix)",  kernel_x1t, 1, true);
    all_pass &= run_test("x2  non-transposed  (2 matrices)", kernel_x2,  2, false);
    all_pass &= run_test("x2  transposed      (2 matrices)", kernel_x2t, 2, true);
    all_pass &= run_test("x4  non-transposed  (4 matrices)", kernel_x4,  4, false);
    all_pass &= run_test("x4  transposed      (4 matrices)", kernel_x4t, 4, true);

    printf("============================================================\n");
    printf("Overall: %s\n", all_pass ? "PASS" : "FAIL");
    return all_pass ? 0 : 1;
}
