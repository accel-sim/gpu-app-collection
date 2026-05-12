// =============================================================================
// wgmma_rs_verify.cu  —  RS WGMMA correctness + A-register layout dump
//
// RS WGMMA = "Register-Source" WGMMA: A operand loaded into registers via
// ldmatrix before the wgmma.mma_async instruction. B still uses smem descriptor.
// This is distinct from SS WGMMA (both A and B from smem descriptors).
//
// Purpose: everything the simulator needs to implement RS WGMMA.
//
// PHASE 1 — RS correctness, column test
//   A=1, B[k][n]=n  =>  D[m][n] = K*n
//   Verifies that RS WGMMA produces correct D and that the D-register
//   position formula (d_frag_pos) is the same as for SS WGMMA.
//   Covers: F32F16, F16F16, F32TF32, F32BF16  ×  N=8,16,32,64,96,128,192,256
//   Output: wgmma_rs_col_<dtype>N<N>.txt
//
// PHASE 2 — RS correctness, row test
//   A[m][k]=m, B=1  =>  D[m][n] = K*m
//   Jointly verifies: (a) ldmatrix correctly loaded A from smem, and
//   (b) RS WGMMA computed the right product.
//   If any ldmatrix address or swizzle is wrong, A values are misread
//   and D deviates from K*m.
//   Same shapes as Phase 1.
//   Output: wgmma_rs_row_<dtype>N<N>.txt
//
// PHASE 3 — A register layout dump  (NO WGMMA)
//   A[m][k] = m*K + k  (unique per matrix element)
//   Loads smem_A into rA via ldmatrix, then dumps rA directly to global
//   memory WITHOUT issuing WGMMA. For each (thread T, register element e):
//     decoded_m = round(val) / K
//     decoded_k = round(val) % K
//   This gives the full (T, e) -> (m, k) mapping for the A register fragment —
//   exactly what the simulator needs to implement RS WGMMA functional mode.
//   Covers: F32F16 N=16 (K=16), F32TF32 N=16 (K=8)
//   BF16 uses same element size / ldmatrix as F16; its layout is identical.
//   Output: wgmma_rs_areg_<dtype>N<N>.txt
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
#include <cute/atom/copy_atom.hpp>
#include <cute/arch/copy_sm75.hpp>  // SM75_U32x4_LDSM_N (ldmatrix)

using namespace cute;

static constexpr int WGSIZE = 128;

// ---------------------------------------------------------------------------
// Kernel 1 & 2: RS correctness (col test and row test)
// COL_MODE=true  : A=1, B[k][n]=n    => D[m][n] = K*n
// COL_MODE=false : A[m][k]=m, B=1    => D[m][n] = K*m
// ---------------------------------------------------------------------------
template<class EA, class EB, class EC, class TileShape, bool COL_MODE>
__global__ void kernel_rs_verify(EC* D_out) {
    const int tid = threadIdx.x;
    const int wg  = tid / cutlass::NumThreadsPerWarpGroup;

    static constexpr int M_val = (int)get<0>(TileShape{});
    static constexpr int N_val = (int)get<1>(TileShape{});
    static constexpr int K_val = (int)get<2>(TileShape{});

    auto rs_op = GMMA::rs_op_selector<EA, EB, EC, TileShape,
                                       GMMA::Major::K, GMMA::Major::K>();
    using TiledMma = decltype(make_tiled_mma(rs_op));
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

    if constexpr (COL_MODE) {
        // A=1 everywhere
        for (int i = tid; i < (int)cosize_v<SmemLayoutA>; i += WGSIZE)
            smem_A[i] = EA(1.0f);
        // B[k][n] = n  (sB logical is (N, K, PIPE))
        for (int i = tid; i < N_val * K_val; i += WGSIZE) {
            int n = i % N_val, k = i / N_val;
            sB(n, k, 0) = EB(float(n));
        }
    } else {
        // A[m][k] = m  (row index)
        for (int i = tid; i < M_val * K_val; i += WGSIZE) {
            int m = i / K_val, k = i % K_val;
            sA(m, k, 0) = EA(float(m));
        }
        // B=1 everywhere
        for (int i = tid; i < (int)cosize_v<SmemLayoutB>; i += WGSIZE)
            smem_B[i] = EB(1.0f);
    }
    __syncthreads();

    // --- Load A from smem into registers via ldmatrix ---
    // make_tiled_copy_A derives the right thread-to-smem mapping from the
    // RS TiledMma's A tile layout (get_layoutA_TV).
    // SM75_U32x4_LDSM_N = ldmatrix.sync.aligned.m8n8.x4.shared.b16
    auto tiled_copy_A = make_tiled_copy_A(
        Copy_Atom<SM75_U32x4_LDSM_N, EA>{}, tiled_mma);

    Layout wg_layout = make_layout(Int<1>{}, Int<cutlass::NumThreadsPerWarpGroup>{});
    auto thread_mma = tiled_mma.get_slice(wg_layout(wg));

    // For RS: make_fragment_A returns a REGISTER tensor (not a smem view).
    // Use single-stage partition so tCrA is rank-3 (MMA, MMA_M, MMA_K),
    // matching rank-3 output of partition_S(sA(_,_,0)) after retile_D.
    auto tCsA_mma = thread_mma.partition_A(sA);
    auto tCrA     = thread_mma.make_fragment_A(tCsA_mma(_,_,_,0));  // rank-3, stage 0

    auto smem_thr_copy_A = tiled_copy_A.get_thread_slice(tid);
    auto tCsA_copy = smem_thr_copy_A.partition_S(sA(_,_,0));   // smem source (stage 0)
    auto tCrA_view = smem_thr_copy_A.retile_D(tCrA);            // register dest view
    copy(tiled_copy_A, tCsA_copy, tCrA_view);                   // execute ldmatrix

    // --- Issue RS WGMMA ---
    auto tCsB = thread_mma.partition_B(sB);
    auto tCrB = thread_mma.make_fragment_B(tCsB);
    auto accum = partition_fragment_C(tiled_mma, take<0,2>(TileShape{}));
    clear(accum);

    warpgroup_fence_operand(accum);
    warpgroup_arrive();
    cute::gemm(tiled_mma, tCrA, tCrB(_,_,_,0), accum);
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    warpgroup_fence_operand(accum);

    const int D_ELEMS = (int)size(accum);
    EC* base = D_out + tid * D_ELEMS;
    for (int e = 0; e < D_ELEMS; e++)
        base[e] = accum(e);
}

// ---------------------------------------------------------------------------
// Kernel 3: A register layout dump — no WGMMA issued
// Fill A[m][k] = m*K+k (unique per element), load via ldmatrix,
// dump raw rA register values to global memory.
// Host decodes: m = round(val)/K, k = round(val)%K.
// ---------------------------------------------------------------------------
template<class EA, class EB, class EC, class TileShape>
__global__ void kernel_rs_areg_dump(float* A_out, int* A_reg_size_out) {
    const int tid = threadIdx.x;
    const int wg  = tid / cutlass::NumThreadsPerWarpGroup;

    static constexpr int M_val = (int)get<0>(TileShape{});
    static constexpr int K_val = (int)get<2>(TileShape{});

    auto rs_op = GMMA::rs_op_selector<EA, EB, EC, TileShape,
                                       GMMA::Major::K, GMMA::Major::K>();
    using TiledMma = decltype(make_tiled_mma(rs_op));
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

    // A[m][k] = m*K + k  (unique per matrix element)
    for (int i = tid; i < M_val * K_val; i += WGSIZE) {
        int m = i / K_val, k = i % K_val;
        sA(m, k, 0) = EA(float(m * K_val + k));
    }
    // B unused but smem needs to be valid to avoid issues
    for (int i = tid; i < (int)cosize_v<SmemLayoutB>; i += WGSIZE)
        smem_B[i] = EB(0.0f);
    __syncthreads();

    auto tiled_copy_A = make_tiled_copy_A(
        Copy_Atom<SM75_U32x4_LDSM_N, EA>{}, tiled_mma);

    Layout wg_layout = make_layout(Int<1>{}, Int<cutlass::NumThreadsPerWarpGroup>{});
    auto thread_mma = tiled_mma.get_slice(wg_layout(wg));

    auto tCsA_mma = thread_mma.partition_A(sA);
    auto tCrA     = thread_mma.make_fragment_A(tCsA_mma(_,_,_,0));  // rank-3, stage 0

    auto smem_thr_copy_A = tiled_copy_A.get_thread_slice(tid);
    auto tCsA_copy = smem_thr_copy_A.partition_S(sA(_,_,0));
    auto tCrA_view = smem_thr_copy_A.retile_D(tCrA);
    copy(tiled_copy_A, tCsA_copy, tCrA_view);  // ldmatrix → registers

    // Dump: no WGMMA, just write register values to global memory
    const int A_regs = (int)size(tCrA);
    if (tid == 0) *A_reg_size_out = A_regs;
    for (int e = 0; e < A_regs; e++)
        A_out[tid * A_regs + e] = (float)tCrA(e);
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------

// D-register position: (thread T, accumulator element e) -> (row, col)
// Valid for F32/S32 accumulator at all N. F16 accumulator uses same formula.
static void d_frag_pos(int T, int e, int /*N*/, int* row, int* col) {
    int warp = T / 32, lane = T % 32;
    int g = e / 4, k = e % 4;
    *row = (lane / 4) + (k / 2) * 8 + warp * 16;
    *col = (lane % 4) * 2 + (k % 2) + g * 8;
}

template<class EA, class EB, class EC, class TileShape, bool COL_MODE>
static void run_verify(const char* dtype_tag, const char* name, int N, int K) {
    const int D_ELEMS = N / 2;
    const int total   = WGSIZE * D_ELEMS;
    std::vector<EC> h_D(total, EC(0));
    EC* d_D;
    cudaMalloc(&d_D, total * sizeof(EC));
    cudaMemset(d_D, 0, total * sizeof(EC));

    kernel_rs_verify<EA, EB, EC, TileShape, COL_MODE><<<1, WGSIZE>>>(d_D);
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        printf("  RS-%-3s %-50s KERNEL ERROR: %s\n",
               COL_MODE ? "Col" : "Row", name, cudaGetErrorString(err));
        cudaFree(d_D); return;
    }
    cudaDeviceSynchronize();
    cudaMemcpy(h_D.data(), d_D, total * sizeof(EC), cudaMemcpyDeviceToHost);
    cudaFree(d_D);

    char fname[128];
    snprintf(fname, sizeof(fname), "wgmma_rs_%s_%sN%d.txt",
             COL_MODE ? "col" : "row", dtype_tag, N);
    FILE* f = fopen(fname, "w");
    if (f) {
        if (COL_MODE)
            fprintf(f, "# %s  RS: A=1  B[k][n]=n  expected D[m][n]=K*n (K=%d)\n"
                       "# thread elem row col expected actual match\n", name, K);
        else
            fprintf(f, "# %s  RS: A[m][k]=m  B=1  expected D[m][n]=K*m (K=%d)\n"
                       "# thread elem row col expected actual match\n", name, K);
    }

    int errors = 0, checked = 0;
    for (int T = 0; T < WGSIZE; T++) {
        for (int e = 0; e < D_ELEMS; e++) {
            int row, col;
            d_frag_pos(T, e, N, &row, &col);
            if (row >= 64 || col >= N) continue;
            float expected = COL_MODE ? (float)(K * col) : (float)(K * row);
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

    printf("  RS-%-3s %-50s errors=%d/%d  -> %s\n",
           COL_MODE ? "Col" : "Row", name, errors, checked,
           errors == 0 ? "PASS" : "FAIL");
}

// A register layout dump: loads smem_A via ldmatrix, writes raw values.
// Produces (T, e) -> (m, k) mapping for the simulator to reference.
template<class EA, class EB, class EC, class TileShape>
static void run_areg_dump(const char* dtype_tag, const char* name, int K) {
    // A_reg_size per thread is K*M/(128) / (sizeof_EA_in_regs) — query from kernel
    // Allocate conservatively: max is K*64/128 = K/2 per thread (for 2-byte types)
    // For F16 K=16: 8 elements per thread. For TF32 K=8: 4 per thread.
    const int MAX_REGS = 16;
    const int total = WGSIZE * MAX_REGS;
    std::vector<float> h_A(total, 0.0f);
    float* d_A;
    int*   d_cnt;
    cudaMalloc(&d_A,   total * sizeof(float));
    cudaMalloc(&d_cnt, sizeof(int));
    cudaMemset(d_A,   0, total * sizeof(float));
    cudaMemset(d_cnt, 0, sizeof(int));

    kernel_rs_areg_dump<EA, EB, EC, TileShape><<<1, WGSIZE>>>(d_A, d_cnt);
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        printf("  RS-AregDump %-46s KERNEL ERROR: %s\n", name, cudaGetErrorString(err));
        cudaFree(d_A); cudaFree(d_cnt); return;
    }
    cudaDeviceSynchronize();
    cudaMemcpy(h_A.data(), d_A, total * sizeof(float), cudaMemcpyDeviceToHost);
    int h_cnt = 0;
    cudaMemcpy(&h_cnt, d_cnt, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_cnt);

    if (h_cnt == 0 || h_cnt > MAX_REGS) {
        printf("  RS-AregDump %-46s unexpected A_reg_size=%d\n", name, h_cnt);
        return;
    }

    char fname[128];
    snprintf(fname, sizeof(fname), "wgmma_rs_areg_%sN%d.txt", dtype_tag,
             (int)get<1>(TileShape{}));
    FILE* f = fopen(fname, "w");
    if (f) {
        fprintf(f, "# %s  A[m][k]=m*K+k loaded via ldmatrix -> rA registers\n", name);
        fprintf(f, "# A_reg_size per thread = %d\n", h_cnt);
        fprintf(f, "# K=%d  (so decoded_m = round(val)/%d, decoded_k = round(val)%%%d)\n",
                K, K, K);
        fprintf(f, "# thread elem raw_value decoded_m decoded_k\n");
    }

    int decode_errors = 0;
    for (int T = 0; T < WGSIZE; T++) {
        for (int e = 0; e < h_cnt; e++) {
            float val = h_A[T * MAX_REGS + e];
            int   iv  = (int)roundf(val);
            int   m   = iv / K;
            int   k   = iv % K;
            bool  ok  = (m >= 0 && m < 64 && k >= 0 && k < K);
            if (!ok) decode_errors++;
            if (f)
                fprintf(f, "%3d %2d %8.3f %2d %2d%s\n",
                        T, e, val, ok ? m : -1, ok ? k : -1,
                        ok ? "" : "  [DECODE_ERROR]");
        }
    }
    if (f) {
        fprintf(f, "# decode_errors=%d / %d entries\n",
                decode_errors, WGSIZE * h_cnt);
        fclose(f);
    }

    printf("  RS-AregDump %-46s A_reg_size=%d  decode_errors=%d/%d  -> %s\n",
           name, h_cnt, decode_errors, WGSIZE * h_cnt,
           decode_errors == 0 ? "OK" : "DECODE_ERROR");
}

// ---------------------------------------------------------------------------
// Convenience macros
// ---------------------------------------------------------------------------
#define SHAPE(M,N,K) decltype(make_shape(Int<M>{}, Int<N>{}, Int<K>{}))

#define RUN_COL(EA, EB, EC, M, N, K, TAG, LABEL) \
    run_verify<EA, EB, EC, SHAPE(M,N,K), true>(TAG, LABEL, N, K)

#define RUN_ROW(EA, EB, EC, M, N, K, TAG, LABEL) \
    run_verify<EA, EB, EC, SHAPE(M,N,K), false>(TAG, LABEL, N, K)

#define RUN_DUMP(EA, EB, EC, M, N, K, TAG, LABEL) \
    run_areg_dump<EA, EB, EC, SHAPE(M,N,K)>(TAG, LABEL, K)

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("wgmma_rs_verify  -  %s (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);

    // -----------------------------------------------------------------------
    // Phase 1: RS correctness — column test  (A=1, B[k][n]=n => D=K*n)
    // Verifies RS WGMMA computes correct D and d_frag_pos formula matches SS.
    // -----------------------------------------------------------------------
    printf("=== Phase 1: RS column test  (A=1, B[k][n]=n => D[m][n]=K*n) ===\n");
    printf("  %-55s  errors/checked\n", "shape");

    printf("\n  -- F32 accumulator, F16 inputs RS (K=16) --\n");
    RUN_COL(cutlass::half_t, cutlass::half_t, float, 64,   8, 16, "F32F16", "F32F16_m64n8k16");
    RUN_COL(cutlass::half_t, cutlass::half_t, float, 64,  16, 16, "F32F16", "F32F16_m64n16k16");
    RUN_COL(cutlass::half_t, cutlass::half_t, float, 64,  32, 16, "F32F16", "F32F16_m64n32k16");
    RUN_COL(cutlass::half_t, cutlass::half_t, float, 64,  64, 16, "F32F16", "F32F16_m64n64k16");
    RUN_COL(cutlass::half_t, cutlass::half_t, float, 64,  96, 16, "F32F16", "F32F16_m64n96k16");
    RUN_COL(cutlass::half_t, cutlass::half_t, float, 64, 128, 16, "F32F16", "F32F16_m64n128k16");
    RUN_COL(cutlass::half_t, cutlass::half_t, float, 64, 192, 16, "F32F16", "F32F16_m64n192k16");
    RUN_COL(cutlass::half_t, cutlass::half_t, float, 64, 256, 16, "F32F16", "F32F16_m64n256k16");

    printf("\n  -- F16 accumulator, F16 inputs RS (K=16) --\n");
    RUN_COL(cutlass::half_t, cutlass::half_t, cutlass::half_t, 64,   8, 16, "F16F16", "F16F16_m64n8k16");
    RUN_COL(cutlass::half_t, cutlass::half_t, cutlass::half_t, 64,  16, 16, "F16F16", "F16F16_m64n16k16");
    RUN_COL(cutlass::half_t, cutlass::half_t, cutlass::half_t, 64,  32, 16, "F16F16", "F16F16_m64n32k16");
    RUN_COL(cutlass::half_t, cutlass::half_t, cutlass::half_t, 64,  64, 16, "F16F16", "F16F16_m64n64k16");
    RUN_COL(cutlass::half_t, cutlass::half_t, cutlass::half_t, 64,  96, 16, "F16F16", "F16F16_m64n96k16");
    RUN_COL(cutlass::half_t, cutlass::half_t, cutlass::half_t, 64, 128, 16, "F16F16", "F16F16_m64n128k16");
    RUN_COL(cutlass::half_t, cutlass::half_t, cutlass::half_t, 64, 192, 16, "F16F16", "F16F16_m64n192k16");
    RUN_COL(cutlass::half_t, cutlass::half_t, cutlass::half_t, 64, 256, 16, "F16F16", "F16F16_m64n256k16");

    printf("\n  -- F32 accumulator, TF32 inputs RS (K=8) --\n");
    RUN_COL(cutlass::tfloat32_t, cutlass::tfloat32_t, float, 64,   8,  8, "F32TF32", "F32TF32_m64n8k8");
    RUN_COL(cutlass::tfloat32_t, cutlass::tfloat32_t, float, 64,  16,  8, "F32TF32", "F32TF32_m64n16k8");
    RUN_COL(cutlass::tfloat32_t, cutlass::tfloat32_t, float, 64,  32,  8, "F32TF32", "F32TF32_m64n32k8");
    RUN_COL(cutlass::tfloat32_t, cutlass::tfloat32_t, float, 64,  64,  8, "F32TF32", "F32TF32_m64n64k8");
    RUN_COL(cutlass::tfloat32_t, cutlass::tfloat32_t, float, 64, 128,  8, "F32TF32", "F32TF32_m64n128k8");
    RUN_COL(cutlass::tfloat32_t, cutlass::tfloat32_t, float, 64, 256,  8, "F32TF32", "F32TF32_m64n256k8");

    printf("\n  -- F32 accumulator, BF16 inputs RS (K=16) --\n");
    RUN_COL(cutlass::bfloat16_t, cutlass::bfloat16_t, float, 64,   8, 16, "F32BF16", "F32BF16_m64n8k16");
    RUN_COL(cutlass::bfloat16_t, cutlass::bfloat16_t, float, 64,  16, 16, "F32BF16", "F32BF16_m64n16k16");
    RUN_COL(cutlass::bfloat16_t, cutlass::bfloat16_t, float, 64,  32, 16, "F32BF16", "F32BF16_m64n32k16");
    RUN_COL(cutlass::bfloat16_t, cutlass::bfloat16_t, float, 64,  64, 16, "F32BF16", "F32BF16_m64n64k16");
    RUN_COL(cutlass::bfloat16_t, cutlass::bfloat16_t, float, 64, 128, 16, "F32BF16", "F32BF16_m64n128k16");
    RUN_COL(cutlass::bfloat16_t, cutlass::bfloat16_t, float, 64, 256, 16, "F32BF16", "F32BF16_m64n256k16");

    // -----------------------------------------------------------------------
    // Phase 2: RS correctness — row test  (A[m][k]=m, B=1 => D=K*m)
    // Jointly verifies ldmatrix loaded A correctly AND RS WGMMA computed D.
    // -----------------------------------------------------------------------
    printf("\n=== Phase 2: RS row test  (A[m][k]=m, B=1 => D[m][n]=K*m) ===\n");
    printf("  %-55s  errors/checked\n", "shape");

    printf("\n  -- F32 accumulator, F16 inputs RS (K=16) --\n");
    RUN_ROW(cutlass::half_t, cutlass::half_t, float, 64,   8, 16, "F32F16", "F32F16_m64n8k16");
    RUN_ROW(cutlass::half_t, cutlass::half_t, float, 64,  16, 16, "F32F16", "F32F16_m64n16k16");
    RUN_ROW(cutlass::half_t, cutlass::half_t, float, 64,  32, 16, "F32F16", "F32F16_m64n32k16");
    RUN_ROW(cutlass::half_t, cutlass::half_t, float, 64,  64, 16, "F32F16", "F32F16_m64n64k16");
    RUN_ROW(cutlass::half_t, cutlass::half_t, float, 64,  96, 16, "F32F16", "F32F16_m64n96k16");
    RUN_ROW(cutlass::half_t, cutlass::half_t, float, 64, 128, 16, "F32F16", "F32F16_m64n128k16");
    RUN_ROW(cutlass::half_t, cutlass::half_t, float, 64, 192, 16, "F32F16", "F32F16_m64n192k16");
    RUN_ROW(cutlass::half_t, cutlass::half_t, float, 64, 256, 16, "F32F16", "F32F16_m64n256k16");

    printf("\n  -- F16 accumulator, F16 inputs RS (K=16) --\n");
    RUN_ROW(cutlass::half_t, cutlass::half_t, cutlass::half_t, 64,  16, 16, "F16F16", "F16F16_m64n16k16");
    RUN_ROW(cutlass::half_t, cutlass::half_t, cutlass::half_t, 64,  64, 16, "F16F16", "F16F16_m64n64k16");
    RUN_ROW(cutlass::half_t, cutlass::half_t, cutlass::half_t, 64, 128, 16, "F16F16", "F16F16_m64n128k16");
    RUN_ROW(cutlass::half_t, cutlass::half_t, cutlass::half_t, 64, 256, 16, "F16F16", "F16F16_m64n256k16");

    printf("\n  -- F32 accumulator, TF32 inputs RS (K=8) --\n");
    RUN_ROW(cutlass::tfloat32_t, cutlass::tfloat32_t, float, 64,  16,  8, "F32TF32", "F32TF32_m64n16k8");
    RUN_ROW(cutlass::tfloat32_t, cutlass::tfloat32_t, float, 64,  64,  8, "F32TF32", "F32TF32_m64n64k8");
    RUN_ROW(cutlass::tfloat32_t, cutlass::tfloat32_t, float, 64, 128,  8, "F32TF32", "F32TF32_m64n128k8");

    printf("\n  -- F32 accumulator, BF16 inputs RS (K=16) --\n");
    RUN_ROW(cutlass::bfloat16_t, cutlass::bfloat16_t, float, 64,  16, 16, "F32BF16", "F32BF16_m64n16k16");
    RUN_ROW(cutlass::bfloat16_t, cutlass::bfloat16_t, float, 64,  64, 16, "F32BF16", "F32BF16_m64n64k16");
    RUN_ROW(cutlass::bfloat16_t, cutlass::bfloat16_t, float, 64, 256, 16, "F32BF16", "F32BF16_m64n256k16");

    // -----------------------------------------------------------------------
    // Phase 3: A register layout dump — the key data for simulator implementation
    // Shows which A[m][k] each (thread, register_element) holds after ldmatrix.
    // BF16 uses same element size and ldmatrix layout as F16 (K=16, 2-byte);
    // its mapping is implied by the F32F16 dump.
    // -----------------------------------------------------------------------
    printf("\n=== Phase 3: A register layout dump  (A[m][k]=m*K+k, no WGMMA) ===\n");
    printf("  Decodes (thread T, register element e) -> (row m, col k) in A tile.\n");
    printf("  This is the reference for simulator RS WGMMA functional implementation.\n\n");

    // F16 K=16: primary case, covers F16 and BF16 (same 2-byte ldmatrix)
    RUN_DUMP(cutlass::half_t, cutlass::half_t, float, 64,  16, 16, "F32F16", "F32F16_m64n16k16");
    RUN_DUMP(cutlass::half_t, cutlass::half_t, float, 64,  64, 16, "F32F16", "F32F16_m64n64k16");

    // TF32 K=8: 4-byte elements, different ldmatrix pattern
    RUN_DUMP(cutlass::tfloat32_t, cutlass::tfloat32_t, float, 64, 16, 8, "F32TF32", "F32TF32_m64n16k8");

    printf("\n  A register layout dumps -> wgmma_rs_areg_<dtype>N<N>.txt\n");
    printf("  Format: thread  elem  raw_value  decoded_m  decoded_k\n");

    printf("\nDone.\n");
    return 0;
}
