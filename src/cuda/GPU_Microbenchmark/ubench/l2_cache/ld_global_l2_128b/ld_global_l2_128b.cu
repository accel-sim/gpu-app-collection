#include <stdio.h>
#include <stdlib.h>

// ─────────────────────────────────────────────────────────────────────────────
// ld_global_l2_128b.cu — exercise `ld.global.L2::128B.u32` and verify from the
// CPU that the *load* still works functionally while the L2::128B qualifier is
// treated as a no-op performance hint.
//
// CONTEXT (CUTLASS / GPGPU-Sim)
// -----------------------------
// CUTLASS emits loads of the form:
//
//     ld.global.L2::128B.u32  %r, [%addr];
//
// `.L2::128B` is a PTX `.level::prefetch_size` hint (PTX ISA 7.4+): it asks the
// hardware to prefetch 128B into L2 alongside the load. It does NOT change the
// data returned by the load — only (maybe) timing on silicon.
//
// In GPGPU-Sim the qualifier is already recognized:
//   ptx.l  :  `.L2::128B`  ->  L2_OPTION
//   ptx_ir :  L2_OPTION falls into the "recognized, no special action" path
// so the hint is effectively NOP'd for functional mode, while `ld.global.u32`
// still performs a real global memory read.
//
// WHAT THIS TEST CHECKS
// ---------------------
// 1. The simulator PARSES `ld.global.L2::128B.u32` without error (a missing
//    lexer/option rule would abort PTX parsing).
// 2. The underlying global load still happens: each thread reads a known value
//    from a host-initialized global buffer via the hinted load, transforms it,
//    and writes to an output buffer. The host checks every slot — proving a
//    real memory transaction occurred, not a silent no-op of the whole insn.
// 3. A plain `ld.global.u32` control path on the same input produces identical
//    results, confirming L2::128B does not affect functional correctness.
//
//   make KERNEL=ld_global_l2_128b
//   ./run.sh ld_global_l2_128b
// ─────────────────────────────────────────────────────────────────────────────

// The instruction under test: load with the L2::128B prefetch hint.
__device__ __forceinline__ unsigned ld_global_l2_128b(const unsigned *ptr) {
    unsigned val;
    asm volatile("ld.global.L2::128B.u32 %0, [%1];"
                 : "=r"(val)
                 : "l"(ptr));
    return val;
}

// Control: same load without the prefetch hint.
__device__ __forceinline__ unsigned ld_global_plain(const unsigned *ptr) {
    unsigned val;
    asm volatile("ld.global.u32 %0, [%1];"
                 : "=r"(val)
                 : "l"(ptr));
    return val;
}

// Each thread:
//   1. loads in[tid] via ld.global.L2::128B.u32  -> out_hinted[tid]
//   2. loads in[tid] via ld.global.u32            -> out_plain[tid]
//   3. stores (loaded + 1) so the host sees a write that depends on the load
__global__ void ld_global_l2_128b_kernel(const unsigned *in,
                                         unsigned *out_hinted,
                                         unsigned *out_plain) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned hinted = ld_global_l2_128b(&in[tid]);
    unsigned plain  = ld_global_plain(&in[tid]);

    // Depend on the loaded values so neither load can be DCE'd, and so the
    // host-visible stores are genuine memory transactions seeded by the loads.
    out_hinted[tid] = hinted + 1u;
    out_plain[tid]  = plain  + 1u;
}

int main() {
    const int block_size    = 256;
    const int num_blocks    = 4;
    const int total_threads = num_blocks * block_size;

    unsigned *h_in      = (unsigned *)malloc(total_threads * sizeof(unsigned));
    unsigned *h_hinted  = (unsigned *)malloc(total_threads * sizeof(unsigned));
    unsigned *h_plain   = (unsigned *)malloc(total_threads * sizeof(unsigned));
    unsigned *d_in      = NULL;
    unsigned *d_hinted  = NULL;
    unsigned *d_plain   = NULL;

    for (int i = 0; i < total_threads; ++i) {
        h_in[i] = (unsigned)(i * 3 + 7);   // matches produce(tid)
    }

    cudaMalloc(&d_in,     total_threads * sizeof(unsigned));
    cudaMalloc(&d_hinted, total_threads * sizeof(unsigned));
    cudaMalloc(&d_plain,  total_threads * sizeof(unsigned));

    // Sentinel-fill outputs so unwritten slots are obvious (0xFF -> all bits 1).
    cudaMemset(d_hinted, 0xFF, total_threads * sizeof(unsigned));
    cudaMemset(d_plain,  0xFF, total_threads * sizeof(unsigned));

    cudaMemcpy(d_in, h_in, total_threads * sizeof(unsigned),
               cudaMemcpyHostToDevice);

    printf("Launching ld_global_l2_128b_kernel: %d blocks x %d threads\n"
           "  path A: ld.global.L2::128B.u32  (hint under test)\n"
           "  path B: ld.global.u32           (plain control)\n\n",
           num_blocks, block_size);

    ld_global_l2_128b_kernel<<<num_blocks, block_size>>>(d_in, d_hinted, d_plain);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaDeviceSynchronize();

    cudaMemcpy(h_hinted, d_hinted, total_threads * sizeof(unsigned),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_plain,  d_plain,  total_threads * sizeof(unsigned),
               cudaMemcpyDeviceToHost);

    int failures = 0;

    // Check 1: hinted load produced the correct functional value for every tid.
    printf("Check 1 — ld.global.L2::128B.u32 functional correctness:\n");
    {
        int ok = 1;
        for (int i = 0; i < total_threads; ++i) {
            unsigned expected = h_in[i] + 1u;
            unsigned got = h_hinted[i];
            if (got != expected) {
                if (failures < 12) {
                    printf("  [FAIL] tid %d: got %u, expected %u "
                           "(input was %u)\n",
                           i, got, expected, h_in[i]);
                }
                ok = 0;
                failures++;
            }
        }
        if (ok) {
            printf("  [PASS] all %d threads: hinted load returned input+1\n",
                   total_threads);
        }
    }
    printf("\n");

    // Check 2: plain load matches the hinted path exactly.
    printf("Check 2 — L2::128B hint is functionally equivalent to plain ld:\n");
    {
        int ok = 1;
        for (int i = 0; i < total_threads; ++i) {
            if (h_hinted[i] != h_plain[i]) {
                if (failures < 12) {
                    printf("  [FAIL] tid %d: hinted=%u plain=%u\n",
                           i, h_hinted[i], h_plain[i]);
                }
                ok = 0;
                failures++;
            }
        }
        if (ok) {
            printf("  [PASS] all %d threads: hinted == plain "
                   "(prefetch hint did not change results)\n",
                   total_threads);
        }
    }

    printf("\n==> %s\n",
           failures == 0 ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED");
    printf("    (a passing run confirms L2::128B was parsed/ignored as a\n"
           "     prefetch hint, while ld.global still performed a real load\n"
           "     that the host can observe through the output buffer)\n");

    cudaFree(d_in);
    cudaFree(d_hinted);
    cudaFree(d_plain);
    free(h_in);
    free(h_hinted);
    free(h_plain);

    return failures == 0 ? 0 : 1;
}
