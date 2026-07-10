#include <stdio.h>
#include <stdlib.h>

// ─────────────────────────────────────────────────────────────────────────────
// bar_warp_sync.cu — exercise the WARP-level barrier (PTX `bar.warp.sync`, i.e.
// the C-level __syncwarp()) and verify from the CPU that the program is still
// correct.
//
// WHY THIS IS DIFFERENT FROM bar.sync
// -----------------------------------
// bar.sync is a CTA-wide barrier: it synchronizes *all warps* in the block, and
// GPGPU-Sim must implement it because warps are scheduled independently.
//
// bar.warp.sync only synchronizes the 32 lanes *within a single warp*. GPGPU-Sim
// already executes every warp in lockstep (all 32 lanes step together) and
// reconverges divergent lanes with the SIMT/PDOM stack, so the synchronization
// bar.warp.sync would provide is ALREADY guaranteed by the model. The
// instruction therefore has no work to do and is NOP'd in the PTX front-end
// (src/cuda-sim/ptx.l: `bar\.warp[...]* -> NOP_OP`). Its real purpose on silicon
// is Independent Thread Scheduling (Volta+), which the SIMT-stack model does not
// implement.
//
// WHAT THIS TEST CHECKS
// ---------------------
// 1. The simulator PARSES and NOPs `bar.warp.sync` without error (a missing
//    lexer rule would abort PTX parsing).
// 2. A warp-internal data exchange guarded by __syncwarp() produces the correct
//    result. Each lane writes its value to shared memory, __syncwarp(), then
//    reads the MIRRORED lane *within the same warp* (lane l reads lane 31-l).
//    Because the exchange stays inside one warp, lockstep execution makes it
//    correct — which is exactly why the barrier can be a NOP here.
//
// Note: unlike the bar.sync test, there is no cross-warp dependency and no need
// to skew warps — the whole point is that intra-warp ordering is implicit.
// ─────────────────────────────────────────────────────────────────────────────

#define WARP_SIZE 32
#define FULL_MASK 0xffffffffu

// Explicit inline PTX for the warp barrier. `bar.warp.sync <membermask>;` is
// what __syncwarp() lowers to. In GPGPU-Sim this lexes to NOP_OP (see ptx.l),
// so it is dropped; the "memory" clobber still prevents the compiler from
// reordering the shared-memory accesses around it.
__device__ __forceinline__ void warp_sync(unsigned mask) {
    asm volatile("bar.warp.sync %0;" ::"r"(mask) : "memory");
}

__device__ __forceinline__ int produce(int global_tid) {
    return global_tid + 1;   // nonzero, so 0 means "unwritten"
}

__global__ void bar_warp_sync_kernel(int *exchange_out) {
    extern __shared__ int smem[];

    const int local_tid  = threadIdx.x;
    const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane       = local_tid % WARP_SIZE;
    const int warp_base  = local_tid - lane;   // first local tid of this warp

    // Each lane publishes its value into this warp's slice of shared memory.
    smem[local_tid] = produce(global_tid);

    // Warp-level barrier: all 32 lanes have published before anyone reads.
    warp_sync(FULL_MASK);

    // Intra-warp mirrored read: lane l reads lane (31 - l) of the SAME warp.
    int partner = warp_base + (WARP_SIZE - 1 - lane);
    exchange_out[global_tid] = smem[partner];
}

int main() {
    const int block_size = 256;                 // 8 warps per block
    const int num_blocks = 2;
    const int total_threads = num_blocks * block_size;
    const size_t smem_bytes = block_size * sizeof(int);

    int *h_exchange = (int *)malloc(total_threads * sizeof(int));
    int *d_exchange = NULL;
    cudaMalloc(&d_exchange, total_threads * sizeof(int));
    cudaMemset(d_exchange, 0xFF, total_threads * sizeof(int));   // -1 sentinel

    printf("Launching bar_warp_sync_kernel: %d blocks x %d threads "
           "(%d warps/block)\n\n",
           num_blocks, block_size, block_size / WARP_SIZE);

    bar_warp_sync_kernel<<<num_blocks, block_size, smem_bytes>>>(d_exchange);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaDeviceSynchronize();

    cudaMemcpy(h_exchange, d_exchange, total_threads * sizeof(int),
               cudaMemcpyDeviceToHost);

    // Verify: each lane l must have read the value produced by lane (31 - l)
    // of the SAME warp.
    int failures = 0;
    printf("Intra-warp mirrored exchange (lane l reads lane 31-l of its warp):\n");
    for (int b = 0; b < num_blocks; ++b) {
        for (int w = 0; w < block_size / WARP_SIZE; ++w) {
            int warp_ok = 1;
            for (int lane = 0; lane < WARP_SIZE; ++lane) {
                int local_tid   = w * WARP_SIZE + lane;
                int global_tid  = b * block_size + local_tid;
                int partner_lane = WARP_SIZE - 1 - lane;
                int partner_global = b * block_size + w * WARP_SIZE + partner_lane;
                int expected = partner_global + 1;   // produce(partner)
                int got = h_exchange[global_tid];
                if (got != expected) {
                    if (failures < 12) {
                        printf("  [FAIL] block %d warp %d lane %d: got %d, "
                               "expected %d\n", b, w, lane, got, expected);
                    }
                    warp_ok = 0;
                    failures++;
                }
            }
            if (warp_ok) {
                printf("  [PASS] block %d warp %d: all 32 lanes read the "
                       "correct mirrored lane\n", b, w);
            }
        }
    }

    printf("\n==> %s\n", failures == 0 ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED");
    printf("    (a passing run confirms bar.warp.sync was parsed and NOP'd "
           "correctly, and intra-warp lockstep ordering holds)\n");

    cudaFree(d_exchange);
    free(h_exchange);

    return failures == 0 ? 0 : 1;
}
