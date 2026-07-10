#include <stdio.h>
#include <stdlib.h>

// ─────────────────────────────────────────────────────────────────────────────
// bar_sync.cu — exercise the CTA-wide barrier (PTX `bar.sync`, i.e. the C-level
// __syncthreads()) and verify from the CPU that it actually synchronized every
// warp in the block.
//
// WHY THIS IS A REAL TEST
// -----------------------
// In GPGPU-Sim each warp (32 lanes) executes in lockstep, but *different warps
// in the same CTA are scheduled independently* — there is no ordering between
// warp 0 and warp 7 unless you force one. bar.sync is that force: it is a gate
// that no warp may pass until EVERY warp in the block has arrived.
//
// A barrier test only proves something if the result is correct *only when the
// barrier works*. So every thread here consumes data produced by a thread in a
// DIFFERENT warp:
//
//   Phase 1 (cross-warp exchange):
//     shared[tid] = value(tid)                 // each warp produces its slice
//     __syncthreads()                          // <-- all warps must finish
//     exchange_out[tid] = shared[BS-1-tid]     // read the *reversed* slot,
//                                              //     always owned by another warp
//
//   Phase 2 (tree reduction — many barriers across warps):
//     shared[tid] = value(tid)
//     for (stride = BS/2; stride > 0; stride >>= 1) {
//         if (tid < stride) shared[tid] += shared[tid + stride];
//         __syncthreads()                      // <-- between every step
//     }
//     block_sum = shared[0]
//
// Without the barriers, the reversed reads and the reduction steps race on
// shared memory produced by other warps, and the CPU checks below fail.
//
// CONTROL EXPERIMENT
// ------------------
// Compile normally  -> barriers ON  -> ALL CHECKS PASSED.
// Compile with -DUSE_BARRIER=0 -> barriers stripped -> checks are expected to
// FAIL (or read garbage), demonstrating that bar.sync is doing real work in the
// simulator rather than being a no-op.
//
//   make KERNEL=bar_sync
//   ./run.sh bar_sync
//   # control:
//   nvcc -DUSE_BARRIER=0 ... bar_sync.cu -o bar_sync_nobar
// ─────────────────────────────────────────────────────────────────────────────

#define WARP_SIZE 32

// Barriers are on by default; -DUSE_BARRIER=0 strips them for the control run.
#ifndef USE_BARRIER
#define USE_BARRIER 1
#endif

// Emit the CTA-wide barrier as EXPLICIT inline PTX so there is no ambiguity
// about which instruction is under test. `bar.sync 0;` is barrier number 0 with
// the implicit "all threads in the CTA" count — exactly what __syncthreads()
// lowers to. In GPGPU-Sim's PTX front-end (src/cuda-sim/ptx.l) the mnemonic
// `bar` lexes to BAR_OP and the `.sync` suffix to SYNC_OPTION, which the decoder
// (cuda-sim.cc set_bar_type) turns into bar_type = SYNC and the shader executes
// via barrier_set_t::warp_reaches_barrier. The "memory" clobber prevents the
// compiler from reordering shared-memory accesses across the barrier, so the
// cross-warp dependency below stays a genuine test.
#if USE_BARRIER
#define BARRIER() asm volatile("bar.sync 0;" ::: "memory")
#else
#define BARRIER() ((void)0)
#endif

// The per-thread "work" result. Kept simple and deterministic so the CPU can
// predict the exact expected value for every slot.
__device__ __forceinline__ int produce(int global_tid) {
    return global_tid + 1;   // value(tid) = tid + 1  (nonzero, so 0 == "unwritten")
}

// Deliberately skew when each warp arrives, so the warps do NOT march in
// lockstep. Later warps (higher warp id) start their shared-memory writes much
// later. This matters because Phase 1 has warp 0 consume the slot produced by
// the *last* warp: if that producer is delayed and there is no barrier, the
// consumer races ahead and reads stale data. With the barrier, everyone waits
// regardless of skew, so the result is still correct. The spin reads the cycle
// counter each iteration, so it cannot be optimized away and it advances real
// simulated time.
__device__ __forceinline__ void skew_by_warp(int warp_id) {
    long long start = clock64();
    long long budget = (long long)warp_id * 4000;
    while (clock64() - start < budget) {
        // busy-wait; the loop condition has a side effect (%clock64 read)
    }
}

__global__ void bar_sync_kernel(int *exchange_out, long long *block_sums,
                                int block_size) {
    extern __shared__ int smem[];

    const int local_tid  = threadIdx.x;
    const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp_id    = local_tid / WARP_SIZE;

    // Force the warps out of lockstep so the cross-warp dependency below is a
    // genuine race when the barrier is absent.
    skew_by_warp(warp_id);

    // ── Phase 1: every warp produces its slice of shared memory ──────────────
    smem[local_tid] = produce(global_tid);

    // Barrier #1: no warp may read another warp's slot until all warps wrote.
    BARRIER();

    // Cross-warp exchange: read the mirrored slot. For a 256-thread block,
    // lane 0 (warp 0) reads slot 255 (warp 7), and vice-versa — every thread
    // reads a slot owned by a *different* warp. This is only correct if the
    // barrier above completed for the whole CTA.
    int mirror = block_size - 1 - local_tid;
    exchange_out[global_tid] = smem[mirror];

    // Barrier #2: make sure all mirror-reads are done before Phase 2 overwrites
    // shared memory.
    BARRIER();

    // ── Phase 2: block-wide tree reduction (a barrier between each step) ──────
    smem[local_tid] = produce(global_tid);
    BARRIER();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (local_tid < stride) {
            smem[local_tid] += smem[local_tid + stride];
        }
        // Each step consumes partial sums written by threads that may live in
        // other warps, so a barrier is required after every step.
        BARRIER();
    }

    if (local_tid == 0) {
        block_sums[blockIdx.x] = (long long)smem[0];
    }
}

int main() {
    const int block_size = 256;                 // 8 warps per block
    const int num_blocks = 4;                   // 4 CTAs -> tests per-CTA scoping
    const int total_threads = num_blocks * block_size;
    const size_t smem_bytes = block_size * sizeof(int);

    // ── Allocations ──────────────────────────────────────────────────────────
    int       *h_exchange = (int *)malloc(total_threads * sizeof(int));
    long long *h_sums     = (long long *)malloc(num_blocks * sizeof(long long));
    int       *d_exchange = NULL;
    long long *d_sums     = NULL;
    cudaMalloc(&d_exchange, total_threads * sizeof(int));
    cudaMalloc(&d_sums,     num_blocks * sizeof(long long));

    // Sentinel-fill so any unwritten slot is obvious (0xFF bytes -> -1).
    cudaMemset(d_exchange, 0xFF, total_threads * sizeof(int));
    cudaMemset(d_sums,     0xFF, num_blocks * sizeof(long long));

    // ── Launch ───────────────────────────────────────────────────────────────
    printf("Launching bar_sync_kernel: %d blocks x %d threads (%d warps/block), "
           "barriers %s\n\n",
           num_blocks, block_size, block_size / WARP_SIZE,
           USE_BARRIER ? "ON" : "OFF (control experiment)");

    bar_sync_kernel<<<num_blocks, block_size, smem_bytes>>>(d_exchange, d_sums,
                                                            block_size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaDeviceSynchronize();

    // ── GPU -> CPU transfer ────────────────────────────────────────────────────
    cudaMemcpy(h_exchange, d_exchange, total_threads * sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sums, d_sums, num_blocks * sizeof(long long),
               cudaMemcpyDeviceToHost);

    // ── Host-side verification ─────────────────────────────────────────────────
    int failures = 0;

    // Check 1: cross-warp exchange. For block b, local thread L must have read
    // the mirrored producer value = produce(global index of (block_size-1-L)).
    printf("Phase 1 — cross-warp shared-memory exchange (reversed index):\n");
    for (int b = 0; b < num_blocks; ++b) {
        int block_ok = 1;
        for (int l = 0; l < block_size; ++l) {
            int global_tid = b * block_size + l;
            int mirror_local = block_size - 1 - l;
            int mirror_global = b * block_size + mirror_local;
            int expected = mirror_global + 1;   // produce(mirror_global)
            int got = h_exchange[global_tid];
            if (got != expected) {
                if (failures < 12) {   // cap the noise
                    printf("  [FAIL] block %d lane %d: got %d, expected %d "
                           "(reversed slot owned by warp %d)\n",
                           b, l, got, expected, mirror_local / WARP_SIZE);
                }
                block_ok = 0;
                failures++;
            }
        }
        if (block_ok) {
            printf("  [PASS] block %d: all %d threads read the correct "
                   "cross-warp mirrored value\n", b, block_size);
        }
    }
    printf("\n");

    // Check 2: block-wide reduction. Sum of produce(tid) over the block.
    // produce(tid) = tid + 1, tids in block b are [b*BS .. b*BS+BS-1].
    printf("Phase 2 — block-wide tree reduction (barrier between each step):\n");
    for (int b = 0; b < num_blocks; ++b) {
        long long expected = 0;
        for (int l = 0; l < block_size; ++l) {
            expected += (b * block_size + l) + 1;   // produce(global_tid)
        }
        long long got = h_sums[b];
        if (got != expected) {
            printf("  [FAIL] block %d: sum got %lld, expected %lld\n",
                   b, got, expected);
            failures++;
        } else {
            printf("  [PASS] block %d: reduced sum = %lld\n", b, got);
        }
    }

    printf("\n==> %s\n", failures == 0 ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED");
#if !USE_BARRIER
    printf("    (barriers were OFF; failures/garbage here confirm bar.sync is "
           "doing real work)\n");
#endif

    // ── Cleanup ────────────────────────────────────────────────────────────────
    cudaFree(d_exchange);
    cudaFree(d_sums);
    free(h_exchange);
    free(h_sums);

    return failures == 0 ? 0 : 1;
}
