#include <stdio.h>
#include <stdlib.h>

// Verify elect.sync by forcing a *specific* leader lane in each warp. The
// instruction has TWO results we want to check:
//
//   1. The predicate p  -> true only for the elected leader lane.
//   2. The register  rx -> the laneid of the elected leader, returned to
//                          EVERY participating thread in the membermask
//                          (not just the leader).
//
// We test both with one slot per thread in two contiguous buffers:
//   - d_leader[tid] : leader writes 500, every other (participating) thread 0.
//   - d_rx[tid]     : each participating thread writes the laneid it read from
//                     rx; non-participating threads write the sentinel -1.
//
// elect.sync elects the lowest active lane in the membermask. So to make lane w
// the leader of warp w, we pass a membermask with the low w bits cleared
// (0xFFFFFFFF << w); the lowest remaining active lane is then exactly w:
//   warp 0 -> lane 0 leader, warp 1 -> lane 1 leader, ... warp w -> lane w.
//
// After the kernel runs, the host checks each warp's 32-slot region:
//   - d_leader: a single 500 at lane == warp id, 0 at the other participants.
//   - d_rx    : every participating lane (l >= warp id) reads rx == warp id.

#define WARP_SIZE 32

__global__ void elect_sync_verify_kernel(int *d_leader, int *d_rx) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int warp_id = tid / WARP_SIZE;
    unsigned int lane = threadIdx.x % WARP_SIZE;   // lane index within the warp

    // We want lane == warp_id to be the elected leader of warp w.
    //
    // elect.sync requires that EVERY thread executing it is named in the
    // membermask, and that all named threads are converged/active. The mask
    // 0xFFFFFFFF << warp_id names lanes [warp_id .. 31], so we must guard the
    // instruction so that only those lanes actually execute it. Inside the
    // branch the converged active set is exactly the masked set, which is the
    // well-defined contract for elect.sync. It then elects the lowest active
    // lane — which is exactly lane == warp_id.
    unsigned int member_mask = 0xFFFFFFFFu << warp_id;

    int value = 0;
    int elected_lane = -1;   // sentinel for threads that don't participate
    if (lane >= warp_id) {
        unsigned int is_leader;
        unsigned int rx;                 // laneid of the elected leader
        asm volatile(
            "{\n\t"
            "  .reg .pred p;\n\t"             // predicate: true only for the leader
            "  elect.sync %0|p, %2;\n\t"      // rx = elected laneid, p = is-leader
            "  selp.u32 %1, 1, 0, p;\n\t"     // is_leader = p ? 1 : 0
            "}"
            : "=r"(rx),                       // %0
              "=r"(is_leader)                 // %1
            : "r"(member_mask)                // %2
        );
        elected_lane = (int)rx;
        if (is_leader) value = 500;
    }

    // Leader writes 500, everyone else writes 0 — contiguous, one slot per thread.
    d_leader[tid] = value;
    // Each participating thread records the laneid it got back from rx.
    d_rx[tid] = elected_lane;
}

int main() {
    const int total_warps = 5;                 // need warp 4 to exist
    const int total_threads = total_warps * WARP_SIZE;

    // ── Allocations ──────────────────────────────────────────────────────────
    int *h_leader = (int *)malloc(total_threads * sizeof(int));
    int *h_rx     = (int *)malloc(total_threads * sizeof(int));
    int *d_leader = NULL;
    int *d_rx     = NULL;
    cudaMalloc(&d_leader, total_threads * sizeof(int));
    cudaMalloc(&d_rx,     total_threads * sizeof(int));

    // Sentinel-fill so any unwritten slot is obvious (0xFF bytes -> -1).
    cudaMemset(d_leader, 0xFF, total_threads * sizeof(int));
    cudaMemset(d_rx,     0xFF, total_threads * sizeof(int));

    // ── Launch (single block, one warp per intended leader lane) ─────────────
    elect_sync_verify_kernel<<<1, total_threads>>>(d_leader, d_rx);
    cudaDeviceSynchronize();

    // ── Copy results back ────────────────────────────────────────────────────
    cudaMemcpy(h_leader, d_leader, total_threads * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_rx,     d_rx,     total_threads * sizeof(int), cudaMemcpyDeviceToHost);

    // ── Dump the raw buffers ─────────────────────────────────────────────────
    printf("Leader buffer (one slot per thread, 500 == elected leader):\n");
    for (int w = 0; w < total_warps; ++w) {
        printf("warp %d:", w);
        for (int l = 0; l < WARP_SIZE; ++l) {
            printf(" %d", h_leader[w * WARP_SIZE + l]);
        }
        printf("\n");
    }
    printf("\n");

    printf("rx buffer (one slot per thread, elected laneid; -1 == did not participate):\n");
    for (int w = 0; w < total_warps; ++w) {
        printf("warp %d:", w);
        for (int l = 0; l < WARP_SIZE; ++l) {
            printf(" %d", h_rx[w * WARP_SIZE + l]);
        }
        printf("\n");
    }
    printf("\n");

    // ── Host-side verification ───────────────────────────────────────────────
    // Expected leader lane in warp w is lane w. Every participating lane
    // (l >= w) must read rx == w; non-participating lanes (l < w) stay -1.
    int failures = 0;
    for (int w = 0; w < total_warps; ++w) {
        int expected_leader = w;
        int warp_ok = 1;
        for (int l = 0; l < WARP_SIZE; ++l) {
            int idx = w * WARP_SIZE + l;

            // Check 1: predicate result via the leader marker.
            int v = h_leader[idx];
            int expected_v = (l == expected_leader) ? 500 : 0;
            if (v != expected_v) {
                printf("  [FAIL] warp %d lane %d: leader marker got %d, expected %d\n",
                       w, l, v, expected_v);
                warp_ok = 0;
                failures++;
            }

            // Check 2: rx register holds the elected laneid for participants.
            int rx = h_rx[idx];
            int expected_rx = (l >= expected_leader) ? expected_leader : -1;
            if (rx != expected_rx) {
                printf("  [FAIL] warp %d lane %d: rx got %d, expected %d\n",
                       w, l, rx, expected_rx);
                warp_ok = 0;
                failures++;
            }
        }
        if (warp_ok) {
            printf("  [PASS] warp %d: leader is lane %d (value 500), "
                   "all participants read rx == %d\n",
                   w, expected_leader, expected_leader);
        }
    }

    printf("\n==> %s\n", failures == 0 ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED");

    // ── Cleanup ──────────────────────────────────────────────────────────────
    cudaFree(d_leader);
    cudaFree(d_rx);
    free(h_leader);
    free(h_rx);

    return failures == 0 ? 0 : 1;
}
