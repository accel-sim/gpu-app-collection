#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

// Status probe for `fence.mbarrier_init.release.cluster` in the GPGPU-Sim
// functional model.
//
// This Hopper (sm_90+) fence is emitted by CUTLASS example 48 right after
// `mbarrier.init`, to make the barrier initialization visible cluster-wide
// (release, .cluster scope) before any CTA arrives on it. Example 48 launches
// with ClusterShape = 1x1x1 — a SINGLE CTA per cluster. In pure functional
// simulation, shared-memory/mbarrier effects are instantaneous and there is
// only one CTA, so the cluster-scope release fence has nothing to order across
// CTAs: it is functionally a no-op. The simulator already treats it as one
// (lexer maps `fence.mbarrier_init...` -> NOP_OP), and this ubench verifies that
// a single-CTA kernel containing the exact instruction both parses and produces
// correct results under pure functional mode.
//
// The barrier is made semantically load-bearing so a broken fence/barrier would
// be caught: every thread publishes payload[tid], then (after the barrier) reads
// its NEIGHBOR's slot. A correct neighbor read requires every producer to have
// executed past the fence. Expected: out[t] == ((t+1) % N) * 2.
//
// The surrounding mbarrier arrive/try_wait path mirrors how CUTLASS uses the
// barrier the fence is initializing; its result is informational (kept live via
// the `info` buffer) and is not part of the pass criterion, which relies on the
// pure-functional-exact __syncthreads() ordering.

#define N 32   // one warp / one CTA

__global__ void ufence_mbarrier_init_kernel(unsigned int *out, unsigned int *info) {
    __shared__ unsigned long long bar;
    __shared__ unsigned int payload[N];
    const unsigned int t = threadIdx.x;

    // 1. Initialize the mbarrier (thread 0): expect all N threads to arrive.
    if (t == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                     : : "l"(&bar), "r"(N) : "memory");
    }

    // 2. INSTRUCTION UNDER TEST — exact CUTLASS-48 spelling and position:
    //    release, cluster-scope fence ordering the mbarrier init before arrives.
    asm volatile("fence.mbarrier_init.release.cluster;" : : : "memory");

    __syncthreads();

    // 3. Producer: publish this thread's slot.
    payload[t] = t * 2u;

    // 4. Exercise the mbarrier arrive/wait path (as CUTLASS does around the
    //    barrier this fence initializes). Informational only.
    unsigned long long state;
    asm volatile("mbarrier.arrive.b64 %0, [%1], %2;"
                 : "=l"(state) : "l"(&bar), "n"(1) : "memory");

    unsigned int phase = 0u;
    unsigned int done = 0u;
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "mbarrier.try_wait.parity.b64 p, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, p;\n\t"
        "}"
        : "=r"(done) : "l"(&bar), "r"(phase) : "memory");

    // 5. Barrier-ordered consumer: neighbor read requires every producer to have
    //    run past the fence. __syncthreads() is the functional ordering the
    //    pure-functional simulator models exactly.
    __syncthreads();
    out[t]  = payload[(t + 1u) % N];
    info[t] = done;   // keep try_wait live; not part of the pass criterion.
}

int main() {
    // Unbuffered stdout so the verdict survives even if a later stage aborts
    // hard inside the simulator.
    setvbuf(stdout, NULL, _IONBF, 0);

    const int n = N;
    unsigned int *d_out = NULL, *d_info = NULL;
    unsigned int h_out[N], h_info[N];
    cudaMalloc(&d_out,  n * sizeof(unsigned int));
    cudaMalloc(&d_info, n * sizeof(unsigned int));
    cudaMemset(d_out,  0xFF, n * sizeof(unsigned int));
    cudaMemset(d_info, 0xFF, n * sizeof(unsigned int));

    // Single-CTA launch — matches CUTLASS-48's per-cluster CTA count (1).
    ufence_mbarrier_init_kernel<<<1, N>>>(d_out, d_info);
    cudaError_t err = cudaDeviceSynchronize();

    int fails = 0;
    if (err != cudaSuccess) {
        printf("launch/sync error: %s\n", cudaGetErrorString(err));
        fails++;
    } else {
        cudaMemcpy(h_out,  d_out,  n * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_info, d_info, n * sizeof(unsigned int), cudaMemcpyDeviceToHost);

        printf("neighbor-exchange after fence.mbarrier_init.release.cluster:\n");
        for (int i = 0; i < n; ++i) {
            unsigned int expected = (unsigned int)(((i + 1) % N) * 2);
            const char *mark = (h_out[i] == expected) ? "ok" : "MISMATCH";
            if (h_out[i] != expected) fails++;
            printf("  t %2d: out = %3u (expected %3u) [%s]  try_wait=%u\n",
                   i, h_out[i], expected, mark, h_info[i]);
        }
    }

    cudaFree(d_out);
    cudaFree(d_info);

    printf("RESULT: %s\n", fails == 0 ? "PASSED" : "FAILED");
    return fails == 0 ? 0 : 1;
}
