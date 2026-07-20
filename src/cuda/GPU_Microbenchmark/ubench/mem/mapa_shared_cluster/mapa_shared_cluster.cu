#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

// ─────────────────────────────────────────────────────────────────────────────
// mapa_shared_cluster.cu — careful functional gate for `mapa.shared::cluster.u32`
//
// WHAT mapa DOES (PTX ISA 7.8+, sm_90+)
// --------------------------------------
//   mapa.shared::cluster.u32  d, a, b;
//
// Remap shared-window address `a` onto cluster CTA rank `b`. Does NOT move data;
// a later ld/st.shared::cluster uses `d` to touch CTA b's copy of that offset.
//
// WHY THE PREVIOUS UBENCH WAS TOO WEAK
// ------------------------------------
// It wrote each CTA's magic into its *own* slot, then remote-loaded via mapa.
// A stub mapa that returns `a` unchanged (ignoring rank) can still look plausible
// when combined with incidental cluster-address encoding elsewhere. This version
// closes those gaps:
//
//   Gate A — RANK SENSITIVITY (address-only)
//     From the same local address, mapa(., 0) and mapa(., 1) MUST differ.
//     The current GPGPU-Sim stub ignores operand b and copies a → a, so
//     mapped0 == mapped1 and this gate fails hard.
//
//   Gate B — REMOTE STORE, LOCAL LOAD (end-to-end)
//     Each CTA writes its magic into the *peer* slot via
//       st.shared::cluster [mapa(local, peer)]
//     then reads its *own* slot with an ordinary shared load (no mapa).
//     Expected local value == peer's magic. Identity mapa writes own slot
//     instead, so the local load sees own magic and fails.
//
//   Gate C — SELF MAP IS IDENTITY-ISH
//     mapa(local, own_rank) must still address this CTA's slot (local load
//     through that mapped addr returns what we wrote remotely into ourselves,
//     i.e. the peer's magic after Gate B).
//
// Host checks all three after D2H. Poison-fill so a silent no-op is obvious.
//
//   make KERNEL=mapa_shared_cluster
//   ./run.sh mapa_shared_cluster
//   ./run.sh mapa_shared_cluster silicon
// ─────────────────────────────────────────────────────────────────────────────

#define CLUSTER_SIZE 2
#define BLOCK_SIZE   32
#define MAGIC_BASE   0xA000u
#define POISON       0xFFFFFFFFu

__device__ __forceinline__ unsigned cluster_ctarank() {
    unsigned rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
    return rank;
}

// Instruction under test: remap shared-window address `addr` onto CTA `rank`.
__device__ __forceinline__ unsigned mapa_shared_cluster_u32(unsigned addr,
                                                           unsigned rank) {
    unsigned mapped;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(mapped)
                 : "r"(addr), "r"(rank));
    return mapped;
}

__device__ __forceinline__ void st_shared_cluster_u32(unsigned addr,
                                                       unsigned val) {
    asm volatile("st.shared::cluster.u32 [%0], %1;"
                 :
                 : "r"(addr), "r"(val)
                 : "memory");
}

__device__ __forceinline__ unsigned ld_shared_cluster_u32(unsigned addr) {
    unsigned val;
    asm volatile("ld.shared::cluster.u32 %0, [%1];"
                 : "=r"(val)
                 : "r"(addr)
                 : "memory");
    return val;
}

__global__ void __cluster_dims__(CLUSTER_SIZE, 1, 1)
mapa_shared_cluster_kernel(unsigned *d_local_after,      // Gate B result
                            unsigned *d_self_via_mapa,    // Gate C result
                            unsigned *d_mapped_rank0,     // Gate A
                            unsigned *d_mapped_rank1,     // Gate A
                            unsigned *d_local_addr,
                            unsigned *d_mapped_peer) {
    __shared__ unsigned slot;

    const unsigned tid  = threadIdx.x;
    const unsigned rank = cluster_ctarank();
    const unsigned peer = rank ^ 1u;

    // Shared-window address of our local slot (CUTLASS / PTX mapa input form).
    const unsigned local_addr =
        static_cast<unsigned>(__cvta_generic_to_shared(&slot));

    // Start from a known empty slot so Gate B cannot see a self-written magic.
    if (tid == 0) {
        slot = 0u;
    }
    __syncthreads();

    // Cluster barrier so every CTA's zeroing is visible before remote stores.
    // (GPGPU-Sim currently lexes these to NOP_OP; silicon is a real barrier.)
    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;" ::: "memory");

    if (tid == 0) {
        // ── Gate B: write *into the peer CTA* via mapa ─────────────────────
        // (Gate A address checks are recorded after the barrier below.)
        const unsigned mapped_peer = mapa_shared_cluster_u32(local_addr, peer);
        st_shared_cluster_u32(mapped_peer, MAGIC_BASE + rank);
    }

    // Make remote stores visible before anyone reads their local slot.
    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;" ::: "memory");
    __syncthreads();

    if (tid == 0) {
        // Ordinary .shared::cta load of *our* slot — peer should have written
        // MAGIC_BASE + peer into it through Gate B's remapped store.
        const unsigned local_after = slot;

        // ── Gate C: self-map still names our own slot ──────────────────────
        const unsigned mapped_self =
            mapa_shared_cluster_u32(local_addr, rank);
        const unsigned self_via_mapa = ld_shared_cluster_u32(mapped_self);

        // Recompute peer map for the host-visible address dump (same as Gate B).
        const unsigned mapped_peer = mapa_shared_cluster_u32(local_addr, peer);
        const unsigned mapped0     = mapa_shared_cluster_u32(local_addr, 0u);
        const unsigned mapped1     = mapa_shared_cluster_u32(local_addr, 1u);

        d_local_after[rank]   = local_after;
        d_self_via_mapa[rank] = self_via_mapa;
        d_mapped_rank0[rank]  = mapped0;
        d_mapped_rank1[rank]  = mapped1;
        d_local_addr[rank]    = local_addr;
        d_mapped_peer[rank]   = mapped_peer;
    }
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);

    printf("=== mapa.shared::cluster.u32 careful functional ubench ===\n");
    printf("cluster=%d CTAs, block=%d threads, magic_base=0x%X\n\n",
           CLUSTER_SIZE, BLOCK_SIZE, MAGIC_BASE);

    const int n = CLUSTER_SIZE;
    const size_t bytes = n * sizeof(unsigned);

    unsigned *h_local_after   = (unsigned *)malloc(bytes);
    unsigned *h_self_via_mapa = (unsigned *)malloc(bytes);
    unsigned *h_mapped0       = (unsigned *)malloc(bytes);
    unsigned *h_mapped1       = (unsigned *)malloc(bytes);
    unsigned *h_local_addr    = (unsigned *)malloc(bytes);
    unsigned *h_mapped_peer   = (unsigned *)malloc(bytes);

    unsigned *d_local_after = NULL, *d_self_via_mapa = NULL;
    unsigned *d_mapped0 = NULL, *d_mapped1 = NULL;
    unsigned *d_local_addr = NULL, *d_mapped_peer = NULL;

    cudaMalloc(&d_local_after,   bytes);
    cudaMalloc(&d_self_via_mapa, bytes);
    cudaMalloc(&d_mapped0,       bytes);
    cudaMalloc(&d_mapped1,       bytes);
    cudaMalloc(&d_local_addr,    bytes);
    cudaMalloc(&d_mapped_peer,   bytes);

    cudaMemset(d_local_after,   0xFF, bytes);
    cudaMemset(d_self_via_mapa, 0xFF, bytes);
    cudaMemset(d_mapped0,       0xFF, bytes);
    cudaMemset(d_mapped1,       0xFF, bytes);
    cudaMemset(d_local_addr,    0xFF, bytes);
    cudaMemset(d_mapped_peer,   0xFF, bytes);

    mapa_shared_cluster_kernel<<<CLUSTER_SIZE, BLOCK_SIZE>>>(
        d_local_after, d_self_via_mapa, d_mapped0, d_mapped1,
        d_local_addr, d_mapped_peer);
    cudaError_t err = cudaDeviceSynchronize();

    int fails = 0;
    if (err != cudaSuccess) {
        printf("launch/sync error: %s\n", cudaGetErrorString(err));
        fails++;
    } else {
        cudaMemcpy(h_local_after,   d_local_after,   bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_self_via_mapa, d_self_via_mapa, bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_mapped0,       d_mapped0,       bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_mapped1,       d_mapped1,       bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_local_addr,    d_local_addr,    bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_mapped_peer,   d_mapped_peer,   bytes, cudaMemcpyDeviceToHost);

        printf("per-CTA results (after D2H):\n");
        for (int r = 0; r < n; ++r) {
            // After remote stores, my slot holds the peer CTA's magic.
            const unsigned expect_local = MAGIC_BASE + (unsigned)(r ^ 1);
            const unsigned expect_self  = expect_local;  // self-map → same slot

            const int wrote =
                (h_local_after[r] != POISON) && (h_self_via_mapa[r] != POISON) &&
                (h_mapped0[r] != POISON) && (h_mapped1[r] != POISON);

            // Gate A: mapa(.,0) and mapa(.,1) must disagree.
            const int rank_sensitive = (h_mapped0[r] != h_mapped1[r]);

            // Gate A': peer map must differ from self map when peer != self.
            const unsigned mapped_self_expect =
                (r == 0) ? h_mapped0[r] : h_mapped1[r];
            const int peer_addr_differs =
                (h_mapped_peer[r] != mapped_self_expect);

            // Gate B / C value checks.
            const int local_ok = (h_local_after[r] == expect_local);
            const int self_ok  = (h_self_via_mapa[r] == expect_self);

            if (!wrote)            fails++;
            if (!rank_sensitive)   fails++;
            if (!peer_addr_differs) fails++;
            if (!local_ok)         fails++;
            if (!self_ok)          fails++;

            printf("  CTA rank %d:\n", r);
            printf("    local_addr        = 0x%08X\n", h_local_addr[r]);
            printf("    mapa(local, 0)    = 0x%08X\n", h_mapped0[r]);
            printf("    mapa(local, 1)    = 0x%08X  [%s]\n",
                   h_mapped1[r],
                   rank_sensitive ? "rank-sensitive ok" : "RANK IGNORED");
            printf("    mapped_peer_addr  = 0x%08X  [%s]\n",
                   h_mapped_peer[r],
                   peer_addr_differs ? "differs from self ok" : "SAME AS SELF");
            printf("    local_after       = 0x%08X  (expected 0x%08X) [%s]\n",
                   h_local_after[r], expect_local,
                   local_ok ? "ok" : "MISMATCH");
            printf("    self_via_mapa     = 0x%08X  (expected 0x%08X) [%s]\n",
                   h_self_via_mapa[r], expect_self,
                   self_ok ? "ok" : "MISMATCH");
        }
    }

    cudaFree(d_local_after);
    cudaFree(d_self_via_mapa);
    cudaFree(d_mapped0);
    cudaFree(d_mapped1);
    cudaFree(d_local_addr);
    cudaFree(d_mapped_peer);
    free(h_local_after);
    free(h_self_via_mapa);
    free(h_mapped0);
    free(h_mapped1);
    free(h_local_addr);
    free(h_mapped_peer);

    printf("\nRESULT: %s\n", fails == 0 ? "PASSED" : "FAILED");
    return fails == 0 ? 0 : 1;
}
