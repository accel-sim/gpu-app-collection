#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

// ─────────────────────────────────────────────────────────────────────────────
// mapa_shared_cluster_extended.cu — wide functional gate for
// `mapa.shared::cluster.u32`, extending mapa_shared_cluster.cu from a 2-CTA
// cluster to a full 8-CTA cluster.
//
// WHAT mapa DOES (PTX ISA 7.8+, sm_90+)
// --------------------------------------
//   mapa.shared::cluster.u32  d, a, b;
//
// Remap shared-window address `a` onto cluster CTA rank `b`. Does NOT move data;
// a later ld/st.shared::cluster uses `d` to touch CTA b's copy of that offset.
//
// WHY AN 8-CTA VERSION IS A STRONGER TEST
// ---------------------------------------
// The 2-CTA test can only ever map to "the one other CTA", so it cannot tell a
// correct rank->CTA mapping apart from "always target the single peer". With 8
// CTAs we pin the mapping down far more tightly:
//
//   Pairing — MIRROR ACROSS THE CLUSTER
//     rank r talks to rank (CLUSTER_SIZE-1 - r): 0<->7, 1<->6, 2<->5, 3<->4.
//     peer() is an involution with peer(r) != r for all r, so the 8 CTAs form
//     4 disjoint pairs and every slot has exactly one remote writer.
//
//   Gate A — RANK SENSITIVITY (address-only)
//     For each CTA, mapa(local, peer) MUST differ from mapa(local, self).
//
//   Gate A+ — FULL RANK ROUND-TRIP (the real upgrade)
//     For every rank k in [0, CLUSTER_SIZE), getctarank(mapa(local, k)) == k.
//     This proves mapa lands on the *correct* CTA for all 8 ranks, not merely
//     that two addresses differ. An off-by-one or "always the peer" mapping is
//     caught here.
//
//   Gate B — REMOTE STORE, LOCAL LOAD (end-to-end)
//     Each CTA writes its magic into its *peer* slot via
//       st.shared::cluster [mapa(local, peer)]
//     then reads its *own* slot with an ordinary shared load (no mapa).
//     Expected local value == peer's magic. A no-op mapa writes its own slot
//     instead, so the local load sees its own magic and fails.
//
//   Gate C — SELF MAP IS IDENTITY-ISH
//     mapa(local, own_rank) must still address this CTA's slot (local load
//     through that mapped addr returns what the peer wrote into us, Gate B).
//
// Host checks all gates after D2H. Poison-fill so a silent no-op is obvious.
//
// NOTE (silicon): 8 CTAs is the portable maximum cluster size, so no
// non-portable opt-in is required. All CTAs of a cluster must be co-resident
// in one GPC on Hopper (sm_90); GPGPU-Sim models the 8-CTA cluster directly
// (needs >= 8 SMs, which the H100 config provides).
//
//   make KERNEL=mapa_shared_cluster_extended
//   ./run.sh mapa_shared_cluster_extended
//   ./run.sh mapa_shared_cluster_extended silicon
// ─────────────────────────────────────────────────────────────────────────────

#define CLUSTER_SIZE 8
#define BLOCK_SIZE   32
#define MAGIC_BASE   0xC000u
#define POISON       0xFFFFFFFFu

// Mirror pairing across the cluster: 0<->7, 1<->6, 2<->5, 3<->4.
__device__ __host__ __forceinline__ unsigned mirror_peer(unsigned rank) {
    return (CLUSTER_SIZE - 1u) - rank;
}

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

// Inverse of mapa's rank encoding: which cluster rank owns this mapped address.
__device__ __forceinline__ unsigned getctarank_shared_cluster_u32(unsigned addr) {
    unsigned rank;
    asm volatile("getctarank.shared::cluster.u32 %0, %1;"
                 : "=r"(rank)
                 : "r"(addr));
    return rank;
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
mapa_shared_cluster_extended_kernel(unsigned *d_local_after,     // Gate B
                                    unsigned *d_self_via_mapa,   // Gate C
                                    unsigned *d_mapped_peer,     // Gate A
                                    unsigned *d_mapped_self,     // Gate A
                                    unsigned *d_local_addr,
                                    unsigned *d_roundtrip_fail,  // Gate A+
                                    unsigned *d_peer) {
    __shared__ unsigned slot;

    const unsigned tid  = threadIdx.x;
    const unsigned rank = cluster_ctarank();
    const unsigned peer = mirror_peer(rank);

    // Shared-window address of our local slot (CUTLASS / PTX mapa input form).
    const unsigned local_addr =
        static_cast<unsigned>(__cvta_generic_to_shared(&slot));

    // Start from a known empty slot so Gate B cannot see a self-written magic.
    if (tid == 0) {
        slot = 0u;
    }
    __syncthreads();

    // Cluster barrier so every CTA's zeroing is globally visible before any
    // remote store — otherwise a late zero could clobber a peer's magic.
    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;" ::: "memory");

    if (tid == 0) {
        // ── Gate B: write *into the peer CTA* via mapa ─────────────────────
        const unsigned mapped_peer = mapa_shared_cluster_u32(local_addr, peer);
        st_shared_cluster_u32(mapped_peer, MAGIC_BASE + rank);
    }

    // Make remote stores visible before anyone reads their local slot.
    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;" ::: "memory");
    __syncthreads();

    if (tid == 0) {
        // Ordinary .shared::cta load of *our* slot — our mirror peer should have
        // written MAGIC_BASE + peer into it through Gate B's remapped store.
        const unsigned local_after = slot;

        // ── Gate C: self-map still names our own slot ──────────────────────
        const unsigned mapped_self =
            mapa_shared_cluster_u32(local_addr, rank);
        const unsigned self_via_mapa = ld_shared_cluster_u32(mapped_self);

        // Recompute peer map for the host-visible address dump (same as Gate B).
        const unsigned mapped_peer = mapa_shared_cluster_u32(local_addr, peer);

        // ── Gate A+: every rank must round-trip through mapa/getctarank ────
        unsigned rt_fail = 0u;
        #pragma unroll
        for (unsigned k = 0; k < CLUSTER_SIZE; ++k) {
            const unsigned m = mapa_shared_cluster_u32(local_addr, k);
            if (getctarank_shared_cluster_u32(m) != k) {
                rt_fail++;
            }
        }

        d_local_after[rank]    = local_after;
        d_self_via_mapa[rank]  = self_via_mapa;
        d_mapped_peer[rank]    = mapped_peer;
        d_mapped_self[rank]    = mapped_self;
        d_local_addr[rank]     = local_addr;
        d_roundtrip_fail[rank] = rt_fail;
        d_peer[rank]           = peer;
    }
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);

    printf("=== mapa.shared::cluster.u32 extended (8-CTA) functional ubench ===\n");
    printf("cluster=%d CTAs, block=%d threads, magic_base=0x%X\n",
           CLUSTER_SIZE, BLOCK_SIZE, MAGIC_BASE);
    printf("pairing: rank r <-> (%d - r)  [0<->%d, 1<->%d, ...]\n\n",
           CLUSTER_SIZE - 1, CLUSTER_SIZE - 1, CLUSTER_SIZE - 2);

    const int n = CLUSTER_SIZE;
    const size_t bytes = n * sizeof(unsigned);

    unsigned *h_local_after   = (unsigned *)malloc(bytes);
    unsigned *h_self_via_mapa = (unsigned *)malloc(bytes);
    unsigned *h_mapped_peer   = (unsigned *)malloc(bytes);
    unsigned *h_mapped_self   = (unsigned *)malloc(bytes);
    unsigned *h_local_addr    = (unsigned *)malloc(bytes);
    unsigned *h_roundtrip     = (unsigned *)malloc(bytes);
    unsigned *h_peer          = (unsigned *)malloc(bytes);

    unsigned *d_local_after = NULL, *d_self_via_mapa = NULL;
    unsigned *d_mapped_peer = NULL, *d_mapped_self = NULL;
    unsigned *d_local_addr = NULL, *d_roundtrip = NULL, *d_peer = NULL;

    cudaMalloc(&d_local_after,   bytes);
    cudaMalloc(&d_self_via_mapa, bytes);
    cudaMalloc(&d_mapped_peer,   bytes);
    cudaMalloc(&d_mapped_self,   bytes);
    cudaMalloc(&d_local_addr,    bytes);
    cudaMalloc(&d_roundtrip,     bytes);
    cudaMalloc(&d_peer,          bytes);

    cudaMemset(d_local_after,   0xFF, bytes);
    cudaMemset(d_self_via_mapa, 0xFF, bytes);
    cudaMemset(d_mapped_peer,   0xFF, bytes);
    cudaMemset(d_mapped_self,   0xFF, bytes);
    cudaMemset(d_local_addr,    0xFF, bytes);
    cudaMemset(d_roundtrip,     0xFF, bytes);
    cudaMemset(d_peer,          0xFF, bytes);

    // 8 CTAs is the portable maximum cluster size, so no non-portable opt-in
    // is needed here.
    mapa_shared_cluster_extended_kernel<<<CLUSTER_SIZE, BLOCK_SIZE>>>(
        d_local_after, d_self_via_mapa, d_mapped_peer, d_mapped_self,
        d_local_addr, d_roundtrip, d_peer);
    cudaError_t err = cudaDeviceSynchronize();

    int fails = 0;
    if (err != cudaSuccess) {
        printf("launch/sync error: %s\n", cudaGetErrorString(err));
        fails++;
    } else {
        cudaMemcpy(h_local_after,   d_local_after,   bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_self_via_mapa, d_self_via_mapa, bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_mapped_peer,   d_mapped_peer,   bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_mapped_self,   d_mapped_self,   bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_local_addr,    d_local_addr,    bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_roundtrip,     d_roundtrip,     bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_peer,          d_peer,          bytes, cudaMemcpyDeviceToHost);

        printf("per-CTA results (after D2H):\n");
        for (int r = 0; r < n; ++r) {
            const unsigned expect_peer  = mirror_peer((unsigned)r);
            // After the remote store, my slot holds my mirror peer's magic.
            const unsigned expect_local = MAGIC_BASE + expect_peer;
            const unsigned expect_self  = expect_local;  // self-map -> same slot

            const int wrote =
                (h_local_after[r] != POISON) && (h_self_via_mapa[r] != POISON) &&
                (h_mapped_peer[r] != POISON) && (h_mapped_self[r] != POISON) &&
                (h_local_addr[r]  != POISON) && (h_roundtrip[r]   != POISON);

            // Gate A: peer map must differ from self map (peer != self always).
            const int peer_addr_differs =
                (h_mapped_peer[r] != h_mapped_self[r]);

            // Gate A+: every rank round-tripped through mapa/getctarank.
            const int roundtrip_ok = (h_roundtrip[r] == 0u);

            // Gate B / C value checks.
            const int peer_ok  = (h_peer[r] == expect_peer);
            const int local_ok = (h_local_after[r] == expect_local);
            const int self_ok  = (h_self_via_mapa[r] == expect_self);

            if (!wrote)             fails++;
            if (!peer_addr_differs) fails++;
            if (!roundtrip_ok)      fails++;
            if (!peer_ok)           fails++;
            if (!local_ok)          fails++;
            if (!self_ok)           fails++;

            printf("  CTA rank %2d (peer %2d):\n", r, expect_peer);
            printf("    local_addr      = 0x%08X\n", h_local_addr[r]);
            printf("    mapped_self     = 0x%08X\n", h_mapped_self[r]);
            printf("    mapped_peer     = 0x%08X  [%s]\n",
                   h_mapped_peer[r],
                   peer_addr_differs ? "differs from self ok" : "SAME AS SELF");
            printf("    roundtrip_fail  = %u/%d  [%s]\n",
                   h_roundtrip[r], CLUSTER_SIZE,
                   roundtrip_ok ? "all ranks map correctly" : "RANK MAP WRONG");
            printf("    local_after     = 0x%08X  (expected 0x%08X) [%s]\n",
                   h_local_after[r], expect_local,
                   local_ok ? "ok" : "MISMATCH");
            printf("    self_via_mapa   = 0x%08X  (expected 0x%08X) [%s]\n",
                   h_self_via_mapa[r], expect_self,
                   self_ok ? "ok" : "MISMATCH");
        }
    }

    cudaFree(d_local_after);
    cudaFree(d_self_via_mapa);
    cudaFree(d_mapped_peer);
    cudaFree(d_mapped_self);
    cudaFree(d_local_addr);
    cudaFree(d_roundtrip);
    cudaFree(d_peer);
    free(h_local_after);
    free(h_self_via_mapa);
    free(h_mapped_peer);
    free(h_mapped_self);
    free(h_local_addr);
    free(h_roundtrip);
    free(h_peer);

    printf("\nRESULT: %s\n", fails == 0 ? "PASSED" : "FAILED");
    return fails == 0 ? 0 : 1;
}
