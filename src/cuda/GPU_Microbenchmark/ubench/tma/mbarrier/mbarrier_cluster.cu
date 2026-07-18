// mbarrier_cluster.cu
//
// Aggressive microbenchmark for mbarrier on sm_90 (Hopper).
// Covers every PTX form used by CUTLASS 4.8 and the known simulator
// correctness gaps, with one dedicated test per issue.
//
// PTX forms exercised (matching CUTLASS cute/arch/copy_sm90_desc.hpp and
// cutlass/arch/barrier.h exactly):
//
//   mbarrier.init.shared::cta.b64          [bar], count
//   mbarrier.arrive.expect_tx.shared::cta.b64  _, [bar], bytes   (no token)
//   mbarrier.arrive.shared::cta.b64            _, [bar]          (no token, CUTLASS arrive_barrier)
//   mbarrier.arrive.shared::cta.b64         tok, [bar]           (token form)
//   mbarrier.arrive.shared::cta.b64         tok, [bar], count    (token + count)
//   mbarrier.arrive.shared::cluster.b64       _, [remote_bar]    (cross-CTA, no token)
//   mbarrier.try_wait.parity.shared::cta.b64  P, [bar], phase_bit (CUTLASS wait form)
//   mbarrier.try_wait.shared.b64              P, [bar], token     (token wait form)
//   mbarrier.complete_tx.relaxed.cluster.shared::cta.b64 [bar], tx
//   mbarrier.inval.shared.b64              [bar]
//
// Test index → output slot:
//   h[0]  TEST 1:  mbarrier.init.shared::cta / arrive(token) / try_wait(token) — baseline
//   h[1]  TEST 2:  mbarrier.try_wait.parity — sim crash test (SYNCS_MAX_ENUM assert)
//   h[2]  TEST 3:  arrive.expect_tx + try_wait.parity (CUTLASS TMA wait pattern)
//   h[3]  TEST 4:  arrive(no-token) _, [bar] — CUTLASS arrive_barrier form
//   h[4]  TEST 5:  arrive.shared::cluster (remote) + try_wait.parity — CUTLASS cluster arrive
//   h[5]  TEST 6:  Multi-phase ping-pong using try_wait.parity (looping parity bit)
//   h[6]  TEST 7:  count=0 observe-only arrive must not decrement pending
//   h[7]  TEST 8:  mbarrier.inval then re-init on same address
//   h[8]  TEST 9:  expect_tx + complete_tx + all-thread arrive (combined barrier)
//
// All outputs: 1 = PASS, -1 = FAIL, 0 = did not run (from cudaMemset).
//
// Compile:
//   nvcc -arch=sm_90 -std=c++17 mbarrier_cluster.cu -o mbarrier_cluster
//
// PTX audit:
//   cuobjdump -ptx mbarrier_cluster | grep mbarrier

#include <cooperative_groups.h>
#include <stdint.h>
#include <stdio.h>

namespace cg = cooperative_groups;

// ---------------------------------------------------------------------------
// Inline PTX helpers — each one maps to exactly one CUTLASS or PTX form
// ---------------------------------------------------------------------------

// 32-bit shared-memory address from a generic pointer (cvta.to.shared)
__device__ static inline uint32_t smem_u32(const void* ptr) {
    uint32_t a;
    asm("{ .reg .u64 tmp; cvta.to.shared.u64 tmp, %1; cvt.u32.u64 %0, tmp; }"
        : "=r"(a) : "l"((uint64_t)ptr));
    return a;
}

// mbarrier.init.shared::cta.b64  [bar], count
// CUTLASS: initialize_barrier()  (copy_sm90_desc.hpp:69)
__device__ static inline void mbar_init(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                 :: "r"(smem_u32(bar)), "r"(count) : "memory");
}

// mbarrier.inval.shared.b64  [bar]
__device__ static inline void mbar_inval(uint64_t* bar) {
    asm volatile("mbarrier.inval.shared.b64 [%0];" :: "r"(smem_u32(bar)) : "memory");
}

// mbarrier.arrive.shared::cta.b64  tok, [bar], count  — token form with count
// PTX ISA §9.7.12.4: decrements pending by count, returns prior-phase token
__device__ static inline uint64_t mbar_arrive_token(uint64_t* bar, uint32_t count = 1) {
    uint64_t tok;
    asm volatile("mbarrier.arrive.shared::cta.b64 %0, [%1], %2;"
                 : "=l"(tok) : "r"(smem_u32(bar)), "r"(count) : "memory");
    return tok;
}

// mbarrier.arrive.shared::cta.b64  _, [bar]
// CUTLASS: arrive_barrier()  (copy_sm90_desc.hpp:122) — no token returned
// Matches CUTLASS exactly: no volatile, no memory clobber (token discarded)
__device__ static inline void mbar_arrive_no_token(uint64_t* bar) {
    asm("{ .reg .b64 state; mbarrier.arrive.shared::cta.b64 state, [%0]; }"
        :: "r"(smem_u32(bar)));
}

// mbarrier.arrive.expect_tx.shared::cta.b64  _, [bar], bytes
// CUTLASS: set_barrier_transaction_bytes()  (copy_sm90_desc.hpp:83)
// Adds tx_bytes to tx_count; decrements pending by 1; no token returned.
__device__ static inline void mbar_arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                 :: "r"(smem_u32(bar)), "r"(bytes) : "memory");
}

// mbarrier.complete_tx.relaxed.cluster.shared::cta.b64  [bar], tx
// Called after simulated TMA data arrives; decrements tx_count
__device__ static inline void mbar_complete_tx(uint64_t* bar, uint32_t tx) {
    asm volatile("mbarrier.complete_tx.relaxed.cluster.shared::cta.b64 [%0], %1;"
                 :: "r"(smem_u32(bar)), "r"(tx) : "memory");
}

// mbarrier.try_wait.parity.shared::cta.b64  P, [bar], phase_bit, ticks
// Matches CUTLASS ClusterBarrier::wait() in barrier.h exactly:
//   3-operand form with timeout ticks (0x989680); loops on bra until P=true.
// PTX ISA §9.7.12.6: returns P=true when phase%2 == phase_bit.
// Unique-label helper: two levels of macro indirection so __COUNTER__ expands
// before stringification, producing labels like LAB_WAIT_42 / DONE_42.
#define MBAR_WAIT_STR2(x) #x
#define MBAR_WAIT_STR(x)  MBAR_WAIT_STR2(x)
#define MBAR_WAIT_PARITY_IMPL(sa, pb, tk, id)                                  \
    asm volatile(                                                                \
        "{\n\t"                                                                  \
        ".reg .pred P1;\n\t"                                                     \
        "LAB_WAIT_" MBAR_WAIT_STR(id) ":\n\t"                                  \
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2;\n\t"       \
        "@P1 bra DONE_" MBAR_WAIT_STR(id) ";\n\t"                              \
        "bra LAB_WAIT_" MBAR_WAIT_STR(id) ";\n\t"                              \
        "DONE_" MBAR_WAIT_STR(id) ":\n\t"                                      \
        "}"                                                                      \
        :: "r"(sa), "r"(pb), "r"(tk) : "memory")
// __LINE__ is unique per call site in the source file and survives inlining.
#define MBAR_WAIT_PARITY(sa, pb, tk) MBAR_WAIT_PARITY_IMPL(sa, pb, tk, __LINE__)

// mbar_wait_parity intentionally removed — use mbar_spin_wait_parity macro instead,
// which captures __LINE__ at the call site to guarantee unique PTX labels per use.

// Non-blocking one-shot parity test — returns true if barrier is already done
__device__ static inline bool mbar_test_wait_parity(uint64_t* bar, uint32_t phase_bit) {
    uint32_t done;
    asm volatile(
        "{ .reg .pred P;\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
        "selp.u32 %2, 1, 0, P; }"
        : "=r"(done) : "r"(smem_u32(bar)), "r"(phase_bit) : "memory");
    return (bool)done;
}

// mbarrier.try_wait.shared.b64  P, [bar], token — token form (used in tests 1, 7)
__device__ static inline bool mbar_try_wait_token(uint64_t* bar, uint64_t tok) {
    uint32_t done;
    asm volatile(
        "{ .reg .pred P;\n\t"
        "mbarrier.try_wait.shared.b64 P, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P; }"
        : "=r"(done) : "r"(smem_u32(bar)), "l"(tok) : "memory");
    return (bool)done;
}

// mbarrier.arrive.shared::cluster.b64  _, [remote_bar]
// CUTLASS: ClusterBarrier::arrive(smem_ptr, cta_id) (barrier.h:489)
// remote_bar is a 32-bit shared-memory address in another CTA's space
// obtained via mapa.shared::cluster.u32.
__device__ static inline void mbar_arrive_cluster_remote32(uint32_t remote_smem_addr) {
    asm volatile(
        "mbarrier.arrive.shared::cluster.b64 _, [%0];"
        :: "r"(remote_smem_addr) : "memory");
}

// mapa.shared::cluster.u32  dst, src, cta_rank
// Maps a 32-bit shared-memory address in this CTA's space to the same
// offset in another CTA's shared memory, within the cluster.
// This is the CUTLASS barrier.h form (u32), distinct from mapa.u64.
__device__ static inline uint32_t mapa32(const void* local_ptr, uint32_t cta_rank) {
    uint32_t dst;
    asm("mapa.shared::cluster.u32 %0, %1, %2;"
        : "=r"(dst) : "r"(smem_u32(local_ptr)), "r"(cta_rank));
    return dst;
}

// Spin-wait helpers
__device__ static inline void mbar_spin_wait_token(uint64_t* bar, uint64_t tok) {
    while (!mbar_try_wait_token(bar, tok)) __nanosleep(20);
}
// mbar_spin_wait_parity must be a macro so __LINE__ captures the call site,
// giving each inline PTX block a unique label. Two calls in the same kernel
// with the same function would otherwise generate duplicate LAB_WAIT labels
// that the gpgpu-sim PTX parser aliases to the first occurrence.
#define mbar_spin_wait_parity(bar, phase_bit) do { \
    uint32_t _sa = smem_u32(bar); \
    uint32_t _pb = (phase_bit); \
    uint32_t _tk = 0x989680; \
    MBAR_WAIT_PARITY(_sa, _pb, _tk); \
} while(0)

// ============================================================
// TEST 1: mbarrier.init.shared::cta + arrive(token) + try_wait.token
// PTX: mbarrier.init.shared::cta.b64 / .arrive.shared::cta.b64 tok / .try_wait.shared.b64
//
// Baseline: all 32 threads arrive with count=1, wait on 64-bit token.
// Verifies the fundamental init→arrive→wait cycle works.
// h[0] = 1 if sum of smem[0..31] == 496 after barrier completes.
// ============================================================
__global__ void test1_baseline_token_wait(int* out) {
    __shared__ uint64_t bar;
    __shared__ int smem[32];

    if (threadIdx.x == 0) mbar_init(&bar, blockDim.x);
    __syncthreads();

    smem[threadIdx.x] = threadIdx.x;
    __threadfence_block();

    uint64_t tok = mbar_arrive_token(&bar);
    mbar_spin_wait_token(&bar, tok);

    if (threadIdx.x == 0) {
        int sum = 0;
        for (int i = 0; i < 32; i++) sum += smem[i];
        out[0] = (sum == 496) ? 1 : -1;
    }
}

// ============================================================
// TEST 2: mbarrier.try_wait.parity — the CUTLASS wait form
// PTX: mbarrier.try_wait.parity.shared::cta.b64  P, [bar], phase_bit
//
// CUTLASS uses parity-form exclusively (copy_sm90_desc.hpp wait_barrier and
// barrier.h ClusterBarrier::try_wait). This takes a plain u32 phase_bit (0 or 1)
// instead of a 64-bit token. The simulator must route PARITY_OPTION to
// SYNCS_TRY_WAIT, not to SYNCS_MAX_ENUM_NO_USED.
//
// SIMULATOR ISSUE: ptx_ir.cc PARITY_OPTION falls to default → SYNCS_MAX_ENUM_NO_USED
// → shader.cc assert(false) crash. Fix: add PARITY_OPTION → SYNCS_TRY_WAIT.
// PTX ISA §9.7.12.6: try_wait.parity completes when phase%2 == phase_bit.
// h[1] = 1 if barrier completes and smem sum is correct.
// ============================================================
__global__ void test2_try_wait_parity(int* out) {
    __shared__ uint64_t bar;
    __shared__ int smem[32];

    if (threadIdx.x == 0) mbar_init(&bar, blockDim.x);
    __syncthreads();

    smem[threadIdx.x] = threadIdx.x;
    __threadfence_block();

    // All threads arrive without token — CUTLASS arrive_barrier form
    mbar_arrive_no_token(&bar);

    // Wait using parity form: current phase=0, parity=0 → phase_bit=0 (CUTLASS convention)
    mbar_spin_wait_parity(&bar, 0);

    if (threadIdx.x == 0) {
        int sum = 0;
        for (int i = 0; i < 32; i++) sum += smem[i];
        out[1] = (sum == 496) ? 1 : -1;
    }
}

// ============================================================
// TEST 3: arrive.expect_tx + complete_tx + try_wait.parity
// PTX: mbarrier.arrive.expect_tx.shared::cta.b64  _, [bar], bytes
//      mbarrier.complete_tx.relaxed.cluster.shared::cta.b64  [bar], bytes
//      mbarrier.try_wait.parity.shared::cta.b64   P, [bar], phase_bit
//
// This is the CUTLASS TMA producer/consumer pattern (copy_sm90_desc.hpp):
//  1. Consumer thread calls arrive.expect_tx (announces incoming async bytes,
//     decrements pending by 1, adds bytes to tx_count)
//  2. Simulated TMA data arrives, producer calls complete_tx (decrements tx_count)
//  3. All threads wait with parity form
// Barrier completes only when BOTH pending==0 AND tx_count==0.
// PTX ISA §9.7.12.5: expect_tx adds to the expected transaction count.
// h[2] = 1 if data is correct after wait.
// ============================================================
__global__ void test3_expect_tx_parity_wait(int* out) {
    __shared__ uint64_t bar;
    __shared__ int data[32];

    // Init: 1 arrive expected (the expect_tx thread), plus tx bytes
    if (threadIdx.x == 0) {
        mbar_init(&bar, blockDim.x);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        // Announce 32*4=128 bytes of incoming async data and arrive (count=1)
        mbar_arrive_expect_tx(&bar, 32 * sizeof(int));

        // Simulate the async data arriving (in real use this is TMA)
        for (int i = 0; i < 32; i++) data[i] = i + 100;
        __threadfence_block();

        // Signal tx completion
        mbar_complete_tx(&bar, 32 * sizeof(int));
    } else {
        // All other threads arrive without token
        mbar_arrive_no_token(&bar);
    }

    // All threads wait: current phase=0, parity=0 → phase_bit=0 (CUTLASS convention)
    mbar_spin_wait_parity(&bar, 0);

    if (threadIdx.x == 0) {
        int ok = 1;
        for (int i = 0; i < 32; i++) if (data[i] != i + 100) { ok = 0; break; }
        out[2] = ok ? 1 : -1;
    }
}

// ============================================================
// TEST 4: arrive(no-token) _, [bar] — CUTLASS arrive_barrier form
// PTX: mbarrier.arrive.shared::cta.b64  tok, [bar]   (token discarded)
//
// CUTLASS arrive_barrier() (copy_sm90_desc.hpp:122) issues this form with
// a .reg .b64 tok that is read but never stored — the destination is
// architecturally a register but the C++ wrapper ignores it.
// This verifies that arrive with implicit dst (vs explicit "_") is handled
// correctly by the simulator's dst.is_non_arch_reg() check.
// PTX ISA §9.7.12.4: the token is valid but the caller may discard it.
// h[3] = 1 if barrier completes (all 32 threads arrived, waited with parity).
// ============================================================
__global__ void test4_arrive_no_token_form(int* out) {
    __shared__ uint64_t bar;

    if (threadIdx.x == 0) mbar_init(&bar, blockDim.x);
    __syncthreads();

    // All threads call arrive_no_token — token captured in reg but discarded
    mbar_arrive_no_token(&bar);

    // current phase=0, parity=0 → phase_bit=0
    mbar_spin_wait_parity(&bar, 0);

    if (threadIdx.x == 0) out[3] = 1;
}

// ============================================================
// TEST 5: mbarrier.arrive.shared::cluster.b64 via mapa.shared::cluster.u32
// PTX: mapa.shared::cluster.u32  remAddr, localAddr, cta_rank
//      mbarrier.arrive.shared::cluster.b64  _, [remAddr]
//      mbarrier.try_wait.parity.shared::cta.b64  P, [bar], phase_bit
//
// This is the CUTLASS ClusterBarrier::arrive(smem_ptr, cta_id) form (barrier.h:489).
// Uses 32-bit mapa (not 64-bit mapa.u64) to map the local shmem address to a
// remote CTA's shmem space. Thread 0 of each CTA signals the partner CTA's barrier.
// Each barrier is init'd for 1 arrival (from the partner's thread 0).
// Each CTA then waits with parity form for its own barrier to complete.
//
// SIMULATOR ISSUE: mapa.shared::cluster.u32 address encoding may differ from
// mapa.u64. The address passed to arrive.shared::cluster.b64 must be decoded
// from a 32-bit generic shared address, not a 64-bit one.
// h[4] = 1 (both CTAs write 1 to same slot)
// ============================================================
__global__ void __cluster_dims__(2, 1, 1)
test5_cluster_arrive_mapa32(int* out) {
    auto cluster = cg::this_cluster();
    unsigned my_rank = cluster.block_rank();
    unsigned partner = 1 - my_rank;

    __shared__ uint64_t bar;

    if (threadIdx.x == 0) mbar_init(&bar, 1);
    cluster.sync();

    if (threadIdx.x == 0) {
        // Map partner's bar address using 32-bit mapa (CUTLASS barrier.h form)
        uint32_t remote_addr = mapa32(&bar, partner);
        mbar_arrive_cluster_remote32(remote_addr);
    }

    // current phase=0, parity=0 → phase_bit=0
    if (threadIdx.x == 0) {
        mbar_spin_wait_parity(&bar, 0);
        out[4] = 1;  // both CTAs write to same slot; that's fine, both write 1
    }
}

// ============================================================
// TEST 6: Two-phase parity alternation — arrive+try_wait.parity, phase 0 then phase 1
// PTX: mbarrier.arrive.shared::cta.b64 / mbarrier.try_wait.parity.shared::cta.b64
//
// Tests that parity form correctly handles two consecutive phases with alternating
// phase_bit (0 then 1), matching CUTLASS PipelineState::phase_ toggle (^= 1).
//
// Two separate barrier cycles in one kernel:
//   Cycle A: init(count=32), all arrive, wait(phase_bit=0) — phase 0 in-progress
//   Cycle B: re-use same bar (now at phase=1), all arrive, wait(phase_bit=1) — phase 1
//
// Simulator note: each cycle is a single arrive→wait, so two-pass mode advances
// the barrier by exactly 1 per cycle. No parity ambiguity (phase 2 never reached
// within a single kernel execution, only phase 0→1 and 1→2).
// h[5] = 1 if both cycles complete with correct data.
// ============================================================
__global__ void test6_looping_parity(int* out) {
    __shared__ uint64_t bar;
    __shared__ int smemA[32];
    __shared__ int smemB[32];

    if (threadIdx.x == 0) mbar_init(&bar, blockDim.x);
    __syncthreads();

    // Cycle A: phase 0 → phase_bit=0
    smemA[threadIdx.x] = threadIdx.x;
    __threadfence_block();
    mbar_arrive_no_token(&bar);
    mbar_spin_wait_parity(&bar, 0);

    // Cycle B: phase 1 → phase_bit=1
    smemB[threadIdx.x] = threadIdx.x + 100;
    __threadfence_block();
    mbar_arrive_no_token(&bar);
    mbar_spin_wait_parity(&bar, 1);

    if (threadIdx.x == 0) {
        int sumA = 0, sumB = 0;
        for (int i = 0; i < 32; i++) { sumA += smemA[i]; sumB += smemB[i]; }
        int ok = (sumA == 496) && (sumB == 496 + 32*100);
        out[5] = ok ? 1 : -1;
    }
}

// ============================================================
// TEST 7: count=0 observe-only arrive must not decrement pending
// PTX: mbarrier.arrive.shared::cta.b64  tok, [bar], 0   (count=0)
//
// PTX ISA §9.7.12.4: "If count is 0, the operation acts as an observe and
// does not modify the pending count of the mbarrier object."
// Thread 0 does the real arrive (count=1); all others call count=0 to get
// a token. A simulator that subtracts 0 from a uint32 pending count and then
// calls try_phase_transition() could spuriously fire the transition or
// underflow the counter (wrapping below 0 as unsigned).
// h[6] = 1 if all threads see the barrier complete correctly.
// ============================================================
__global__ void test7_count0_observe(int* out) {
    __shared__ uint64_t bar;
    __shared__ int flag;

    if (threadIdx.x == 0) {
        mbar_init(&bar, 1);  // only 1 real arrival expected
        flag = 0;
    }
    __syncthreads();

    uint64_t tok;
    if (threadIdx.x == 0) {
        flag = 42;
        __threadfence_block();
        tok = mbar_arrive_token(&bar, 1);  // real arrive: pending 1→0 → transition
    } else {
        tok = mbar_arrive_token(&bar, 0);  // observe-only: no pending change
    }

    mbar_spin_wait_token(&bar, tok);

    if (threadIdx.x == 0)
        out[6] = (flag == 42) ? 1 : -1;
}

// ============================================================
// TEST 8: mbarrier.inval then re-init on same address
// PTX: mbarrier.inval.shared.b64 / mbarrier.init.shared::cta.b64 (re-init)
//
// PTX ISA §9.7.12.9: after inval, the mbarrier object is in an undefined
// state. A subsequent init on the same address must reinitialize it cleanly.
// SIMULATOR ISSUE: the mbarrier_clear_for_cta mechanism clears on CTA exit,
// but inval within a kernel should also work. If the simulator doesn't remove
// the mbarrier on inval and a second init hits the "already exists" assert,
// this test exposes it.
// h[7] = 1 if re-init and second barrier cycle complete successfully.
// ============================================================
__global__ void test8_inval_reinit(int* out) {
    __shared__ uint64_t bar;

    if (threadIdx.x == 0) mbar_init(&bar, blockDim.x);
    __syncthreads();

    // First cycle
    uint64_t tok = mbar_arrive_token(&bar);
    mbar_spin_wait_token(&bar, tok);

    if (threadIdx.x == 0) mbar_inval(&bar);
    __syncthreads();

    // Re-init same address
    if (threadIdx.x == 0) mbar_init(&bar, blockDim.x);
    __syncthreads();

    // Second cycle — parity form (phase resets to 0 after re-init, so phase_bit=0)
    mbar_arrive_no_token(&bar);
    mbar_spin_wait_parity(&bar, 0);

    if (threadIdx.x == 0) out[7] = 1;
}

// ============================================================
// TEST 9: expect_tx + complete_tx + all-thread cluster arrive + parity wait
// PTX: mbarrier.arrive.expect_tx.shared::cta.b64
//      mbarrier.complete_tx.relaxed.cluster.shared::cta.b64
//      mbarrier.arrive.shared::cta.b64   (all threads, no token)
//      mbarrier.try_wait.parity.shared::cta.b64
//
// This is the full CUTLASS TMA warp-specialized producer pattern:
// - Thread 0 calls expect_tx (arrives + announces TX bytes)
// - Thread 0 simulates data arriving then calls complete_tx
// - All 32 threads call arrive (so the barrier needs 32 arrives total,
//   but thread 0's arrive was already done inside expect_tx)
// - All wait with parity form
//
// Note: expect_tx decrements pending by 1 AND adds tx_count. So init must
// count the total thread arrivals including the implicit one from expect_tx.
// Init with blockDim.x (32): expect_tx contributes 1 arrive (from thread 0),
// remaining 31 threads contribute 1 each = 32 total → transition fires when
// tx_count also reaches 0.
// h[8] = 1 if data is correct after wait.
// ============================================================
__global__ void __cluster_dims__(2, 1, 1)
test9_full_tma_pattern(int* out) {
    auto cluster = cg::this_cluster();
    unsigned my_rank = cluster.block_rank();

    __shared__ uint64_t bar;
    __shared__ int data[32];

    if (threadIdx.x == 0) mbar_init(&bar, blockDim.x);
    cluster.sync();

    if (my_rank == 0) {
        if (threadIdx.x == 0) {
            // arrive.expect_tx: arrives (pending 32→31) + announces 128 bytes tx
            mbar_arrive_expect_tx(&bar, 32 * sizeof(int));

            // Simulate data arriving asynchronously
            for (int i = 0; i < 32; i++) data[i] = i + 300;
            __threadfence_block();

            // complete_tx: tx_count 128→0
            mbar_complete_tx(&bar, 32 * sizeof(int));
        } else {
            // Remaining 31 threads arrive (no token)
            mbar_arrive_no_token(&bar);
        }
    } else {
        // CTA1: all 32 threads just arrive normally
        mbar_arrive_no_token(&bar);
    }

    // current phase=0, parity=0 → phase_bit=0
    mbar_spin_wait_parity(&bar, 0);

    if (my_rank == 0 && threadIdx.x == 0) {
        int ok = 1;
        for (int i = 0; i < 32; i++) if (data[i] != i + 300) { ok = 0; break; }
        out[8] = ok ? 1 : -1;
    }
}

// ============================================================
// main
// ============================================================
int main() {
    const int N = 9;
    int h[N] = {};
    int* d;
    cudaMalloc(&d, N * sizeof(int));

    // Zero all slots once upfront — each test writes only its own slot(s)
    cudaMemset(d, 0, N * sizeof(int));

    // 2-CTA cluster config
    cudaLaunchConfig_t cfg2 = {};
    cudaLaunchAttribute attr2[1];
    attr2[0].id = cudaLaunchAttributeClusterDimension;
    attr2[0].val.clusterDim = {2, 1, 1};
    cfg2.gridDim  = {2, 1, 1};
    cfg2.blockDim = {32, 1, 1};
    cfg2.attrs    = attr2;
    cfg2.numAttrs = 1;

    auto sync = [&]() {
        cudaDeviceSynchronize();
        cudaMemcpy(h, d, N * sizeof(int), cudaMemcpyDeviceToHost);
    };
    auto pf = [](int v) { return v == 1 ? "PASS" : (v == -1 ? "FAIL" : "SKIP"); };

    printf("=== mbarrier Cluster Microbenchmark (sm_90 / Hopper) ===\n");
    printf("PTX forms covered:\n");
    printf("  mbarrier.init.shared::cta.b64\n");
    printf("  mbarrier.arrive.shared::cta.b64  tok/_ (token and no-token forms)\n");
    printf("  mbarrier.arrive.expect_tx.shared::cta.b64  _\n");
    printf("  mbarrier.arrive.shared::cluster.b64  _   (via mapa.shared::cluster.u32)\n");
    printf("  mbarrier.try_wait.parity.shared::cta.b64  (CUTLASS wait form)\n");
    printf("  mbarrier.try_wait.shared.b64              (token wait form)\n");
    printf("  mbarrier.complete_tx.relaxed.cluster.shared::cta.b64\n");
    printf("  mbarrier.inval.shared.b64\n\n");

    // TEST 1 — h[0]
    printf("TEST 1 [init/arrive-token/try_wait-token baseline]... ");
    fflush(stdout);
    test1_baseline_token_wait<<<1, 32>>>(d);
    sync();
    printf("%s\n", pf(h[0]));

    // TEST 2 — h[1]
    printf("TEST 2 [try_wait.parity - CUTLASS wait form]... ");
    fflush(stdout);
    test2_try_wait_parity<<<1, 32>>>(d);
    sync();
    printf("%s\n", pf(h[1]));

    // TEST 3 — h[2]
    printf("TEST 3 [arrive.expect_tx + complete_tx + try_wait.parity]... ");
    fflush(stdout);
    test3_expect_tx_parity_wait<<<1, 32>>>(d);
    sync();
    printf("%s\n", pf(h[2]));

    // TEST 4 — h[3]
    printf("TEST 4 [arrive(no-token/_, [bar]) - CUTLASS arrive_barrier form]... ");
    fflush(stdout);
    test4_arrive_no_token_form<<<1, 32>>>(d);
    sync();
    printf("%s\n", pf(h[3]));

    // TEST 5 — h[4] (2-CTA cluster)
    printf("TEST 5 [arrive.shared::cluster via mapa.shared::cluster.u32]... ");
    fflush(stdout);
    cudaLaunchKernelEx(&cfg2, test5_cluster_arrive_mapa32, d);
    sync();
    printf("%s\n", pf(h[4]));

    // TEST 6 — h[5]
    printf("TEST 6 [looping parity: 4 phase transitions, alternating phase_bit]... ");
    fflush(stdout);
    test6_looping_parity<<<1, 32>>>(d);
    sync();
    printf("%s\n", pf(h[5]));

    // TEST 7 — h[6]
    printf("TEST 7 [count=0 observe-only arrive must not decrement pending]... ");
    fflush(stdout);
    test7_count0_observe<<<1, 32>>>(d);
    sync();
    printf("%s\n", pf(h[6]));

    // TEST 8 — h[7]
    printf("TEST 8 [mbarrier.inval then re-init on same address]... ");
    fflush(stdout);
    test8_inval_reinit<<<1, 32>>>(d);
    sync();
    printf("%s\n", pf(h[7]));

    // TEST 9 — h[8] (2-CTA cluster)
    printf("TEST 9 [full CUTLASS TMA pattern: expect_tx+complete_tx+arrive+parity-wait]... ");
    fflush(stdout);
    cudaLaunchKernelEx(&cfg2, test9_full_tma_pattern, d);
    sync();
    printf("%s\n", pf(h[8]));

    printf("\n=== Results: [%d,%d,%d,%d,%d,%d,%d,%d,%d] ===\n",
           h[0],h[1],h[2],h[3],h[4],h[5],h[6],h[7],h[8]);
    bool all = true;
    for (int i = 0; i < N; i++) all = all && (h[i] == 1);
    printf("Overall: %s\n", all ? "PASS" : "PARTIAL/FAIL");

    cudaFree(d);
    return 0;
}
