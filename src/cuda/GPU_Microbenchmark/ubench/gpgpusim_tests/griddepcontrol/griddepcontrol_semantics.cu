// griddepcontrol_semantics.cu
// Semantic tests for CUDA Programmatic Dependent Launch / grid dependency
// control.
//
// Covers:
//   1. Explicit trigger smoke test
//   2. Implicit trigger smoke test
//   3. Large memory visibility: pre-trigger and post-trigger writes
//   4. Uneven CTA trigger timing: dependent launch must wait for all primary
//      CTAs to trigger
//   5. No-attribute control: without programmatic stream serialization, normal
//      stream ordering applies
//
// Notes for simulator bring-up:
//   - overlap_seen is intentionally a soft observation, not a hard requirement.
//   - Correctness is based on data visibility after
//     cudaGridDependencySynchronize().
//   - The memory-visibility checks intentionally do not rely on __threadfence()
//     around primary writes.

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            fprintf(stderr, "%s:%d: CUDA error: %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(err__));          \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

static constexpr int BLOCKS            = 2;
static constexpr int THREADS           = 32;
static constexpr int WORDS_PER_BLOCK   = 256;
static constexpr int PRE_SPIN          = 200;
static constexpr int POST_TRIGGER_SPIN = 200;
static constexpr int LONG_SPIN         = 4000;

// Device-global witness flags.
// secondary_started: set by secondary as soon as it begins executing
// overlap_seen:      set by primary if it sees secondary_started before
//                    finishing
__device__ volatile int secondary_started;
__device__ volatile int overlap_seen;

__device__ __forceinline__ void burn_cycles(int n) {
    volatile int x = threadIdx.x + blockIdx.x;
    for (int i = 0; i < n; ++i) {
        x = x * 1664525 + 1013904223;
    }
}

__device__ __forceinline__ int pre_pattern(int block, int i) {
    return 0x10000000 ^ (block * 0x1009) ^ i;
}

__device__ __forceinline__ int post_pattern(int block, int i) {
    return 0x20000000 ^ (block * 0x2003) ^ i;
}

__device__ __forceinline__ void write_pre_region(int *buf) {
    int base = blockIdx.x * WORDS_PER_BLOCK;
    for (int i = threadIdx.x; i < WORDS_PER_BLOCK / 2; i += blockDim.x) {
        buf[base + i] = pre_pattern(blockIdx.x, i);
    }
}

__device__ __forceinline__ void write_post_region(int *buf) {
    int base = blockIdx.x * WORDS_PER_BLOCK;
    for (int i = WORDS_PER_BLOCK / 2 + threadIdx.x; i < WORDS_PER_BLOCK;
         i += blockDim.x) {
        buf[base + i] = post_pattern(blockIdx.x, i);
    }
}

__device__ __forceinline__ void observe_secondary_and_finish(int *done) {
    for (int i = 0; i < POST_TRIGGER_SPIN; ++i) {
        if (threadIdx.x == 0 && secondary_started) {
            overlap_seen = 1;
        }
        burn_cycles(1);
    }

    if (threadIdx.x == 0) {
        done[blockIdx.x] = 1;
    }
}

// Explicit primary: write first region, trigger dependent launch eligibility,
// then continue with post-trigger writes.
__global__ void primary_explicit_memory(int *done, int *buf) {
    burn_cycles(PRE_SPIN);
    write_pre_region(buf);

    if (threadIdx.x == 0) {
        cudaTriggerProgrammaticLaunchCompletion();
    }

    write_post_region(buf);
    observe_secondary_and_finish(done);
}

// Implicit primary: no explicit trigger; release is implicit at primary
// completion.
__global__ void primary_implicit_memory(int *done, int *buf) {
    burn_cycles(PRE_SPIN);
    write_pre_region(buf);
    write_post_region(buf);
    observe_secondary_and_finish(done);
}

// Uneven explicit trigger timing:
//   block 0 writes pre, triggers early, then spins and writes post
//   block 1 writes pre, spins first, triggers late, then writes post
// The secondary may start only after all primary CTAs have reached
// trigger/completion eligibility.
__global__ void primary_explicit_uneven(int *done, int *buf) {
    write_pre_region(buf);

    if (blockIdx.x == 0) {
        if (threadIdx.x == 0) {
            cudaTriggerProgrammaticLaunchCompletion();
        }
        burn_cycles(LONG_SPIN);
    } else {
        burn_cycles(LONG_SPIN);
        if (threadIdx.x == 0) {
            cudaTriggerProgrammaticLaunchCompletion();
        }
    }

    write_post_region(buf);
    observe_secondary_and_finish(done);
}

__global__ void secondary_check_full(const int *done,
                                     const int *buf,
                                     int *errors,
                                     int *presync_counter) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        secondary_started = 1;
    }

    if (threadIdx.x == 0) {
        atomicAdd(presync_counter, 1);
    }

    // The key operation under test.
    cudaGridDependencySynchronize();

    // Check original-style per-primary-CTA completion witness.
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < BLOCKS) {
        if (done[tid] != 1) {
            atomicAdd(errors, 1);
        }
    }

    // Check larger memory visibility, including pre-trigger and post-trigger
    // writes.
    int total_words = BLOCKS * WORDS_PER_BLOCK;
    for (int idx = tid; idx < total_words; idx += gridDim.x * blockDim.x) {
        int b = idx / WORDS_PER_BLOCK;
        int i = idx % WORDS_PER_BLOCK;
        int expected =
            (i < WORDS_PER_BLOCK / 2) ? pre_pattern(b, i) : post_pattern(b, i);
        if (buf[idx] != expected) {
            atomicAdd(errors, 1);
        }
    }
}

static void reset_device_flags() {
    int zero = 0;
    CHECK_CUDA(cudaMemcpyToSymbol(secondary_started, &zero, sizeof(int)));
    CHECK_CUDA(cudaMemcpyToSymbol(overlap_seen, &zero, sizeof(int)));
}

static int copy_overlap_seen_to_host() {
    int h_overlap = 0;
    CHECK_CUDA(cudaMemcpyFromSymbol(&h_overlap, overlap_seen, sizeof(int)));
    return h_overlap;
}

static int copy_secondary_started_to_host() {
    int h_started = 0;
    CHECK_CUDA(cudaMemcpyFromSymbol(&h_started, secondary_started, sizeof(int)));
    return h_started;
}

enum class PrimaryKind {
    ExplicitMemory,
    ImplicitMemory,
    ExplicitUneven,
};

static void launch_primary(PrimaryKind kind,
                           int *d_done,
                           int *d_buf,
                           cudaStream_t stream) {
    switch (kind) {
        case PrimaryKind::ExplicitMemory:
            primary_explicit_memory<<<BLOCKS, THREADS, 0, stream>>>(d_done,
                                                                    d_buf);
            break;
        case PrimaryKind::ImplicitMemory:
            primary_implicit_memory<<<BLOCKS, THREADS, 0, stream>>>(d_done,
                                                                    d_buf);
            break;
        case PrimaryKind::ExplicitUneven:
            primary_explicit_uneven<<<BLOCKS, THREADS, 0, stream>>>(d_done,
                                                                    d_buf);
            break;
    }
    CHECK_CUDA(cudaGetLastError());
}

static void run_test(const char *name,
                     PrimaryKind kind,
                     bool enable_pdl_attribute) {
    int *d_done    = nullptr;
    int *d_buf     = nullptr;
    int *d_errors  = nullptr;
    int *d_presync = nullptr;

    const int total_words = BLOCKS * WORDS_PER_BLOCK;

    CHECK_CUDA(cudaMalloc(&d_done, BLOCKS * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_buf, total_words * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_errors, sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_presync, sizeof(int)));

    CHECK_CUDA(cudaMemset(d_done, 0, BLOCKS * sizeof(int)));
    CHECK_CUDA(cudaMemset(d_buf, 0, total_words * sizeof(int)));
    CHECK_CUDA(cudaMemset(d_errors, 0, sizeof(int)));
    CHECK_CUDA(cudaMemset(d_presync, 0, sizeof(int)));

    reset_device_flags();

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed =
        enable_pdl_attribute ? 1 : 0;

    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(1);
    cfg.blockDim = dim3(128);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream;
    cfg.attrs = enable_pdl_attribute ? &attr : nullptr;
    cfg.numAttrs = enable_pdl_attribute ? 1 : 0;

    launch_primary(kind, d_done, d_buf, stream);

    CHECK_CUDA(cudaLaunchKernelEx(&cfg, secondary_check_full, d_done, d_buf,
                                  d_errors, d_presync));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    int h_errors = -1;
    int h_presync = -1;
    int h_overlap = copy_overlap_seen_to_host();
    int h_secondary_started = copy_secondary_started_to_host();
    std::vector<int> h_done(BLOCKS, 0);

    CHECK_CUDA(cudaMemcpy(&h_errors, d_errors, sizeof(int),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&h_presync, d_presync, sizeof(int),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_done.data(), d_done, BLOCKS * sizeof(int),
                          cudaMemcpyDeviceToHost));

    bool all_done = true;
    for (int i = 0; i < BLOCKS; ++i) {
        if (h_done[i] != 1) {
            all_done = false;
            break;
        }
    }

    printf("[%s] pdl_attr=%d, secondary_started=%d, presync_counter=%d, "
           "overlap_seen=%d, errors=%d, all_done=%s\n",
           name, enable_pdl_attribute ? 1 : 0, h_secondary_started, h_presync,
           h_overlap, h_errors, all_done ? "true" : "false");

    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(d_done));
    CHECK_CUDA(cudaFree(d_buf));
    CHECK_CUDA(cudaFree(d_errors));
    CHECK_CUDA(cudaFree(d_presync));

    // Hard correctness requirements:
    //   1. dependent reads after cudaGridDependencySynchronize() must be correct
    //   2. all primary blocks must eventually finish
    if (h_errors != 0 || !all_done) {
        fprintf(stderr, "[%s] FAILED: semantic correctness check failed\n",
                name);
        std::exit(2);
    }

    // Soft observations:
    //   - explicit PDL cases may show overlap_seen=1, but this is
    //     scheduling/resource dependent.
    //   - no_attr_control is expected to have overlap_seen=0 under normal
    //     stream serialization, but correctness should not depend on asserting
    //     that in this portable test.
}

int main() {
    run_test("explicit_basic_memory", PrimaryKind::ExplicitMemory, true);
    run_test("explicit_uneven_trigger", PrimaryKind::ExplicitUneven, true);
    run_test("implicit_memory", PrimaryKind::ImplicitMemory, true);
    run_test("no_attribute_control", PrimaryKind::ExplicitMemory, false);

    printf("PASS\n");
    return 0;
}
