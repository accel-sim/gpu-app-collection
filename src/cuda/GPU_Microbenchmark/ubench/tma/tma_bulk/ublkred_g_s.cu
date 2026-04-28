/**
 * UBLKRED_G_S Unit Test: TMA Bulk Reduction Global <- Shared
 * ===========================================================
 * Tests that cp.reduce.async.bulk correctly reduces data from shared
 * into global memory for several reduction operators.
 *
 * Strategy (per element i):
 *   - Global memory is initialized to an operator-specific BASE value.
 *   - Shared memory is initialized with the pattern (0, 1, 2, ...).
 *   - TMA performs a bulk reduction: global[i] = reduce(global[i], shared[i]).
 *   - Host computes the expected result and compares element-wise.
 *
 * Supported reduction kinds:
 *   - add : global += shared
 *   - min : global  = min(global, shared)
 *   - max : global  = max(global, shared)
 *   - and : global &= shared
 *   - or  : global |= shared
 *   - xor : global ^= shared
 *
 * Usage:
 *   ./ublkred_g_s [-n <elements>] [-i <iterations>] [-r <kind|all>]
 *
 *   -n <elements>   : number of elements (default: 1024)
 *   -i <iterations> : number of reduction iterations in-kernel (default: 1)
 *   -r <kind|all>   : one of {add,min,max,and,or,xor,all} (default: add)
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <unistd.h>
#include <cuda/ptx>

#define CUDA_SAFECALL(call)                                                 \
    {                                                                       \
        call;                                                               \
        cudaError err = cudaGetLastError();                                 \
        if (cudaSuccess != err) {                                           \
            fprintf(stderr,                                                 \
                "Cuda error in '%s' at %s:%d : %s.\n",                      \
                #call, __FILE__, __LINE__, cudaGetErrorString(err));        \
            fflush(stderr);                                                 \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    }

namespace ptx = cuda::ptx;

#if defined(__CUDA_MINIMUM_ARCH__) && __CUDA_MINIMUM_ARCH__ < 900
static_assert(false, "Requires sm_90 or newer for TMA bulk operations.");
#endif

static constexpr size_t buf_len = 1024;

// Base values used for each reduction kind.
static constexpr int32_t BASE_ADD = 100;
static constexpr int32_t BASE_MIN = 1000000;
static constexpr int32_t BASE_MAX = -1000000;
static constexpr int32_t BASE_AND = ~0;
static constexpr int32_t BASE_OR  = 0;
static constexpr int32_t BASE_XOR = 0xAAAAAAAA;

enum class ReductionKind : int {
    ADD = 0,
    MIN,
    MAX,
    AND,
    OR,
    XOR
};

static const char* reduction_kind_name(ReductionKind kind)
{
    switch (kind) {
        case ReductionKind::ADD: return "ADD";
        case ReductionKind::MIN: return "MIN";
        case ReductionKind::MAX: return "MAX";
        case ReductionKind::AND: return "AND";
        case ReductionKind::OR:  return "OR";
        case ReductionKind::XOR: return "XOR";
        default:                 return "UNKNOWN";
    }
}

// ============================================================================
// Kernel
// ============================================================================
__global__ void test_UBLKRED_G_S(int32_t *data, int run_iters, ReductionKind kind)
{
    // Shared memory buffer. The destination shared memory buffer of
    // a bulk operation should be 16-byte aligned.
    __shared__ alignas(16) int32_t smem_data[buf_len];

    size_t offset = blockIdx.x * blockDim.x;

    // Generate the test pattern (0, 1, 2, ...) in shared memory
    for (int i = threadIdx.x; i < (int)buf_len; i += blockDim.x) {
        smem_data[i] = offset + i;
    }

    ptx::fence_proxy_async(ptx::space_shared);
    __syncthreads();

    for (int iter = 0; iter < run_iters; iter++) {
        if (threadIdx.x == 0) {
            switch (kind) {
                case ReductionKind::ADD:
                    ptx::cp_reduce_async_bulk(
                        ptx::space_global,
                        ptx::space_shared,
                        ptx::op_add,
                        data + offset, smem_data, sizeof(smem_data));
                    break;
                case ReductionKind::MIN:
                    ptx::cp_reduce_async_bulk(
                        ptx::space_global,
                        ptx::space_shared,
                        ptx::op_min,
                        data + offset, smem_data, sizeof(smem_data));
                    break;
                case ReductionKind::MAX:
                    ptx::cp_reduce_async_bulk(
                        ptx::space_global,
                        ptx::space_shared,
                        ptx::op_max,
                        data + offset, smem_data, sizeof(smem_data));
                    break;
                case ReductionKind::AND:
                    ptx::cp_reduce_async_bulk(
                        ptx::space_global,
                        ptx::space_shared,
                        ptx::op_and_op,
                        data + offset, smem_data, sizeof(smem_data));
                    break;
                case ReductionKind::OR:
                    ptx::cp_reduce_async_bulk(
                        ptx::space_global,
                        ptx::space_shared,
                        ptx::op_or_op,
                        data + offset, smem_data, sizeof(smem_data));
                    break;
                case ReductionKind::XOR:
                    ptx::cp_reduce_async_bulk(
                        ptx::space_global,
                        ptx::space_shared,
                        ptx::op_xor_op,
                        data + offset, smem_data, sizeof(smem_data));
                    break;
            }
            ptx::cp_async_bulk_commit_group();
            ptx::cp_async_bulk_wait_group_read(ptx::n32_t<0>());
        }
    }
}

// ============================================================================
// Host-side helpers
// ============================================================================
static int32_t base_for_kind(ReductionKind kind)
{
    switch (kind) {
        case ReductionKind::ADD: return BASE_ADD;
        case ReductionKind::MIN: return BASE_MIN;
        case ReductionKind::MAX: return BASE_MAX;
        case ReductionKind::AND: return BASE_AND;
        case ReductionKind::OR:  return BASE_OR;
        case ReductionKind::XOR: return BASE_XOR;
        default:                 return 0;
    }
}

static void init_global(int32_t *h_data, int n, ReductionKind kind)
{
    int32_t base = base_for_kind(kind);
    for (int i = 0; i < n; i++) {
        h_data[i] = base;
    }
}

static int32_t expected_value_for_index(int idx, ReductionKind kind, int run_iters)
{
    const int32_t i = (int32_t)idx;

    switch (kind) {
        case ReductionKind::ADD:
            // After run_iters iterations: base + run_iters * i
            return BASE_ADD + (int32_t)(run_iters * i);

        case ReductionKind::MIN:
            // min(BASE_MIN, i)
            return (i < BASE_MIN) ? i : BASE_MIN;

        case ReductionKind::MAX:
            // max(BASE_MAX, i)
            return (i > BASE_MAX) ? i : BASE_MAX;

        case ReductionKind::AND:
            // BASE_AND & i (idempotent across iterations)
            return (BASE_AND & i);

        case ReductionKind::OR:
            // BASE_OR | i (idempotent across iterations)
            return (BASE_OR | i);

        case ReductionKind::XOR:
            // XOR toggles each iteration
            if ((run_iters & 1) == 0) {
                return BASE_XOR;
            } else {
                return BASE_XOR ^ i;
            }
    }
    return 0;
}

static bool verify_reduction(int32_t *h_result, int n, ReductionKind kind, int run_iters)
{
    int mismatch_count = 0;
    int first_mismatch_idx = -1;
    int32_t first_got = 0;
    int32_t first_expected = 0;

    for (int idx = 0; idx < n; idx++) {
        int32_t expected = expected_value_for_index(idx, kind, run_iters);
        if (h_result[idx] != expected) {
            mismatch_count++;
            if (first_mismatch_idx < 0) {
                first_mismatch_idx = idx;
                first_got = h_result[idx];
                first_expected = expected;
            }
        }
    }

    if (mismatch_count > 0) {
        printf("  First mismatch at index %d: got %d, expected %d\n",
               first_mismatch_idx, first_got, first_expected);
        printf("  FAIL: %d/%d elements have wrong values\n", mismatch_count, n);
        return false;
    }

    return true;
}

static bool run_single_reduction(ReductionKind kind, int n, int run_iters)
{
    size_t bytes = (size_t)n * sizeof(int32_t);
    uint32_t blockSize = 1024;
    uint32_t gridSize = (uint32_t)ceil((float)n / blockSize);

    int32_t *h_data = (int32_t *)malloc(bytes);
    if (!h_data) {
        fprintf(stderr, "Host allocation failed\n");
        return false;
    }

    int32_t *d_a = nullptr;
    cudaMalloc(&d_a, bytes);

    // Initialize global memory on host and copy to device
    init_global(h_data, n, kind);
    cudaMemcpy(d_a, h_data, bytes, cudaMemcpyHostToDevice);

    // Run kernel
    CUDA_SAFECALL((test_UBLKRED_G_S<<<gridSize, blockSize>>>(d_a, run_iters, kind)));
    cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(h_data, d_a, bytes, cudaMemcpyDeviceToHost);

    // Verify
    printf("=== UBLKRED_G_S (TMA Bulk Reduction: %s) ===\n", reduction_kind_name(kind));
    bool pass = verify_reduction(h_data, n, kind, run_iters);
    printf("RESULT: %s\n\n", pass ? "PASS" : "FAIL");

    free(h_data);
    cudaFree(d_a);

    return pass;
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char *argv[])
{
    int n = 1024;
    int run_iters = 1;
    const char *reduction_str = "add";
    bool test_all = false;

    int opt;
    while ((opt = getopt(argc, argv, "n:i:r:")) != -1) {
        switch (opt) {
            case 'n':
                n = atoi(optarg);
                break;
            case 'i':
                run_iters = atoi(optarg);
                break;
            case 'r':
                reduction_str = optarg;
                break;
            default:
                fprintf(stderr,
                        "Usage: %s [-n <elements>] [-i <iterations>] [-r <kind|all>]\n",
                        argv[0]);
                fprintf(stderr,
                        "  -r <kind|all> where <kind> is one of: add, min, max, and, or, xor, all\n");
                return 1;
        }
    }

    ReductionKind selected = ReductionKind::ADD;

    if (strcmp(reduction_str, "add") == 0) {
        selected = ReductionKind::ADD;
    } else if (strcmp(reduction_str, "min") == 0) {
        selected = ReductionKind::MIN;
    } else if (strcmp(reduction_str, "max") == 0) {
        selected = ReductionKind::MAX;
    } else if (strcmp(reduction_str, "and") == 0) {
        selected = ReductionKind::AND;
    } else if (strcmp(reduction_str, "or") == 0) {
        selected = ReductionKind::OR;
    } else if (strcmp(reduction_str, "xor") == 0) {
        selected = ReductionKind::XOR;
    } else if (strcmp(reduction_str, "all") == 0) {
        test_all = true;
    } else {
        fprintf(stderr,
                "Invalid reduction type '%s'. Expected one of: add, min, max, and, or, xor, all\n",
                reduction_str);
        return 1;
    }

    bool overall_pass = true;

    if (test_all) {
        ReductionKind kinds[] = {
            ReductionKind::ADD,
            ReductionKind::MIN,
            ReductionKind::MAX,
            ReductionKind::AND,
            ReductionKind::OR,
            ReductionKind::XOR
        };
        const int num_kinds = (int)(sizeof(kinds) / sizeof(kinds[0]));

        for (int i = 0; i < num_kinds; i++) {
            bool pass = run_single_reduction(kinds[i], n, run_iters);
            overall_pass = overall_pass && pass;
        }
    } else {
        overall_pass = run_single_reduction(selected, n, run_iters);
    }

    return overall_pass ? 0 : 1;
}

