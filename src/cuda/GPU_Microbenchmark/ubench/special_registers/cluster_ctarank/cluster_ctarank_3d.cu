#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace {

constexpr unsigned kClusterX = 2;
constexpr unsigned kClusterY = 2;
constexpr unsigned kClusterZ = 2;
constexpr unsigned kGridX = 4;
constexpr unsigned kGridY = 4;
constexpr unsigned kGridZ = 4;
constexpr unsigned kThreadsPerCta = 32;
constexpr unsigned kTotalCtas = kGridX * kGridY * kGridZ;
constexpr unsigned kCtasPerCluster = kClusterX * kClusterY * kClusterZ;
// struct used to report results
struct CtaResult {
    unsigned block_x;
    unsigned block_y;
    unsigned block_z;
    unsigned ctaid_x;
    unsigned ctaid_y;
    unsigned ctaid_z;
    unsigned cluster_ctarank;
};
//error checking macro
#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        const cudaError_t error = (call);                                     \
        if (error != cudaSuccess) {                                           \
            std::fprintf(stderr, "%s:%d: CUDA error: %s\n", __FILE__,       \
                         __LINE__, cudaGetErrorString(error));                 \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

// gpu fns to read the %ctaid register in each dimension
__device__ __forceinline__ unsigned read_ctaid_x() {
    unsigned value;
    asm volatile("mov.u32 %0, %%ctaid.x;" : "=r"(value));
    return value;
}

__device__ __forceinline__ unsigned read_ctaid_y() {
    unsigned value;
    asm volatile("mov.u32 %0, %%ctaid.y;" : "=r"(value));
    return value;
}

__device__ __forceinline__ unsigned read_ctaid_z() {
    unsigned value;
    asm volatile("mov.u32 %0, %%ctaid.z;" : "=r"(value));
    return value;
}

// gpu fn to read the %cluster_ctarank register
__device__ __forceinline__ unsigned read_cluster_ctarank() {
    unsigned value;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(value));
    return value;
}

__global__ void cluster_ctarank_3d_kernel(CtaResult *results) {
     // The special register is the same for every thread in the CTA, so only read if
    // thread 0 to get single print and avoid interleaving of prints
    if (threadIdx.x != 0) {
        return;
    }

    const unsigned linearized_block_idx =
        blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
    results[linearized_block_idx] = {blockIdx.x,
                             blockIdx.y,
                             blockIdx.z,
                             read_ctaid_x(),
                             read_ctaid_y(),
                             read_ctaid_z(),
                             read_cluster_ctarank()};
}

}  // namespace

int main() {
    int device = 0;
    CHECK_CUDA(cudaGetDevice(&device));

    cudaDeviceProp properties{};
    CHECK_CUDA(cudaGetDeviceProperties(&properties, device));

    int cluster_launch_supported = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&cluster_launch_supported,
                                      cudaDevAttrClusterLaunch, device));
    if (!cluster_launch_supported) {
        std::fprintf(stderr, "%s does not support thread-block clusters\n",
                     properties.name);
        return EXIT_FAILURE;
    }

    CtaResult *device_results = nullptr;
    CHECK_CUDA(cudaMalloc(&device_results, kTotalCtas * sizeof(CtaResult)));

    cudaLaunchAttribute attribute{};
    attribute.id = cudaLaunchAttributeClusterDimension;
    attribute.val.clusterDim.x = kClusterX;
    attribute.val.clusterDim.y = kClusterY;
    attribute.val.clusterDim.z = kClusterZ;

    cudaLaunchConfig_t config{};
    config.gridDim = dim3(kGridX, kGridY, kGridZ);
    config.blockDim = dim3(kThreadsPerCta, 1, 1);
    config.attrs = &attribute;
    config.numAttrs = 1;

        //check supported cluster size
    int max_cluster_size = 0;
    CHECK_CUDA(cudaOccupancyMaxPotentialClusterSize(
        &max_cluster_size,
        (void*)cluster_ctarank_3d_kernel,
        &config));

    printf("Maximum supported cluster size: %d\n", max_cluster_size);
    
    // verify ctas/cluster requested is valid
    if (kCtasPerCluster > static_cast<unsigned>(max_cluster_size)) {
        std::fprintf(stderr,
                    "Requested %u CTAs/cluster, but this GPU supports at most %d\n",
                    kTotalCtas, max_cluster_size);
        return EXIT_FAILURE;
    }

    CHECK_CUDA(cudaLaunchKernelEx(&config, cluster_ctarank_3d_kernel,
                                  device_results));
    CHECK_CUDA(cudaDeviceSynchronize());

    CtaResult host_results[kTotalCtas];
    CHECK_CUDA(cudaMemcpy(host_results, device_results, sizeof(host_results),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(device_results));

    std::printf("GPU: %s (compute capability %d.%d)\n", properties.name,
                properties.major, properties.minor);
    std::printf("grid=(%u,%u,%u), cluster=(%u,%u,%u), total=%u CTAs\n\n",
                kGridX, kGridY, kGridZ, kClusterX, kClusterY, kClusterZ,
                kTotalCtas);
    std::printf("cluster(x,y,z)  blockIdx(x,y,z)  %%ctaid(x,y,z)  "
                "%%cluster_ctarank  expected\n");

    unsigned failures = 0;
    for (unsigned z = 0; z < kGridZ; ++z) {
        for (unsigned y = 0; y < kGridY; ++y) {
            for (unsigned x = 0; x < kGridX; ++x) {
                const unsigned index = x + kGridX * (y + kGridY * z);
                const CtaResult result = host_results[index];
                const unsigned cluster_x = x / kClusterX;
                const unsigned cluster_y = y / kClusterY;
                const unsigned cluster_z = z / kClusterZ;
                const unsigned local_x = x % kClusterX;
                const unsigned local_y = y % kClusterY;
                const unsigned local_z = z % kClusterZ;
                const unsigned expected_rank =
                    local_x + kClusterX * (local_y + kClusterY * local_z);

                const bool pass = result.block_x == x && result.block_y == y &&
                                  result.block_z == z && result.ctaid_x == x &&
                                  result.ctaid_y == y && result.ctaid_z == z &&
                                  result.cluster_ctarank == expected_rank;
                std::printf("(%u,%u,%u)         (%u,%u,%u)            "
                            "(%u,%u,%u)          %2u              %2u  %s\n",
                            cluster_x, cluster_y, cluster_z, result.block_x,
                            result.block_y, result.block_z, result.ctaid_x,
                            result.ctaid_y, result.ctaid_z,
                            result.cluster_ctarank, expected_rank,
                            pass ? "PASS" : "FAIL");
                failures += !pass;
            }
        }
    }

    if (failures != 0) {
        std::printf("\nFAIL: %u CTA result(s) did not match\n", failures);
        return EXIT_FAILURE;
    }
    std::printf("\nPASS: 3D %%ctaid matched blockIdx and all cluster-local "
                "ranks matched\n");
    return EXIT_SUCCESS;
}
