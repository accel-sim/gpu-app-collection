#include <cuda_runtime.h>

#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>

namespace {
/*
 modify these values using cli args to test different numbers of clusters and
  ctas per clusters.
  example: 4 clusters, 8 ctas per cluster
  'cluster_ctarank 4 8'
*/
constexpr unsigned kDefaultClusters = 4;
constexpr unsigned kDefaultCTAsPerCluster = 4;
constexpr unsigned kThreadsPerCTA = 32;

// struct used to report results
struct CTAResult {
    unsigned block_idx;
    unsigned ctaid_x;
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

// gpu fn to read the %cluster_ctarank register
__device__ __forceinline__ unsigned read_cluster_ctarank() {
    unsigned rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
    return rank;
}
// gpu fn to read the %ctaid register in x dimension only
__device__ __forceinline__ unsigned read_ctaid_x() {
    unsigned ctaid_x;
    asm volatile("mov.u32 %0, %%ctaid.x;" : "=r"(ctaid_x));
    return ctaid_x;
}

__global__ void cluster_ctarank_kernel(CTAResult *results) {
    // The special register is the same for every thread in the CTA, so only read if
    // thread 0 to get single print and avoid interleaving of prints
    // blockIdx.x == ctaid.x
    if (threadIdx.x == 0) {
        results[blockIdx.x] =
            {blockIdx.x, read_ctaid_x(), read_cluster_ctarank()};
    }
}

//parse cli args and ensure positive
unsigned parse_positive(const char *text, const char *name) {
    errno = 0;
    char *end = nullptr;
    const unsigned long value = std::strtoul(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value == 0 ||
        value > UINT_MAX) {
        std::fprintf(stderr, "%s must be a positive integer (got '%s')\n",
                     name, text);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<unsigned>(value);
}

}  // namespace

int main(int argc, char **argv) {
    if (argc > 3) {
        std::fprintf(stderr, "usage: %s [clusters] [ctas-per-cluster]\n",
                     argv[0]);
        return EXIT_FAILURE;
    }

    const unsigned num_clusters =
        argc >= 2 ? parse_positive(argv[1], "clusters") : kDefaultClusters;
    const unsigned ctas_per_cluster =
        argc >= 3 ? parse_positive(argv[2], "ctas-per-cluster")
                  : kDefaultCTAsPerCluster;

    int device = 0;
    CHECK_CUDA(cudaGetDevice(&device));

    // check clusters are supported
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
    const unsigned total_ctas = num_clusters * ctas_per_cluster;

    //descirbe requested cluster shape
    cudaLaunchAttribute attribute{};
    attribute.id = cudaLaunchAttributeClusterDimension;
    attribute.val.clusterDim.x = ctas_per_cluster;
    attribute.val.clusterDim.y = 1;
    attribute.val.clusterDim.z = 1;

    // build kernel launch config
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(total_ctas, 1, 1);
    config.blockDim = dim3(kThreadsPerCTA, 1, 1);
    config.attrs = &attribute;
    config.numAttrs = 1;

    //check supported cluster size
    int max_cluster_size = 0;
    CHECK_CUDA(cudaOccupancyMaxPotentialClusterSize(
        &max_cluster_size,
        (void*)cluster_ctarank_kernel,
        &config));

    printf("Maximum supported cluster size: %d\n", max_cluster_size);

    // verify ctas/cluster requested is valid
    if (ctas_per_cluster > static_cast<unsigned>(max_cluster_size)) {
        std::fprintf(stderr,
                    "Requested %u CTAs/cluster, but this GPU supports at most %d\n",
                    ctas_per_cluster, max_cluster_size);
        return EXIT_FAILURE;
    }

    // calculate total number of CTAS and malloc enough space for device results
    const size_t bytes = total_ctas * sizeof(CTAResult);
    CTAResult *device_results = nullptr;
    CHECK_CUDA(cudaMalloc(&device_results, bytes));

    //launch kernel
    CHECK_CUDA(cudaLaunchKernelEx(
                                &config, 
                                cluster_ctarank_kernel, 
                                device_results));
    CHECK_CUDA(cudaDeviceSynchronize());

    CTAResult *host_results =
        static_cast<CTAResult *>(std::malloc(bytes));
    if (host_results == nullptr) {
        std::fprintf(stderr, "host allocation failed\n");
        CHECK_CUDA(cudaFree(device_results));
        return EXIT_FAILURE;
    }
    CHECK_CUDA(cudaMemcpy(host_results, device_results, bytes,
                          cudaMemcpyDeviceToHost));

    std::printf("GPU: %s (compute capability %d.%d)\n", properties.name,
                properties.major, properties.minor);
    std::printf("launch: %u clusters x %u CTAs/cluster = %u CTAs\n\n",
                num_clusters, ctas_per_cluster, total_ctas);
    std::printf("logical_cluster  blockIdx.x  %%ctaid.x  "
                "%%cluster_ctarank  expected_rank\n");

    unsigned failures = 0;
    for (unsigned block = 0; block < total_ctas; ++block) {
        const unsigned logical_cluster = block / ctas_per_cluster;
        const unsigned expected_rank = block % ctas_per_cluster;
        const CTAResult result = host_results[block];
        const bool pass = result.block_idx == block &&
                          result.ctaid_x == result.block_idx &&
                          result.cluster_ctarank == expected_rank;
        std::printf("%15u  %10u  %8u  %18u  %13u  %s\n", logical_cluster,
                    result.block_idx, result.ctaid_x, result.cluster_ctarank,
                    expected_rank,
                    pass ? "PASS" : "FAIL");
        failures += !pass;
    }

    std::free(host_results);
    CHECK_CUDA(cudaFree(device_results));

    if (failures != 0) {
        std::printf("\nFAIL: %u CTA result(s) did not match\n", failures);
        return EXIT_FAILURE;
    }
    std::printf("\nPASS: %%ctaid.x matched blockIdx.x and %%cluster_ctarank "
                "restarted at zero in every cluster\n");
    return EXIT_SUCCESS;
}
