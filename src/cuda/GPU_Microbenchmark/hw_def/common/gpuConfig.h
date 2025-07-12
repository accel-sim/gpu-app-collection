#ifndef GPU_CONFIG_H
#define GPU_CONFIG_H

#include <string>
#include <cstdlib>
#include <iostream>

// Holds all GPU configuration parameters
struct GpuConfig
{
    unsigned SM_NUMBER = 80;              // Number of SMs
    unsigned WARP_SIZE = 32;              // Max threads per warp
    unsigned MAX_THREADS_PER_SM = 2048;   // Max threads per SM
    unsigned MAX_SHARED_MEM_SIZE = 98304; // Max shared memory size per SM (bytes)
    unsigned MAX_WARPS_PER_SM = 64;       // Max warps per SM
    unsigned MAX_REG_PER_SM = 65536;      // Max registers per SM

    unsigned MAX_THREAD_BLOCK_SIZE = 1024;          // Max threads per thread block
    unsigned MAX_SHARED_MEM_SIZE_PER_BLOCK = 49152; // Max shared memory per block (bytes)
    unsigned MAX_REG_PER_BLOCK = 32768;             // Max registers per block

    size_t L1_SIZE = 48 * 1024;       // L1 cache size (bytes)
    size_t L2_SIZE = 4 * 1024 * 1024; // L2 cache size (bytes)

    size_t MEM_SIZE = 16ULL * 1024 * 1024 * 1024; // Global memory size (bytes)
    unsigned MEM_CLK_FREQUENCY = 1375;            // Memory clock frequency (MHz)
    unsigned MEM_BITWIDTH = 512;                  // Memory interface bit width
    unsigned CLK_FREQUENCY = 1500;                // Core clock frequency (MHz)

    unsigned THREADS_PER_BLOCK = 256; // Threads per block (launch config)
    unsigned BLOCKS_PER_SM = 8;       // Blocks per SM (launch config)
    unsigned THREADS_PER_SM = 2048;   // Threads per SM (launch config)
    unsigned BLOCKS_NUM = 640;        // Total blocks launched
    unsigned TOTAL_THREADS = 163840;  // Total threads launched
};
GpuConfig config;
// Parses short flags like --sm 80 into a GpuConfig object
inline void parseGpuConfigArgs(int argc, char *argv[])
{

    auto to_uint = [](const std::string &s)
    { return static_cast<unsigned>(std::stoul(s)); };
    auto to_size = [](const std::string &s)
    { return static_cast<size_t>(std::stoull(s)); };

    for (int i = 1; i < argc - 1; ++i)
    {
        std::string flag(argv[i]);
        std::string val(argv[i + 1]);

        if (flag == "--sm")
            config.SM_NUMBER = to_uint(val);
        else if (flag == "--ws")
            config.WARP_SIZE = to_uint(val);
        else if (flag == "--mtsm")
            config.MAX_THREADS_PER_SM = to_uint(val);
        else if (flag == "--msmem")
            config.MAX_SHARED_MEM_SIZE = to_uint(val);
        else if (flag == "--mrpsm")
            config.MAX_REG_PER_SM = to_uint(val);

        else if (flag == "--mtbs")
            config.MAX_THREAD_BLOCK_SIZE = to_uint(val);
        else if (flag == "--msmemb")
            config.MAX_SHARED_MEM_SIZE_PER_BLOCK = to_uint(val);
        else if (flag == "--mrpb")
            config.MAX_REG_PER_BLOCK = to_uint(val);

        else if (flag == "--l1")
            config.L1_SIZE = to_size(val);
        else if (flag == "--l2")
            config.L2_SIZE = to_size(val);

        else if (flag == "--mem")
            config.MEM_SIZE = to_size(val);
        else if (flag == "--memclk")
            config.MEM_CLK_FREQUENCY = to_uint(val);
        else if (flag == "--membw")
            config.MEM_BITWIDTH = to_uint(val);
        else if (flag == "--clk")
            config.CLK_FREQUENCY = to_uint(val);
        else if (flag == "--tpb")
            config.THREADS_PER_BLOCK = to_uint(val);
        else if (flag == "--bpsm")
            config.BLOCKS_PER_SM = to_uint(val);
        else if (flag == "--tpsm")
            config.THREADS_PER_SM = to_uint(val);
        else if (flag == "--blocks")
            config.BLOCKS_NUM = to_uint(val);
        else if (flag == "--total")
            config.TOTAL_THREADS = to_uint(val);

        else
            continue;

        ++i;
    }
    config.MAX_WARPS_PER_SM = config.MAX_THREADS_PER_SM / config.WARP_SIZE;
    config.MEM_CLK_FREQUENCY = config.MEM_CLK_FREQUENCY * 1e-3f;
    config.BLOCKS_PER_SM = config.MAX_THREADS_PER_SM / config.THREADS_PER_BLOCK;
    config.THREADS_PER_SM = config.BLOCKS_PER_SM * config.THREADS_PER_BLOCK;
    config.TOTAL_THREADS = config.THREADS_PER_BLOCK * config.BLOCKS_NUM;
}

// Optional: for debugging or confirmation
inline void printGpuConfig(const GpuConfig &c = config)
{
    std::cout << "SM_NUMBER: " << c.SM_NUMBER << "\n"
              << "WARP_SIZE: " << c.WARP_SIZE << "\n"
              << "MAX_THREADS_PER_SM: " << c.MAX_THREADS_PER_SM << "\n"
              << "MAX_SHARED_MEM_SIZE: " << c.MAX_SHARED_MEM_SIZE << "\n"
              << "MAX_WARPS_PER_SM: " << c.MAX_WARPS_PER_SM << "\n"
              << "MAX_REG_PER_SM: " << c.MAX_REG_PER_SM << "\n"
              << "MAX_THREAD_BLOCK_SIZE: " << c.MAX_THREAD_BLOCK_SIZE << "\n"
              << "MAX_SHARED_MEM_SIZE_PER_BLOCK: " << c.MAX_SHARED_MEM_SIZE_PER_BLOCK << "\n"
              << "MAX_REG_PER_BLOCK: " << c.MAX_REG_PER_BLOCK << "\n"
              << "L1_SIZE: " << c.L1_SIZE << "\n"
              << "L2_SIZE: " << c.L2_SIZE << "\n"
              << "MEM_SIZE: " << c.MEM_SIZE << "\n"
              << "MEM_CLK_FREQUENCY: " << c.MEM_CLK_FREQUENCY << "\n"
              << "MEM_BITWIDTH: " << c.MEM_BITWIDTH << "\n"
              << "CLK_FREQUENCY: " << c.CLK_FREQUENCY << "\n"
              << "THREADS_PER_BLOCK: " << c.THREADS_PER_BLOCK << "\n"
              << "BLOCKS_PER_SM: " << c.BLOCKS_PER_SM << "\n"
              << "THREADS_PER_SM: " << c.THREADS_PER_SM << "\n"
              << "BLOCKS_NUM: " << c.BLOCKS_NUM << "\n"
              << "TOTAL_THREADS: " << c.TOTAL_THREADS << "\n";
}

// GPU error check
#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
                line);
        if (abort)
            exit(code);
    }
}

cudaDeviceProp deviceProp;

unsigned intilizeDeviceProp(unsigned deviceID, int argc, char *argv[])
{
#ifdef TUNER

#pragma message("TUNER")
    cudaSetDevice(deviceID);
    cudaGetDeviceProperties(&deviceProp, deviceID);

    // core stats

    config.SM_NUMBER = deviceProp.multiProcessorCount;
    config.MAX_THREADS_PER_SM = deviceProp.maxThreadsPerMultiProcessor;
    config.MAX_SHARED_MEM_SIZE = deviceProp.sharedMemPerMultiprocessor;
    config.WARP_SIZE = deviceProp.warpSize;
    config.MAX_WARPS_PER_SM =
        deviceProp.maxThreadsPerMultiProcessor / deviceProp.warpSize;
    config.MAX_REG_PER_SM = deviceProp.regsPerMultiprocessor;

    // threadblock stats
    config.MAX_THREAD_BLOCK_SIZE = deviceProp.maxThreadsPerBlock;
    config.MAX_SHARED_MEM_SIZE_PER_BLOCK = deviceProp.sharedMemPerBlock;
    config.MAX_REG_PER_BLOCK = deviceProp.regsPerBlock;

    // launched thread blocks to ensure GPU is fully occupied as much as possible
    config.THREADS_PER_BLOCK = deviceProp.maxThreadsPerBlock;
    config.BLOCKS_PER_SM =
        deviceProp.maxThreadsPerMultiProcessor / deviceProp.maxThreadsPerBlock;
    config.THREADS_PER_SM = config.BLOCKS_PER_SM * config.THREADS_PER_BLOCK;
    config.BLOCKS_NUM = config.BLOCKS_PER_SM * config.SM_NUMBER;
    config.TOTAL_THREADS = config.THREADS_PER_BLOCK * config.BLOCKS_NUM;

    // L2 cache
    config.L2_SIZE = deviceProp.l2CacheSize;

    // memory
    config.MEM_SIZE = deviceProp.totalGlobalMem;
    config.MEM_CLK_FREQUENCY = deviceProp.memoryClockRate * 1e-3f;
    config.MEM_BITWIDTH = deviceProp.memoryBusWidth;
#else
    parseGpuConfigArgs(argc, argv);

#endif

    printGpuConfig();

    return 1;
}
#endif // GPU_CONFIG_H
