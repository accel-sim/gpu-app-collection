#ifndef GPU_CONFIG_H
#define GPU_CONFIG_H

#include <string>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <cstring>


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

    unsigned FBP_COUNT = 0;           // Frame Buffer Partitions
    unsigned L2_BANKS = 0;            // L2 Cache Banks (LTCs)
};
inline GpuConfig config;
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
              << "TOTAL_THREADS: " << c.TOTAL_THREADS << "\n"
              << "FBP_COUNT: " << c.FBP_COUNT << "\n"
              << "L2_BANKS: " << c.L2_BANKS << "\n";
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

inline cudaDeviceProp deviceProp;

// NVIDIA RM API defines
#define NV_IOCTL_MAGIC 'F'
#define NV_ESC_RM_ALLOC 0x2b
#define NV_ESC_RM_CONTROL 0x2a
#define NV_ESC_RM_FREE 0x29
#define NV01_ROOT_CLIENT 0x00000041
#define NV01_DEVICE_0 0x00000080
#define NV20_SUBDEVICE_0 0x00002080
#define NV2080_CTRL_CMD_GR_GET_INFO 0x20801201

// https://github.com/NVIDIA/open-gpu-kernel-modules/blob/580.95.05/src/common/sdk/nvidia/inc/ctrl/ctrl0080/ctrl0080gr.h#L142
#define NV2080_CTRL_GR_INFO_INDEX_LITTER_NUM_FBPS 0x00000015
#define NV2080_CTRL_GR_INFO_INDEX_LITTER_NUM_LTCS 0x00000025

typedef uint32_t NvHandle;
typedef uint32_t NvV32;
typedef uint64_t NvP64;

// Query single GR info index using NVIDIA RM API
inline unsigned queryGrInfo(uint32_t info_index)
{
    struct NVOS21_PARAMETERS { NvHandle hRoot, hObjectParent, hObjectNew; NvV32 hClass; NvP64 pAllocParms; uint32_t paramsSize, status; };
    struct NVOS54_PARAMETERS { NvHandle hClient, hObject; NvV32 cmd, flags; NvP64 params; uint32_t paramsSize, status; };
    struct NVOS00_PARAMETERS { NvHandle hRoot, hObjectParent, hObjectOld; uint32_t status; };
    struct NV0080_ALLOC_PARAMETERS { uint32_t deviceId; NvHandle hClientShare, hTargetClient, hTargetDevice; NvV32 flags; uint32_t _pad0; uint64_t vaSpaceSize, vaStartInternal, vaLimitInternal; NvV32 vaMode; uint32_t _pad1; };
    struct NV2080_ALLOC_PARAMETERS { uint32_t subDeviceId; };
    struct NVXXXX_CTRL_XXX_INFO { uint32_t index, data; };
    struct NV0080_CTRL_GR_ROUTE_INFO { uint32_t flags, _pad; uint64_t route; };
    struct NV2080_CTRL_GR_GET_INFO_PARAMS { uint32_t grInfoListSize, _pad; NvP64 grInfoList; NV0080_CTRL_GR_ROUTE_INFO grRouteInfo; };

    int ctl_fd = open("/dev/nvidiactl", O_RDWR);
    if (ctl_fd < 0) {
        fprintf(stderr, "DEBUG GR: Failed to open /dev/nvidiactl (errno=%d)\n", errno);
        return 0;
    }

    auto rm_alloc = [&](NvHandle hClient, NvHandle hParent, NvHandle hObject, uint32_t hClass, void *pParams, uint32_t size) {
        NVOS21_PARAMETERS p = {hClient, hParent, hObject, hClass, (NvP64)(uintptr_t)pParams, size, 0};
        bool success = ioctl(ctl_fd, _IOWR(NV_IOCTL_MAGIC, NV_ESC_RM_ALLOC, NVOS21_PARAMETERS), &p) >= 0 && p.status == 0;
        if (!success) fprintf(stderr, "DEBUG GR: rm_alloc failed for class 0x%x, status=0x%x\n", hClass, p.status);
        return success;
    };
    auto rm_control = [&](NvHandle hClient, NvHandle hObject, uint32_t cmd, void *pParams, uint32_t size) {
        NVOS54_PARAMETERS p = {hClient, hObject, cmd, 0, (NvP64)(uintptr_t)pParams, size, 0};
        bool success = ioctl(ctl_fd, _IOWR(NV_IOCTL_MAGIC, NV_ESC_RM_CONTROL, NVOS54_PARAMETERS), &p) >= 0 && p.status == 0;
        if (!success) fprintf(stderr, "DEBUG GR: rm_control failed for cmd 0x%x, status=0x%x\n", cmd, p.status);
        return success;
    };
    auto rm_free = [&](NvHandle hClient, NvHandle hParent, NvHandle hObject) {
        NVOS00_PARAMETERS p = {hClient, hParent, hObject, 0};
        ioctl(ctl_fd, _IOWR(NV_IOCTL_MAGIC, NV_ESC_RM_FREE, NVOS00_PARAMETERS), &p);
    };

    NvHandle hClient = 0xCAFE0001, hDevice = 0xCAFE0002, hSubDevice = 0xCAFE0003;
    NV0080_ALLOC_PARAMETERS devParams = {0};
    NV2080_ALLOC_PARAMETERS subdevParams = {0};
    NVXXXX_CTRL_XXX_INFO infoList[1] = {{info_index, 0}};
    NV2080_CTRL_GR_GET_INFO_PARAMS grParams = {1, 0, (NvP64)(uintptr_t)infoList, {0, 0, 0}};

    unsigned result = 0;
    if (rm_alloc(hClient, hClient, hClient, NV01_ROOT_CLIENT, NULL, 0) &&
        rm_alloc(hClient, hClient, hDevice, NV01_DEVICE_0, &devParams, sizeof(devParams)) &&
        rm_alloc(hClient, hDevice, hSubDevice, NV20_SUBDEVICE_0, &subdevParams, sizeof(subdevParams)) &&
        rm_control(hClient, hSubDevice, NV2080_CTRL_CMD_GR_GET_INFO, &grParams, sizeof(grParams))) {
        result = infoList[0].data;
        fprintf(stderr, "DEBUG GR: Successfully queried index 0x%x = %u\n", info_index, result);
    } else {
        fprintf(stderr, "DEBUG GR: Query sequence failed for index 0x%x\n", info_index);
    }

    rm_free(hClient, hDevice, hSubDevice);
    rm_free(hClient, hClient, hDevice);
    rm_free(hClient, hClient, hClient);
    close(ctl_fd);
    return result;
}

inline unsigned intilizeDeviceProp(unsigned deviceID, int argc, char *argv[])
{
    // Check if running in GPGPU-Sim by looking for gpgpusim.config
    std::ifstream configFile("gpgpusim.config");
    bool isGpgpuSim = configFile.is_open();

    if (isGpgpuSim) {
        // Parse gpgpusim.config for available parameters
        unsigned n_mem = 32, n_sub_partition = 2;  // defaults
        unsigned l2_nsets = 32, l2_linesize = 128, l2_assoc = 24;  // defaults for L2 per bank
        std::string line;
        while (std::getline(configFile, line)) {
            std::istringstream iss(line);
            std::string key;
            if (iss >> key) {
                if (key == "-gpgpu_n_mem") {
                    iss >> n_mem;   // number of memory controllers
                } else if (key == "-gpgpu_n_sub_partition_per_mchannel") {
                    iss >> n_sub_partition; // number of L2 banks per memory controller
                } else if (key == "-gpgpu_cache:dl2") {
                    // Format: X:nsets:linesize:assoc,... where X is any letter
                    std::string cacheConfig;
                    iss >> cacheConfig;
                    // Parse X:nsets:linesize:assoc using sscanf, skip first char
                    char dummy;
                    sscanf(cacheConfig.c_str(), "%c:%u:%u:%u", &dummy, &l2_nsets, &l2_linesize, &l2_assoc);
                }
            }
        }
        configFile.close();

        // Use struct default values (already initialized in GpuConfig)
        // Override FBP_COUNT and L2_BANKS from gpgpusim.config
        config.FBP_COUNT = n_mem;
        config.L2_BANKS = n_mem * n_sub_partition;
        // L2_SIZE = (nsets * linesize * assoc) per bank * banks_per_controller * num_controllers
        size_t l2_size_per_bank = (size_t)l2_nsets * l2_linesize * l2_assoc;
        config.L2_SIZE = l2_size_per_bank * n_sub_partition * n_mem;
    } else {
        // Running on real hardware - query device properties
        cudaSetDevice(deviceID);
        cudaGetDeviceProperties(&deviceProp, deviceID);

        int clockRateKHz;
        cudaDeviceGetAttribute(&clockRateKHz, cudaDevAttrClockRate, deviceID);

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
        config.CLK_FREQUENCY = clockRateKHz * 1e-3f;

        // Get FBP_COUNT and L2_BANKS from NVIDIA RM API
        config.FBP_COUNT = queryGrInfo(NV2080_CTRL_GR_INFO_INDEX_LITTER_NUM_FBPS);
        config.L2_BANKS = queryGrInfo(NV2080_CTRL_GR_INFO_INDEX_LITTER_NUM_LTCS);
    }

    parseGpuConfigArgs(argc, argv);
    printGpuConfig();

    return 1;
}
#endif // GPU_CONFIG_H
