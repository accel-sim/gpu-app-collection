/*
Some of the code is adopted from device query benchmark
from CUDA SDK
*/

#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <iostream>
#include <memory>
#include <string>

static int getDeviceAttributeOrZero(cudaDeviceAttr attr, int device_id) {
  int value = 0;
  if (cudaDeviceGetAttribute(&value, attr, device_id) != cudaSuccess) {
    return 0;
  }
  return value;
}

int main(int argc, char **argv) {
  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

  if (error_id != cudaSuccess) {
    printf("cudaGetDeviceCount returned %d\n-> %s\n",
           static_cast<int>(error_id), cudaGetErrorString(error_id));
    printf("Result = FAIL\n");
    exit(EXIT_FAILURE);
  }

  // This function call returns 0 if there are no CUDA capable devices.
  if (deviceCount == 0) {
    printf("There are no available device(s) that support CUDA\n");
  }

  int dev, driverVersion = 0, runtimeVersion = 0;

  for (dev = 0; dev < deviceCount; ++dev) {
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    int smClockKHz = getDeviceAttributeOrZero(cudaDevAttrClockRate, dev);
    int memoryClockKHz =
        getDeviceAttributeOrZero(cudaDevAttrMemoryClockRate, dev);
    int memoryBusWidthBits =
        getDeviceAttributeOrZero(cudaDevAttrGlobalMemoryBusWidth, dev);

    // device
    printf("  Device : \"%s\"\n\n", deviceProp.name);
    printf("  CUDA version number                         : %d.%d\n",
           deviceProp.major, deviceProp.minor);

    // core
    printf("  GPU Max Clock rate                             : %.0f MHz \n",
           smClockKHz * 1e-3f);
    printf("  Multiprocessors Count                       : %d\n",
           deviceProp.multiProcessorCount);
    printf("  Maximum number of threads per multiprocessor: %d\n",
           deviceProp.maxThreadsPerMultiProcessor);
    printf("  CUDA Cores per multiprocessor               : %d \n",
           _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor));
    printf("  Registers per multiprocessor                : %d\n",
           deviceProp.regsPerMultiprocessor);
    printf("  Shared memory per multiprocessor            : %lu bytes\n",
           deviceProp.sharedMemPerMultiprocessor);
    printf("  Warp size                                   : %d\n",
           deviceProp.warpSize);

    // threadblock config
    printf("  Maximum number of threads per block         : %d\n",
           deviceProp.maxThreadsPerBlock);
    printf("  Shared memory per block                     : %lu bytes\n",
           deviceProp.sharedMemPerBlock);
    printf("  Registers per block                         : %d\n",
           deviceProp.regsPerBlock);

    // L1 cache
    printf("  globalL1CacheSupported                      : %d\n",
           deviceProp.globalL1CacheSupported);
    printf("  localL1CacheSupported                       : %d\n",
           deviceProp.localL1CacheSupported);

    // L2 cache
    if (deviceProp.l2CacheSize) {
      printf("  L2 Cache Size                             : %.0f MB\n",
             static_cast<float>(deviceProp.l2CacheSize / 1048576.0f));
    }

    // memory
    char msg[256];
    snprintf(msg, sizeof(msg),
             "  Global memory size                        : %.0f GB\n",
             static_cast<float>(deviceProp.totalGlobalMem / 1073741824.0f));
    printf("%s", msg);
    printf("  Memory Clock rate                           : %.0f Mhz\n",
           memoryClockKHz * 1e-3f);
    printf("  Memory Bus Width                            : %d bit\n",
           memoryBusWidthBits > 0 ? memoryBusWidthBits
                                  : deviceProp.memoryBusWidth);

    printf(" ////////////////////////// \n");
  }
}
