#include <iostream>
using namespace std;

#include "../../../hw_def/hw_def.h"
// #define CLK_FREQUENCY 1665

int main(int argc, char *argv[])
{

  initializeDeviceProp(0, argc, argv);

  printf("Device Name = %s\n", deviceProp.name);
  printf("GPU Max Clock rate = %.0f MHz \n", config.CLK_FREQUENCY * 1e-3f);
  // printf("GPU Base Clock rate = %d MHz \n", CLK_FREQUENCY);
  printf("SM Count = %d\n", config.SM_NUMBER);
  printf("CUDA version number = %d.%d\n", deviceProp.major, deviceProp.minor);

  if (ACCEL_SIM_MODE)
  {

    std::cout << "\n//Accel_Sim config: \n";

    float mem_freq_MHZ = (config.MEM_CLK_FREQUENCY * 1e-3f * 2) /
                         dram_model_freq_ratio[DRAM_MODEL];
    std::cout << "-gpgpu_compute_capability_major " << deviceProp.major
              << std::endl;
    std::cout << "-gpgpu_compute_capability_minor " << deviceProp.minor
              << std::endl;
    std::cout << "-gpgpu_n_clusters " << config.SM_NUMBER
              << std::endl;
    std::cout << "-gpgpu_n_cores_per_cluster 1" << std::endl;
    std::cout << "-gpgpu_clock_domains " << config.CLK_FREQUENCY << ":"
              << config.CLK_FREQUENCY << ":" << config.CLK_FREQUENCY << ":" << mem_freq_MHZ
              << std::endl;
  }

  return 0;
}
