#include <cuda.h>
#include "lat_gmma.h"
#include "../../../hw_def/hw_def.h"

int main(int argc, char *argv[])
{
  intilizeDeviceProp(0, argc, argv);

  // Run comprehensive sweep over all valid MMA operations
  run_all_wgmma_latency_tests();

  return 0;
}
