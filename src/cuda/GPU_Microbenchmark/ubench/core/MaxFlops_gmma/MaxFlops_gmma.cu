#include <cuda.h>
#include "MaxFlops_gmma.h"
#include "../../../hw_def/hw_def.h"

int main(int argc, char *argv[])
{
  initializeDeviceProp(0, argc, argv);

  // Run comprehensive sweep over all valid MMA operations
  run_all_wgmma_maxflops_tests();

  return 0;
}
