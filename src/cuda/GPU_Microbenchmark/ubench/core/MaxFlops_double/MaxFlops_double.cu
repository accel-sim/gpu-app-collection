#include "MaxFlops_double.h"

int main(int argc, char* argv[]) {

 
  intilizeDeviceProp(0,argc,argv);  printGpuConfig();


  dpu_max_flops();

  return 1;
}
