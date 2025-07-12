#include "lat_double.h"

int main(int argc, char* argv[]) {

 
  intilizeDeviceProp(0,argc,argv);  printGpuConfig();

  dpu_latency();

  return 1;
}
