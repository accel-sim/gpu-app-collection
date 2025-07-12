#include "sfu_lat_fsqrt.h"

int main(int argc, char* argv[]) {

 
  intilizeDeviceProp(0,argc,argv);  printGpuConfig();

  sfu_latency();

  return 1;
}
