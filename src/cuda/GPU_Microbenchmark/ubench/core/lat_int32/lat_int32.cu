#include "lat_int32.h"

int main(int argc, char* argv[]) {

 
  intilizeDeviceProp(0,argc,argv);  printGpuConfig();

  int32_latency();

  return 1;
}
