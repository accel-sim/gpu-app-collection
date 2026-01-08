#include "lat_double.h"

int main(int argc, char *argv[])
{

  initializeDeviceProp(0, argc, argv);

  dpu_latency();

  return 0;
}
