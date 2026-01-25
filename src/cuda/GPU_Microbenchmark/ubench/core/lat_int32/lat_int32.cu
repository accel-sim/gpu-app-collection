#include "lat_int32.h"

int main(int argc, char *argv[])
{

  initializeDeviceProp(0, argc, argv);

  int32_latency();

  return 0;
}
