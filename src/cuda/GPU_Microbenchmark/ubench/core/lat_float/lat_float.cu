#include "lat_float.h"

int main(int argc, char *argv[])
{

  initializeDeviceProp(0, argc, argv);

  fpu_latency();

  return 0;
}
