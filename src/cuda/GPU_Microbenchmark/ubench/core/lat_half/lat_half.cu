#include "lat_half.h"

int main(int argc, char *argv[])
{

  initializeDeviceProp(0, argc, argv);

  fpu16_latency();

  return 0;
}
