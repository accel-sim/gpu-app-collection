#include "lat_float.h"

int main(int argc, char *argv[])
{

  intilizeDeviceProp(0, argc, argv);

  fpu_latency();

  return 1;
}
