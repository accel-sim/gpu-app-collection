#include "MaxFlops_float.h"

int main(int argc, char *argv[])
{

  intilizeDeviceProp(0, argc, argv);

  fpu_max_flops();

  return 1;
}
