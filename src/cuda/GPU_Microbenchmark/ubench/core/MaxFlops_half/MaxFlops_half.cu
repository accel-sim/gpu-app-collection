#include "MaxFlops_half.h"

int main(int argc, char *argv[])
{

  intilizeDeviceProp(0, argc, argv);

  fpu16_max_flops();

  return 1;
}
