#include "MaxFlops_double.h"

int main(int argc, char *argv[])
{

  intilizeDeviceProp(0, argc, argv);

  dpu_max_flops();

  return 1;
}
