#include "tensor_bw_half.h"

int main(int argc, char *argv[])
{

  intilizeDeviceProp(0, argc, argv);

  if (deviceProp.major < 6) // tesnore unit was added since Volta
    return 1;

  std::cout << "FP16 operand, FP32 accumalte:\n";
  tensor_max_flops<half, float>();

  std::cout << "\nFP16 operand, FP16 accumalte:\n";
  tensor_max_flops<half, half>();

  // tensor_max_flops<char,int>();

  return 1;
}
