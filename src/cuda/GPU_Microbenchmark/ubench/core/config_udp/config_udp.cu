#include <iostream>
using namespace std;

#include "../../../hw_def/hw_def.h"

int main(int argc, char *argv[])
{

  intilizeDeviceProp(0, argc, argv);

  if (ACCEL_SIM_MODE)
  {

    /* we cannot meaure uniform instrcution for now as they only exist at
      SASS level not at PTX nor CUDA level, so assume constant latency and BW
      for now

      dedicated uniform unit was added since Turing SM 7.0
      */
    if (deviceProp.major >= 7)
    {
      // assume UDP unit is on spec unit 4
      std::cout << "-specialized_unit_4 1," << WARP_SCHEDS_PER_SM << ",4,"
                << WARP_SCHEDS_PER_SM << "," << WARP_SCHEDS_PER_SM << ",UDP"
                << std::endl;

      std::cout << "-trace_opcode_latency_initiation_spec_op_4 4,1"
                << std::endl;
    }
  }

  return 1;
}
