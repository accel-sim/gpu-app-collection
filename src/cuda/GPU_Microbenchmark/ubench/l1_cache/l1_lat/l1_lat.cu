#include "l1_lat.h"

int main() {


  float lat = l1_lat();


    std::cout << "\n//Accel_Sim config: \n";
    std::cout << "-gpgpu_l1_latency " << (unsigned)lat << std::endl;
  

  return 1;
}
