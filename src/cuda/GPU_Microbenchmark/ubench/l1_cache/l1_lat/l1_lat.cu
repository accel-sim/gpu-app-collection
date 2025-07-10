#include "l1_lat.h"

int main(int argc, char* argv[]) {

 


  float lat = l1_lat(argc,argv);


    std::cout << "\n//Accel_Sim config: \n";
    std::cout << "-gpgpu_l1_latency " << (unsigned)lat << std::endl;
  

  return 1;
}
