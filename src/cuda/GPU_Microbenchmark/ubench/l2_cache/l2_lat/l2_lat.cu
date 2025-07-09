#include "../../l1_cache/l1_lat/l1_lat.h"
#include "l2_lat.h"

int main() {


  float lat2 = l2_hit_lat();


    float lat1 = l1_lat();

    std::cout << "\n//Accel_Sim config: \n";
    std::cout << "-gpgpu_l2_rop_latency " << (unsigned)(lat2 - lat1)
              << std::endl;
  

  return 1;
}
