#include "../../l2_cache/l2_lat/l2_lat.h"
#include "mem_lat.h"
#include <iostream>

int main(int argc, char *argv[])
{

  float lat_mem = mem_lat(argc, argv);
  float lat2 = 0;
#ifdef TUNER
  lat2 = l2_hit_lat(argc, argv);
#endif
  std::cout << "\n//Accel_Sim config: \n";
  std::cout << "-dram_latency " << (unsigned)(lat_mem - lat2) << std::endl;

  return 1;
}
