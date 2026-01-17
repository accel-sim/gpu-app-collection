#include "../../l2_cache/l2_lat/l2_lat.h"
#include "mem_lat.h"
#include <iostream>
#include <string.h>

int main(int argc, char *argv[])
{

  float lat_mem = mem_lat(argc, argv);
  float lat2 = 0;

  // Check for --fast flag
  bool fast_mode = false;
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--fast") == 0) {
      fast_mode = true;
      break;
    }
  }

  // Only run l2_hit_lat if not in fast mode
  if (!fast_mode) {
    lat2 = l2_hit_lat(argc, argv);
  }

  std::cout << "\n//Accel_Sim config: \n";
  std::cout << "-dram_latency " << (unsigned)(lat_mem - lat2) << std::endl;

  return 0;
}
