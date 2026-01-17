#include "../../l1_cache/l1_lat/l1_lat.h"
#include "l2_lat.h"
#include <string.h>

int main(int argc, char *argv[])
{

  float lat2 = l2_hit_lat(argc, argv);
  float lat1 = 0;

  // Check for --fast flag
  bool fast_mode = false;
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--fast") == 0) {
      fast_mode = true;
      break;
    }
  }

  // Only run l1_lat if not in fast mode
  if (!fast_mode) {
    lat1 = l1_lat(argc, argv);
  }

  std::cout << "\n//Accel_Sim config: \n";
  std::cout << "-gpgpu_l2_rop_latency " << (unsigned)(lat2 - lat1)
            << std::endl;

  return 0;
}
