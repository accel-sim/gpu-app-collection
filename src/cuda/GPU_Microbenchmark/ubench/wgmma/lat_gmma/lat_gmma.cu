#include <cuda.h>
#include <stdio.h>
#include "lat_gmma.h"
#include "../../../hw_def/hw_def.h"

FILE* g_lat_gmma_outfile = NULL;

int main(int argc, char *argv[])
{
  initializeDeviceProp(0, argc, argv);

  g_lat_gmma_outfile = fopen("lat_gmma_results.csv", "w");
  if (g_lat_gmma_outfile)
    fprintf(g_lat_gmma_outfile, "config,cycles_per_mma\n");

  run_all_wgmma_latency_tests();

  if (g_lat_gmma_outfile)
    fclose(g_lat_gmma_outfile);

  return 0;
}
