#include <cuda.h>
#include "MaxFlops_gmma_common.h"

void run_f32bf16bf16_64x256x16_test();

int main()
{
  // Simple a test kernel to check if the library is working
  try {
    using TileShape = decltype(make_shape(cute::Int<64>{}, cute::Int<256>{}, cute::Int<16>{}));
    // Repeat once for the test
    using RepeatTimes = cute::Int<1>;
    float num_flop_per_warpgroup = 2 * 64 * 256 * 16;
    float warp_inst_per_cycle = run_wgmma_maxflops_test_typed<half_t, half_t, float, TileShape, RepeatTimes>();
    const float warps_to_warpgroup = 0.25;
    float flop_per_cycle_per_warpgroup = num_flop_per_warpgroup * warp_inst_per_cycle * warps_to_warpgroup;
    printf("MMA_64x256x16_F32F16F16_SS: %6.4f warp instructions/cycle\n", warp_inst_per_cycle);
    printf("MMA_64x256x16_F32F16F16_SS: %6.4f flop/warpgroup inst/cycle\n", flop_per_cycle_per_warpgroup);
  } catch (...) {
    printf("MMA_64x256x16_F32F16F16_SS: FAILED\n");
  }
  return 0;
}
