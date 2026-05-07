#include <cuda.h>
#include "lat_gmma_common.h"

int main()
{
  // A simple test kernel to check if the library is working
  try {
    using TileShape = decltype(make_shape(cute::Int<64>{}, cute::Int<256>{}, cute::Int<16>{}));
    // Repeat once for the test
    using RepeatTimes = cute::Int<1>;
    float lat = run_wgmma_latency_test_typed<half_t, half_t, float, TileShape, RepeatTimes>();
    printf("MMA_64x256x16_F32F16F16_SS: %6.2f cycles\n", lat);
  } catch (...) {
    printf("MMA_64x256x16_F32F16F16_SS: FAILED\n");
  }
  return 0;
}
