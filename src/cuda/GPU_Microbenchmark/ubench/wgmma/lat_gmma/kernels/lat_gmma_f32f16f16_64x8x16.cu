#include "lat_gmma_common.h"

void run_f32f16f16_64x8x16_test() {
    TEST_MMA_CONFIG(half_t, half_t, float, 64, 8, 16, "MMA_64x8x16_F32F16F16_SS");
}
