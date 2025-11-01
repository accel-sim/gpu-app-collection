#include "lat_gmma_common.h"

void run_f32f16f16_64x64x16_test() {
    TEST_MMA_CONFIG(half_t, half_t, float, 64, 64, 16, "MMA_64x64x16_F32F16F16_SS");
}
