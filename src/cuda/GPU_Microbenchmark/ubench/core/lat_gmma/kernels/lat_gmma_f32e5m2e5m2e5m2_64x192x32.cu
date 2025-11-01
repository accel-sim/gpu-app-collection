#include "lat_gmma_common.h"

void run_f32e5m2e5m2e5m2_64x192x32_test() {
    TEST_MMA_CONFIG(float_e5m2_t, float_e5m2_t, float, 64, 192, 32, "MMA_64x192x32_F32E5M2E5M2_SS_TN");
}
