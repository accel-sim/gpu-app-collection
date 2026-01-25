#include "lat_gmma_common.h"

void run_f16e5m2e4m3_64x64x32_test() {
    TEST_MMA_CONFIG(float_e5m2_t, float_e4m3_t, half_t, 64, 64, 32, "MMA_64x64x32_F16E5M2E4M3_SS_TN");
}
