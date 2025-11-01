#include "lat_gmma_common.h"

void run_f16e4m3e5m2_64x128x32_test() {
    TEST_MMA_CONFIG(float_e4m3_t, float_e5m2_t, half_t, 64, 128, 32, "MMA_64x128x32_F16E4M3E5M2_SS_TN");
}
