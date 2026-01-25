#include "lat_gmma_common.h"

void run_f16e4m3e4m3_64x128x32_test() {
    TEST_MMA_CONFIG(float_e4m3_t, float_e4m3_t, half_t, 64, 128, 32, "MMA_64x128x32_F16E4M3E4M3_SS_TN");
}
