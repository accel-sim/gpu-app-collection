#include "MaxFlops_gmma_common.h"

void run_f32e4m3e4m3e4m3_64x128x32_test() {
    TEST_MMA_CONFIG(float_e4m3_t, float_e4m3_t, float, 64, 128, 32, "MMA_64x128x32_F32E4M3E4M3_SS_TN");
}
