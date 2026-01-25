#include "MaxFlops_gmma_common.h"

void run_f32e5m2e4m3e5m2_64x8x32_test() {
    TEST_MMA_CONFIG(float_e5m2_t, float_e4m3_t, float, 64, 8, 32, "MMA_64x8x32_F32E5M2E4M3_SS_TN");
}
