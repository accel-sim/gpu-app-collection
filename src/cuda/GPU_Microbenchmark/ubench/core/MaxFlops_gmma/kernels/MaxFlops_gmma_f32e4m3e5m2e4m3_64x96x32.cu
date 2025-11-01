#include "MaxFlops_gmma_common.h"

void run_f32e4m3e5m2e4m3_64x96x32_test() {
    TEST_MMA_CONFIG(float_e4m3_t, float_e5m2_t, float, 64, 96, 32, "MMA_64x96x32_F32E4M3E5M2_SS_TN");
}
