#include "MaxFlops_gmma_common.h"

void run_f16e5m2e5m2_64x32x32_test() {
    TEST_MMA_CONFIG(float_e5m2_t, float_e5m2_t, half_t, 64, 32, 32, "MMA_64x32x32_F16E5M2E5M2_SS_TN");
}
