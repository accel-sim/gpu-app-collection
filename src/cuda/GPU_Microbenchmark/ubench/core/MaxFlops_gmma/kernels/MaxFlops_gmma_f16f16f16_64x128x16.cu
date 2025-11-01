#include "MaxFlops_gmma_common.h"

void run_f16f16f16_64x128x16_test() {
    TEST_MMA_CONFIG(half_t, half_t, half_t, 64, 128, 16, "MMA_64x128x16_F16F16F16_SS");
}
