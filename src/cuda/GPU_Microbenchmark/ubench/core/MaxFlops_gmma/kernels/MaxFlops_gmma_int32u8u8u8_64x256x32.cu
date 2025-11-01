#include "MaxFlops_gmma_common.h"

void run_int32u8u8u8_64x256x32_test() {
    TEST_MMA_CONFIG(uint8_t, uint8_t, int32_t, 64, 256, 32, "MMA_64x256x32_S32U8U8_SS_TN");
}
