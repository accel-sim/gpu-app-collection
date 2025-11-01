#include "lat_gmma_common.h"

void run_int32u8u8u8_64x64x32_test() {
    TEST_MMA_CONFIG(uint8_t, uint8_t, int32_t, 64, 64, 32, "MMA_64x64x32_S32U8U8_SS_TN");
}
