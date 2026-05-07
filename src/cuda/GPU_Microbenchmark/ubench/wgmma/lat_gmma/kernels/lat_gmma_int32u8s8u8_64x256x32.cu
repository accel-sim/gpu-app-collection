#include "lat_gmma_common.h"

void run_int32u8s8u8_64x256x32_test() {
    TEST_MMA_CONFIG(uint8_t, int8_t, int32_t, 64, 256, 32, "MMA_64x256x32_S32U8S8_SS_TN");
}
