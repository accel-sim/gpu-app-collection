#include "lat_gmma_common.h"

void run_int32s8u8s8_64x32x32_test() {
    TEST_MMA_CONFIG(int8_t, uint8_t, int32_t, 64, 32, 32, "MMA_64x32x32_S32S8U8_SS_TN");
}
