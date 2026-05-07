#include "lat_gmma_common.h"

void run_int32s8s8s8_64x256x32_test() {
    TEST_MMA_CONFIG(int8_t, int8_t, int32_t, 64, 256, 32, "MMA_64x256x32_S32S8S8_SS_TN");
}
