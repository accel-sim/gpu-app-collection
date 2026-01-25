#include "lat_gmma_common.h"

void run_f32tf32tf32tf32_64x128x8_test() {
    TEST_MMA_CONFIG(tfloat32_t, tfloat32_t, float, 64, 128, 8, "MMA_64x128x8_F32TF32TF32_SS_TN");
}
