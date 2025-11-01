#include "MaxFlops_gmma_common.h"

void run_f32tf32tf32tf32_64x32x8_test() {
    TEST_MMA_CONFIG(tfloat32_t, tfloat32_t, float, 64, 32, 8, "MMA_64x32x8_F32TF32TF32_SS_TN");
}
