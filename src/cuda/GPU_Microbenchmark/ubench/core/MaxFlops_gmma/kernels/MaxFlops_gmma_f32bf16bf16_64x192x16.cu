#include "MaxFlops_gmma_common.h"

void run_f32bf16bf16_64x192x16_test() {
    TEST_MMA_CONFIG(bfloat16_t, bfloat16_t, float, 64, 192, 16, "MMA_64x192x16_F32BF16BF16_SS");
}
