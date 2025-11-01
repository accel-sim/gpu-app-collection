/***************************************************************************************************
 * GMMA Latency Microbenchmark for SM90 (Hopper Architecture)
 *
 * This file provides a comprehensive sweep over all valid WGMMA (Warpgroup Matrix Multiply-
 * Accumulate) operations supported by the NVIDIA Hopper architecture (SM90).
 *
 * The sweep covers:
 *
 * 1. F32 Accumulator (ElementC = float):
 *    - FP16 x FP16 -> FP32
 *    - BF16 x BF16 -> FP32
 *    - TF32 x TF32 -> FP32
 *    - E4M3 x E4M3 -> FP32
 *    - E4M3 x E5M2 -> FP32
 *    - E5M2 x E4M3 -> FP32
 *    - E5M2 x E5M2 -> FP32
 *
 * 2. F16 Accumulator (ElementC = half_t):
 *    - FP16 x FP16 -> FP16
 *    - E4M3 x E4M3 -> FP16
 *    - E4M3 x E5M2 -> FP16
 *    - E5M2 x E4M3 -> FP16
 *    - E5M2 x E5M2 -> FP16
 *
 * 3. INT32 Accumulator (ElementC = int32_t):
 *    - INT8 x INT8 -> INT32
 *    - INT8 x UINT8 -> INT32
 *    - UINT8 x INT8 -> INT32
 *    - UINT8 x UINT8 -> INT32
 *
 * For each data type combination, the sweep tests multiple tile shapes:
 *    - 64x8, 64x16, 64x32, 64x64, 64x96, 64x128, 64x192, 64x256
 *
 * Usage:
 *    Call run_all_wgmma_latency_tests() to run the comprehensive sweep.
 *
 * NOTE: This file has been refactored for parallel compilation. Test implementations
 *       are split across lat_gmma_f32.cu, lat_gmma_f16.cu, and lat_gmma_int32.cu.
 *
 **************************************************************************************************/

#ifndef LAT_GMMA_DEF_H
#define LAT_GMMA_DEF_H

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstdint>


// Function declarations for test suites
// These are defined in separate .cu files for parallel compilation

// F32 accumulator tests - TF32 x TF32 -> F32
void run_f32tf32tf32tf32_64x8x8_test();
void run_f32tf32tf32tf32_64x16x8_test();
void run_f32tf32tf32tf32_64x32x8_test();
void run_f32tf32tf32tf32_64x64x8_test();
void run_f32tf32tf32tf32_64x96x8_test();
void run_f32tf32tf32tf32_64x128x8_test();
void run_f32tf32tf32tf32_64x192x8_test();
void run_f32tf32tf32tf32_64x256x8_test();

// F32 accumulator tests - E4M3 x E4M3 -> F32
void run_f32e4m3e4m3e4m3_64x8x32_test();
void run_f32e4m3e4m3e4m3_64x16x32_test();
void run_f32e4m3e4m3e4m3_64x32x32_test();
void run_f32e4m3e4m3e4m3_64x64x32_test();
void run_f32e4m3e4m3e4m3_64x96x32_test();
void run_f32e4m3e4m3e4m3_64x128x32_test();
void run_f32e4m3e4m3e4m3_64x192x32_test();
void run_f32e4m3e4m3e4m3_64x256x32_test();

// F32 accumulator tests - E4M3 x E5M2 -> F32
void run_f32e4m3e5m2e4m3_64x8x32_test();
void run_f32e4m3e5m2e4m3_64x16x32_test();
void run_f32e4m3e5m2e4m3_64x32x32_test();
void run_f32e4m3e5m2e4m3_64x64x32_test();
void run_f32e4m3e5m2e4m3_64x96x32_test();
void run_f32e4m3e5m2e4m3_64x128x32_test();
void run_f32e4m3e5m2e4m3_64x192x32_test();
void run_f32e4m3e5m2e4m3_64x256x32_test();

// F32 accumulator tests - E5M2 x E4M3 -> F32
void run_f32e5m2e4m3e5m2_64x8x32_test();
void run_f32e5m2e4m3e5m2_64x16x32_test();
void run_f32e5m2e4m3e5m2_64x32x32_test();
void run_f32e5m2e4m3e5m2_64x64x32_test();
void run_f32e5m2e4m3e5m2_64x96x32_test();
void run_f32e5m2e4m3e5m2_64x128x32_test();
void run_f32e5m2e4m3e5m2_64x192x32_test();
void run_f32e5m2e4m3e5m2_64x256x32_test();

// F32 accumulator tests - E5M2 x E5M2 -> F32
void run_f32e5m2e5m2e5m2_64x8x32_test();
void run_f32e5m2e5m2e5m2_64x16x32_test();
void run_f32e5m2e5m2e5m2_64x32x32_test();
void run_f32e5m2e5m2e5m2_64x64x32_test();
void run_f32e5m2e5m2e5m2_64x96x32_test();
void run_f32e5m2e5m2e5m2_64x128x32_test();
void run_f32e5m2e5m2e5m2_64x192x32_test();
void run_f32e5m2e5m2e5m2_64x256x32_test();

// INT32 accumulator tests - INT8 x INT8 -> INT32
void run_int32s8s8s8_64x8x32_test();
void run_int32s8s8s8_64x16x32_test();
void run_int32s8s8s8_64x32x32_test();
void run_int32s8s8s8_64x64x32_test();
void run_int32s8s8s8_64x96x32_test();
void run_int32s8s8s8_64x128x32_test();
void run_int32s8s8s8_64x192x32_test();
void run_int32s8s8s8_64x256x32_test();

// INT32 accumulator tests - INT8 x UINT8 -> INT32
void run_int32s8u8s8_64x8x32_test();
void run_int32s8u8s8_64x16x32_test();
void run_int32s8u8s8_64x32x32_test();
void run_int32s8u8s8_64x64x32_test();
void run_int32s8u8s8_64x96x32_test();
void run_int32s8u8s8_64x128x32_test();
void run_int32s8u8s8_64x192x32_test();
void run_int32s8u8s8_64x256x32_test();

// INT32 accumulator tests - UINT8 x INT8 -> INT32
void run_int32u8s8u8_64x8x32_test();
void run_int32u8s8u8_64x16x32_test();
void run_int32u8s8u8_64x32x32_test();
void run_int32u8s8u8_64x64x32_test();
void run_int32u8s8u8_64x96x32_test();
void run_int32u8s8u8_64x128x32_test();
void run_int32u8s8u8_64x192x32_test();
void run_int32u8s8u8_64x256x32_test();

// INT32 accumulator tests - UINT8 x UINT8 -> INT32
void run_int32u8u8u8_64x8x32_test();
void run_int32u8u8u8_64x16x32_test();
void run_int32u8u8u8_64x32x32_test();
void run_int32u8u8u8_64x64x32_test();
void run_int32u8u8u8_64x96x32_test();
void run_int32u8u8u8_64x128x32_test();
void run_int32u8u8u8_64x192x32_test();
void run_int32u8u8u8_64x256x32_test();

// F16 accumulator tests (defined in lat_gmma_f16.cu)
// F32 accumulator tests - FP16 x FP16 -> F32
void run_f32f16f16_64x8x16_test();
void run_f32f16f16_64x16x16_test();
void run_f32f16f16_64x32x16_test();
void run_f32f16f16_64x64x16_test();
void run_f32f16f16_64x96x16_test();
void run_f32f16f16_64x128x16_test();
void run_f32f16f16_64x192x16_test();
void run_f32f16f16_64x256x16_test();

// F32 accumulator tests - BF16 x BF16 -> F32
void run_f32bf16bf16_64x8x16_test();
void run_f32bf16bf16_64x16x16_test();
void run_f32bf16bf16_64x32x16_test();
void run_f32bf16bf16_64x64x16_test();
void run_f32bf16bf16_64x96x16_test();
void run_f32bf16bf16_64x128x16_test();
void run_f32bf16bf16_64x192x16_test();
void run_f32bf16bf16_64x256x16_test();

// F16 accumulator tests - FP16 x FP16 -> F16
void run_f16f16f16_64x8x16_test();
void run_f16f16f16_64x16x16_test();
void run_f16f16f16_64x32x16_test();
void run_f16f16f16_64x64x16_test();
void run_f16f16f16_64x96x16_test();
void run_f16f16f16_64x128x16_test();
void run_f16f16f16_64x192x16_test();
void run_f16f16f16_64x256x16_test();

// F16 accumulator tests - E4M3 x E4M3 -> F16
void run_f16e4m3e4m3_64x8x32_test();
void run_f16e4m3e4m3_64x16x32_test();
void run_f16e4m3e4m3_64x32x32_test();
void run_f16e4m3e4m3_64x64x32_test();
void run_f16e4m3e4m3_64x96x32_test();
void run_f16e4m3e4m3_64x128x32_test();
void run_f16e4m3e4m3_64x192x32_test();
void run_f16e4m3e4m3_64x256x32_test();

// F16 accumulator tests - E4M3 x E5M2 -> F16
void run_f16e4m3e5m2_64x8x32_test();
void run_f16e4m3e5m2_64x16x32_test();
void run_f16e4m3e5m2_64x32x32_test();
void run_f16e4m3e5m2_64x64x32_test();
void run_f16e4m3e5m2_64x96x32_test();
void run_f16e4m3e5m2_64x128x32_test();
void run_f16e4m3e5m2_64x192x32_test();
void run_f16e4m3e5m2_64x256x32_test();

// F16 accumulator tests - E5M2 x E4M3 -> F16
void run_f16e5m2e4m3_64x8x32_test();
void run_f16e5m2e4m3_64x16x32_test();
void run_f16e5m2e4m3_64x32x32_test();
void run_f16e5m2e4m3_64x64x32_test();
void run_f16e5m2e4m3_64x96x32_test();
void run_f16e5m2e4m3_64x128x32_test();
void run_f16e5m2e4m3_64x192x32_test();
void run_f16e5m2e4m3_64x256x32_test();

// F16 accumulator tests - E5M2 x E5M2 -> F16
void run_f16e5m2e5m2_64x8x32_test();
void run_f16e5m2e5m2_64x16x32_test();
void run_f16e5m2e5m2_64x32x32_test();
void run_f16e5m2e5m2_64x64x32_test();
void run_f16e5m2e5m2_64x96x32_test();
void run_f16e5m2e5m2_64x128x32_test();
void run_f16e5m2e5m2_64x192x32_test();
void run_f16e5m2e5m2_64x256x32_test();

void run_f16accumulator_tests() {
  run_f16f16f16_64x8x16_test();
  run_f16f16f16_64x16x16_test();
  run_f16f16f16_64x32x16_test();
  run_f16f16f16_64x64x16_test();
  run_f16f16f16_64x96x16_test();
  run_f16f16f16_64x128x16_test();
  run_f16f16f16_64x192x16_test();
  run_f16f16f16_64x256x16_test();
  run_f16e4m3e4m3_64x8x32_test();
  run_f16e4m3e4m3_64x16x32_test();
  run_f16e4m3e4m3_64x32x32_test();
  run_f16e4m3e4m3_64x64x32_test();
  run_f16e4m3e4m3_64x96x32_test();
  run_f16e4m3e4m3_64x128x32_test();
  run_f16e4m3e4m3_64x192x32_test();
  run_f16e4m3e4m3_64x256x32_test();
  run_f16e4m3e5m2_64x8x32_test();
  run_f16e4m3e5m2_64x16x32_test();
  run_f16e4m3e5m2_64x32x32_test();
  run_f16e4m3e5m2_64x64x32_test();
  run_f16e4m3e5m2_64x96x32_test();
  run_f16e4m3e5m2_64x128x32_test();
  run_f16e4m3e5m2_64x192x32_test();
  run_f16e4m3e5m2_64x256x32_test();
  run_f16e5m2e4m3_64x8x32_test();
  run_f16e5m2e4m3_64x16x32_test();
  run_f16e5m2e4m3_64x32x32_test();
  run_f16e5m2e4m3_64x64x32_test();
  run_f16e5m2e4m3_64x96x32_test();
  run_f16e5m2e4m3_64x128x32_test();
  run_f16e5m2e4m3_64x192x32_test();
  run_f16e5m2e4m3_64x256x32_test();
  run_f16e5m2e5m2_64x8x32_test();
  run_f16e5m2e5m2_64x16x32_test();
  run_f16e5m2e5m2_64x32x32_test();
  run_f16e5m2e5m2_64x64x32_test();
  run_f16e5m2e5m2_64x96x32_test();
  run_f16e5m2e5m2_64x128x32_test();
  run_f16e5m2e5m2_64x192x32_test();
  run_f16e5m2e5m2_64x256x32_test();
}

void run_f32accumulator_tests() {
  run_f32tf32tf32tf32_64x8x8_test();
  run_f32tf32tf32tf32_64x16x8_test();
  run_f32tf32tf32tf32_64x32x8_test();
  run_f32tf32tf32tf32_64x64x8_test();
  run_f32tf32tf32tf32_64x96x8_test();
  run_f32tf32tf32tf32_64x128x8_test();
  run_f32tf32tf32tf32_64x192x8_test();
  run_f32tf32tf32tf32_64x256x8_test();
  run_f32f16f16_64x8x16_test();
  run_f32f16f16_64x16x16_test();
  run_f32f16f16_64x32x16_test();
  run_f32f16f16_64x64x16_test();
  run_f32f16f16_64x96x16_test();
  run_f32f16f16_64x128x16_test();
  run_f32f16f16_64x192x16_test();
  run_f32f16f16_64x256x16_test();
  run_f32bf16bf16_64x8x16_test();
  run_f32bf16bf16_64x16x16_test();
  run_f32bf16bf16_64x32x16_test();
  run_f32bf16bf16_64x64x16_test();
  run_f32bf16bf16_64x96x16_test();
  run_f32bf16bf16_64x128x16_test();
  run_f32bf16bf16_64x192x16_test();
  run_f32bf16bf16_64x256x16_test();
  run_f32e4m3e4m3e4m3_64x8x32_test();
  run_f32e4m3e4m3e4m3_64x16x32_test();
  run_f32e4m3e4m3e4m3_64x32x32_test();
  run_f32e4m3e4m3e4m3_64x64x32_test();
  run_f32e4m3e4m3e4m3_64x96x32_test();
  run_f32e4m3e4m3e4m3_64x128x32_test();
  run_f32e4m3e4m3e4m3_64x192x32_test();
  run_f32e4m3e4m3e4m3_64x256x32_test();
  run_f32e4m3e5m2e4m3_64x8x32_test();
  run_f32e4m3e5m2e4m3_64x16x32_test();
  run_f32e4m3e5m2e4m3_64x32x32_test();
  run_f32e4m3e5m2e4m3_64x64x32_test();
  run_f32e4m3e5m2e4m3_64x96x32_test();
  run_f32e4m3e5m2e4m3_64x128x32_test();
  run_f32e4m3e5m2e4m3_64x192x32_test();
  run_f32e4m3e5m2e4m3_64x256x32_test();
  run_f32e5m2e4m3e5m2_64x8x32_test();
  run_f32e5m2e4m3e5m2_64x16x32_test();
  run_f32e5m2e4m3e5m2_64x32x32_test();
  run_f32e5m2e4m3e5m2_64x64x32_test();
  run_f32e5m2e4m3e5m2_64x96x32_test();
  run_f32e5m2e4m3e5m2_64x128x32_test();
  run_f32e5m2e4m3e5m2_64x192x32_test();
  run_f32e5m2e4m3e5m2_64x256x32_test();
  run_f32e5m2e5m2e5m2_64x8x32_test();
  run_f32e5m2e5m2e5m2_64x16x32_test();
  run_f32e5m2e5m2e5m2_64x32x32_test();
  run_f32e5m2e5m2e5m2_64x64x32_test();
  run_f32e5m2e5m2e5m2_64x96x32_test();
  run_f32e5m2e5m2e5m2_64x128x32_test();
  run_f32e5m2e5m2e5m2_64x192x32_test();
  run_f32e5m2e5m2e5m2_64x256x32_test();
}

void run_int32accumulator_tests() {
  run_int32s8s8s8_64x8x32_test();
  run_int32s8s8s8_64x16x32_test();
  run_int32s8s8s8_64x32x32_test();
  run_int32s8s8s8_64x64x32_test();
  run_int32s8s8s8_64x96x32_test();
  run_int32s8s8s8_64x128x32_test();
  run_int32s8s8s8_64x192x32_test();
  run_int32s8s8s8_64x256x32_test();
  run_int32s8u8s8_64x8x32_test();
  run_int32s8u8s8_64x16x32_test();
  run_int32s8u8s8_64x32x32_test();
  run_int32s8u8s8_64x64x32_test();
  run_int32s8u8s8_64x96x32_test();
  run_int32s8u8s8_64x128x32_test();
  run_int32s8u8s8_64x192x32_test();
  run_int32s8u8s8_64x256x32_test();
  run_int32u8s8u8_64x8x32_test();
  run_int32u8s8u8_64x16x32_test();
  run_int32u8s8u8_64x32x32_test();
  run_int32u8s8u8_64x64x32_test();
  run_int32u8s8u8_64x96x32_test();
  run_int32u8s8u8_64x128x32_test();
  run_int32u8s8u8_64x192x32_test();
  run_int32u8s8u8_64x256x32_test();
  run_int32u8u8u8_64x8x32_test();
  run_int32u8u8u8_64x16x32_test();
  run_int32u8u8u8_64x32x32_test();
  run_int32u8u8u8_64x64x32_test();
  run_int32u8u8u8_64x96x32_test();
  run_int32u8u8u8_64x128x32_test();
  run_int32u8u8u8_64x192x32_test();
  run_int32u8u8u8_64x256x32_test();
}

// ============================================================================
// Main Test Function - Run All Configurations
// ============================================================================

inline void run_all_wgmma_latency_tests() {
  printf("\n");
  printf("================================================================================\n");
  printf("                    SM90 GMMA Latency Comprehensive Sweep\n");
  printf("================================================================================\n");
  printf("\n");

  // Run F32 accumulator tests
  run_f32accumulator_tests();

  // Run F16 accumulator tests
  run_f16accumulator_tests();

  // Run INT32 accumulator tests
  run_int32accumulator_tests();

  printf("================================================================================\n");
  printf("                              Sweep Complete\n");
  printf("================================================================================\n");
  printf("\n");
}

// Legacy function signatures for compatibility
float gmma_latency_ss() {
  printf("Running comprehensive WGMMA latency tests...\n");
  run_all_wgmma_latency_tests();
  return 0.0f;
}

#endif
