/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */



/*
 * Example showing the use of LTO callbacks with CUFFT to perform
 * truncation with zero padding.
 *
*/

#include <cuda_runtime_api.h>
#include <cufftXt.h>
#include <cstring>
#include "r2c_c2r_reference.h"
#include "common.h"
#include "callback_params.h"

// NOTE: Header containing the compiled LTO callback device function in a C array, generated with bin2c
#include "r2c_c2r_lto_callback_device_fatbin.h"

// Note: Removed static_assert since window_size and signal_size are now runtime variables

int test_r2c_window_c2r() {

	// Padded array for in-place transforms - use heap allocation for large sizes
	const size_t array_size = batches * 2 * complex_signal_size;
	float *input_signals = new float[array_size]();
	float *output_signals = new float[array_size];
	float *reference = new float[array_size];

	init_input_signals(batches, signal_size, input_signals);

	const size_t complex_size_bytes = batches * complex_signal_size * 2 * sizeof(float);

	// Allocate and copy input from host to GPU
	float *device_signals;
	CHECK_ERROR(cudaMalloc((void **)&device_signals, complex_size_bytes));
	CHECK_ERROR(cudaMemcpy(device_signals, input_signals, complex_size_bytes, cudaMemcpyHostToDevice));

	// Create a CUFFT plan for the forward transform, and a cuFFT plan for the inverse transform with load callback
	cufftHandle forward_plan, inverse_plan_cb;
	size_t work_size;

	CHECK_ERROR(cufftCreate(&forward_plan));
	CHECK_ERROR(cufftCreate(&inverse_plan_cb));

	// NOTE: LTO callbacks must be set before plan creation and cannot be unset (yet)
#ifdef CB_USE_CONSTANT_MEMORY
	cb_params *device_params = nullptr;
	std::string callback_name = "windowing_constant_memory_callback";
#else
	// Define a structure used to pass in the window size
	cb_params host_params;
	host_params.window_size = window_size;
	host_params.signal_size = complex_signal_size;

	// Allocate and copy callback parameters from host to GPU
	cb_params *device_params;
	CHECK_ERROR(cudaMalloc((void **)&device_params, sizeof(cb_params)));
	CHECK_ERROR(cudaMemcpy(device_params, &host_params, sizeof(cb_params), cudaMemcpyHostToDevice));

	std::string callback_name = "windowing_callback";
#endif
	size_t lto_callback_fatbin_size = sizeof(window_callback);
	printf("Setting up LTO callback '%s', fatbin size: %zu bytes\n", callback_name.c_str(), lto_callback_fatbin_size);
	cufftResult cb_result = cufftXtSetJITCallback(inverse_plan_cb,
                                      callback_name.c_str(),
                                      (void*)window_callback,
                                      lto_callback_fatbin_size,
                                      CUFFT_CB_LD_COMPLEX,
                                      (void **)&device_params);
	printf("cufftXtSetJITCallback returned: %d\n", cb_result);
	CHECK_ERROR(cb_result);

	printf("Creating forward plan (R2C): signal_size=%u, batches=%u\n", signal_size, batches);
	CHECK_ERROR(cufftMakePlan1d(forward_plan, signal_size, CUFFT_R2C, batches, &work_size));
	printf("Creating inverse plan (C2R) with callback: signal_size=%u, batches=%u\n", signal_size, batches);
	CHECK_ERROR(cufftMakePlan1d(inverse_plan_cb, signal_size, CUFFT_C2R, batches, &work_size));

	// Transform signal forward
	printf("Transforming signal cufftExecR2C\n");
	CHECK_ERROR(cufftExecR2C(forward_plan,    (cufftReal *)device_signals, (cufftComplex *)device_signals));

	// Apply window via load callback and inverse-transform the signal
	printf("Transforming signal cufftExecC2R\n");
	CHECK_ERROR(cufftExecC2R(inverse_plan_cb, (cufftComplex *)device_signals, (cufftReal *)device_signals));

	// Copy device memory to host
	CHECK_ERROR(cudaMemcpy(output_signals, device_signals, complex_size_bytes, cudaMemcpyDeviceToHost));

	// Destroy CUFFT context
	CHECK_ERROR(cufftDestroy(forward_plan));
	CHECK_ERROR(cufftDestroy(inverse_plan_cb));

	// Cleanup memory
	CHECK_ERROR(cudaFree(device_signals));
	CHECK_ERROR(cudaFree(device_params));

	// Compute reference
	if(reference_r2c_window_c2r(batches, signal_size, window_size, input_signals, reference) != PASS_VALUE) {
		printf("Failed to compute the reference");
		delete[] input_signals;
		delete[] output_signals;
		delete[] reference;
		return ERROR_VALUE;
	};

	double l2_error = compute_error<float>(reference, output_signals, batches, signal_size);
	printf("L2 error: %e\n", l2_error);

	// Cleanup heap-allocated arrays
	delete[] input_signals;
	delete[] output_signals;
	delete[] reference;

	return (l2_error < threshold) ? PASS_VALUE : ERROR_VALUE;
}

// Define global variables for size configuration
unsigned batches = 100;
unsigned signal_size = 128;
unsigned window_size = 16;
unsigned complex_signal_size = signal_size / 2 + 1;

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    // Parse named command-line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--batches") == 0 || strcmp(argv[i], "-b") == 0) {
            if (i + 1 < argc) {
                batches = atoi(argv[++i]);
            } else {
                printf("Error: %s requires a value\n", argv[i]);
                printf("Usage: %s [--batches|-b <value>] [--signal-size|-s <value>] [--window-size|-w <value>]\n", argv[0]);
                printf("   or: %s <small|medium|large>\n", argv[0]);
                return ERROR_VALUE;
            }
        } else if (strcmp(argv[i], "--signal-size") == 0 || strcmp(argv[i], "-s") == 0) {
            if (i + 1 < argc) {
                signal_size = atoi(argv[++i]);
            } else {
                printf("Error: %s requires a value\n", argv[i]);
                printf("Usage: %s [--batches|-b <value>] [--signal-size|-s <value>] [--window-size|-w <value>]\n", argv[0]);
                printf("   or: %s <small|medium|large>\n", argv[0]);
                return ERROR_VALUE;
            }
        } else if (strcmp(argv[i], "--window-size") == 0 || strcmp(argv[i], "-w") == 0) {
            if (i + 1 < argc) {
                window_size = atoi(argv[++i]);
            } else {
                printf("Error: %s requires a value\n", argv[i]);
                printf("Usage: %s [--batches|-b <value>] [--signal-size|-s <value>] [--window-size|-w <value>]\n", argv[0]);
                printf("   or: %s <small|medium|large>\n", argv[0]);
                return ERROR_VALUE;
            }
        } else if (strcmp(argv[i], "small") == 0) {
            batches = 128;
            signal_size = 64;
            window_size = 16;
        } else if (strcmp(argv[i], "medium") == 0) {
            batches = 500;
            signal_size = 256;
            window_size = 32;
        } else if (strcmp(argv[i], "large") == 0) {
            batches = 2000;
            signal_size = 32;
            window_size = 8;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: %s [OPTIONS]\n", argv[0]);
            printf("\nOptions:\n");
            printf("  -b, --batches <value>      Number of FFT batches (default: 100)\n");
            printf("  -s, --signal-size <value>  Size of each signal (default: 128)\n");
            printf("  -w, --window-size <value>  Window size for truncation (default: 16)\n");
            printf("\nPresets:\n");
            printf("  small   : batches=128,  signal_size=64,  window_size=16\n");
            printf("  medium  : batches=500,  signal_size=256, window_size=32\n");
            printf("  large   : batches=2000, signal_size=32,  window_size=8\n");
            printf("\nExamples:\n");
            printf("  %s --batches 1024 --signal-size 512 --window-size 32\n", argv[0]);
            printf("  %s -b 1024 -s 512 -w 32\n", argv[0]);
            printf("  %s medium --batches 1000\n", argv[0]);
            printf("  %s small\n", argv[0]);
            return PASS_VALUE;
        } else {
            printf("Error: Unknown argument '%s'\n", argv[i]);
            printf("Usage: %s [--batches|-b <value>] [--signal-size|-s <value>] [--window-size|-w <value>]\n", argv[0]);
            printf("   or: %s <small|medium|large>\n", argv[0]);
            printf("   or: %s --help\n", argv[0]);
            return ERROR_VALUE;
        }
    }

    complex_signal_size = signal_size / 2 + 1;

    printf("==============================================\n");
    printf("cuFFT LTO R2C:C2R Example (Scalable)\n");
    printf("==============================================\n");
    printf("Batches: %u\n", batches);
    printf("Signal size: %u\n", signal_size);
    printf("Window size: %u\n", window_size);
    printf("==============================================\n\n");

    struct cudaDeviceProp properties;
    int device;
    CHECK_ERROR(cudaGetDevice(&device));
    CHECK_ERROR(cudaGetDeviceProperties(&properties, device));
    if (!(properties.major >= 5)) {
        printf("cuFFT with LTO requires CUDA architecture SM5.0 or higher\n");
        return ERROR_VALUE;
    }

    int result = test_r2c_window_c2r();

    printf("\n==============================================\n");
    if (result == PASS_VALUE) {
        printf("SUCCESS: LTO R2C:C2R completed\n");
    } else {
        printf("FAILED: LTO R2C:C2R\n");
    }
    printf("==============================================\n");

    return result;
}