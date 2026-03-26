/*
 * SPDX-FileCopyrightText: Copyright (c) 2020 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


#include <array>
#include <complex>
#include <iostream>
#include <random>
#include <vector>
#include <cstring>
#include <cuda_runtime.h>
#include <cufft.h>

#include "cufft_utils.h"

using dim_t = std::array<int, 3>;

int main(int argc, char *argv[]) {
    cufftHandle plan;
    cudaStream_t stream = NULL;

    // Default values
    int n = 16;
    int batch_size = 4;

    // Parse named command-line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--n") == 0 || strcmp(argv[i], "-n") == 0) {
            if (i + 1 < argc) {
                n = atoi(argv[++i]);
            } else {
                std::printf("Error: %s requires a value\n", argv[i]);
                std::printf("Usage: %s [--n|-n <value>] [--batch-size|-b <value>]\n", argv[0]);
                std::printf("   or: %s <small|medium|large>\n", argv[0]);
                return EXIT_FAILURE;
            }
        } else if (strcmp(argv[i], "--batch-size") == 0 || strcmp(argv[i], "-b") == 0) {
            if (i + 1 < argc) {
                batch_size = atoi(argv[++i]);
            } else {
                std::printf("Error: %s requires a value\n", argv[i]);
                std::printf("Usage: %s [--n|-n <value>] [--batch-size|-b <value>]\n", argv[0]);
                std::printf("   or: %s <small|medium|large>\n", argv[0]);
                return EXIT_FAILURE;
            }
        } else if (strcmp(argv[i], "small") == 0) {
            n = 16;           // 16×16×16 = 4K elements
            batch_size = 4;
        } else if (strcmp(argv[i], "medium") == 0) {
            n = 32;           // 32×32×32 = 32K elements
            batch_size = 8;
        } else if (strcmp(argv[i], "large") == 0) {
            n = 64;          // 64×64×64 = 262K elements
            batch_size = 8;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            std::printf("Usage: %s [OPTIONS]\n", argv[0]);
            std::printf("\nOptions:\n");
            std::printf("  -n, --n <value>            3D FFT dimension (n×n×n) (default: 16)\n");
            std::printf("  -b, --batch-size <value>   Number of batched 3D FFTs (default: 4)\n");
            std::printf("\nPresets:\n");
            std::printf("  small   : n=16,  batch_size=4  (16×16×16 = 4K elements)\n");
            std::printf("  medium  : n=32,  batch_size=8  (32×32×32 = 32K elements)\n");
            std::printf("  large   : n=64,  batch_size=8  (64×64×64 = 262K elements)\n");
            std::printf("\nExamples:\n");
            std::printf("  %s --n 64 --batch-size 16\n", argv[0]);
            std::printf("  %s -n 128 -b 4\n", argv[0]);
            std::printf("  %s medium --batch-size 16\n", argv[0]);
            std::printf("  %s large\n", argv[0]);
            return EXIT_SUCCESS;
        } else {
            std::printf("Error: Unknown argument '%s'\n", argv[i]);
            std::printf("Usage: %s [--n|-n <value>] [--batch-size|-b <value>]\n", argv[0]);
            std::printf("   or: %s <small|medium|large>\n", argv[0]);
            std::printf("   or: %s --help\n", argv[0]);
            return EXIT_FAILURE;
        }
    }

    dim_t fft = {n, n, n};
    int fft_size = fft[0] * fft[1] * fft[2];

    std::printf("==============================================\n");
    std::printf("cuFFT 3D C2C Example (Scalable)\n");
    std::printf("==============================================\n");
    std::printf("FFT dimension: %d×%d×%d\n", n, n, n);
    std::printf("FFT size: %d\n", fft_size);
    std::printf("Batch size: %d\n", batch_size);
    std::printf("==============================================\n\n");

    using scalar_type = float;
    using data_type = std::complex<scalar_type>;

    std::vector<data_type> data(fft_size * batch_size);

    // Initialize with simple pattern
    std::random_device rd;
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (int i = 0; i < data.size(); i++) {
        data[i] = data_type(dist(gen), dist(gen));
    }

    if (n <= 16) {
        std::printf("Input array (first 8 elements):\n");
        for (int i = 0; i < std::min(8, (int)data.size()); i++) {
            std::printf("%f + %fj\n", data[i].real(), data[i].imag());
        }
        std::printf("=====\n");
    }

    cufftComplex *d_data = nullptr;

    // inembed/onembed being nullptr indicates contiguous data for each batch, then the stride and dist settings are ignored
    CUFFT_CALL(cufftPlanMany(&plan, fft.size(), fft.data(),
                             nullptr, 1, 0, // *inembed, istride, idist
                             nullptr, 1, 0, // *onembed, ostride, odist
                             CUFFT_C2C, batch_size));

    CUDA_RT_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUFFT_CALL(cufftSetStream(plan, stream));

    // Create device data arrays
    CUDA_RT_CALL(cudaMalloc(reinterpret_cast<void **>(&d_data), sizeof(data_type) * data.size()));
    CUDA_RT_CALL(cudaMemcpyAsync(d_data, data.data(), sizeof(data_type) * data.size(),
                                 cudaMemcpyHostToDevice, stream));

    /*
     * Note:
     *  Identical pointers to data and output arrays implies in-place transformation
     */
    std::printf("Executing forward FFT...\n");
    CUFFT_CALL(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));
    CUDA_RT_CALL(cudaStreamSynchronize(stream));
    std::printf("Forward FFT complete.\n");

    if (n <= 16) {
        CUDA_RT_CALL(cudaMemcpyAsync(data.data(), d_data, sizeof(data_type) * data.size(),
                                     cudaMemcpyDeviceToHost, stream));
        CUDA_RT_CALL(cudaStreamSynchronize(stream));
        std::printf("Output after Forward (first 8 elements):\n");
        for (int i = 0; i < std::min(8, (int)data.size()); i++) {
            std::printf("%f + %fj\n", data[i].real(), data[i].imag());
        }
        std::printf("=====\n");
    }

    // Normalize the data and inverse FFT
    std::printf("Executing inverse FFT...\n");
    scaling_kernel<<<(data.size() + 127) / 128, 128, 0, stream>>>(d_data, data.size(), 1.f/fft_size);
    CUFFT_CALL(cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE));
    CUDA_RT_CALL(cudaStreamSynchronize(stream));
    std::printf("Inverse FFT complete.\n");

    if (n <= 16) {
        CUDA_RT_CALL(cudaMemcpyAsync(data.data(), d_data, sizeof(data_type) * data.size(),
                                     cudaMemcpyDeviceToHost, stream));
        CUDA_RT_CALL(cudaStreamSynchronize(stream));
        std::printf("Output after Inverse (first 8 elements):\n");
        for (int i = 0; i < std::min(8, (int)data.size()); i++) {
            std::printf("%f + %fj\n", data[i].real(), data[i].imag());
        }
        std::printf("=====\n");
    }



    /* free resources */
    CUDA_RT_CALL(cudaFree(d_data));

    CUFFT_CALL(cufftDestroy(plan));

    CUDA_RT_CALL(cudaStreamDestroy(stream));

    CUDA_RT_CALL(cudaDeviceReset());

    std::printf("\n==============================================\n");
    std::printf("SUCCESS: 3D C2C FFT completed\n");
    std::printf("==============================================\n");

    return EXIT_SUCCESS;
}