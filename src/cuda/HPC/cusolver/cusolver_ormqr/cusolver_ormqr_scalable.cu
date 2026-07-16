/*
 * Modified cuSOLVER ormqr example with scalable input sizes
 * Based on NVIDIA's cusolver_ormqr_example.cu
 *
 * Accepts command-line arguments for matrix size:
 *   small:  16x16
 *   medium: 256x256
 *   large:  768x768
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "cusolver_utils.h"

void print_usage(const char* prog_name) {
    printf("Usage: %s [OPTIONS]\n", prog_name);
    printf("\n");
    printf("Options:\n");
    printf("  -m, --m <value>    Matrix dimension (creates mxm matrix)\n");
    printf("\n");
    printf("Presets:\n");
    printf("  small              16x16 matrix\n");
    printf("  medium             256x256 matrix\n");
    printf("  large              768x768 matrix\n");
    printf("\n");
    printf("Examples:\n");
    printf("  %s --m 512         # 512x512 matrix\n", prog_name);
    printf("  %s -m 1024         # 1024x1024 matrix\n", prog_name);
    printf("  %s small           # 16x16 matrix\n", prog_name);
    printf("  %s medium          # 256x256 matrix\n", prog_name);
    printf("\n");
}

int main(int argc, char *argv[]) {
    // Parse command line arguments
    int m = 16;  // Default: small
    const char* size_name = "small";
    bool custom_m = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "--m") == 0 || strcmp(argv[i], "-m") == 0) {
            if (i + 1 < argc) {
                m = atoi(argv[++i]);
                custom_m = true;
                size_name = "custom";
            } else {
                fprintf(stderr, "Error: %s requires a value\n", argv[i]);
                print_usage(argv[0]);
                return 1;
            }
        } else if (strcmp(argv[i], "small") == 0) {
            m = 16;
            size_name = "small";
        } else if (strcmp(argv[i], "medium") == 0) {
            m = 256;
            size_name = "medium";
        } else if (strcmp(argv[i], "large") == 0) {
            m = 768;
            size_name = "large";
        } else {
            fprintf(stderr, "Error: Unknown argument '%s'\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    printf("==============================================\n");
    printf("cuSOLVER ormqr Example (Scalable)\n");
    printf("==============================================\n");
    if (custom_m) {
        printf("Matrix size: %dx%d\n", m, m);
    } else {
        printf("Matrix size: %s (%dx%d)\n", size_name, m, m);
    }
    printf("==============================================\n\n");

    cusolverDnHandle_t cusolverH = NULL;
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream{};

    const int lda = m;
    const int ldb = m;
    const int nrhs = 1; // number of right hand side vectors

    // Generate random matrix A and vector B
    std::vector<double> A(m * m);
    std::vector<double> B(m);
    std::vector<double> XC(ldb * nrhs, 0); // solution matrix from GPU

    // Initialize with random values for reproducibility
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<double> dist(0.0, 10.0);

    for (int i = 0; i < m * m; i++) {
        A[i] = dist(gen);
    }

    for (int i = 0; i < m; i++) {
        B[i] = dist(gen);
    }

    // For small matrices, print them
    if (m <= 16) {
        std::printf("A = (first 8x8 block, matlab base-1)\n");
        int print_size = std::min(m, 8);
        for (int row = 0; row < print_size; row++) {
            for (int col = 0; col < print_size; col++) {
                printf("%.2f ", A[col * m + row]);
            }
            printf("\n");
        }
        std::printf("=====\n");
        std::printf("B = (first 8 elements, matlab base-1)\n");
        for (int i = 0; i < std::min(m, 8); i++) {
            printf("%.2f ", B[i]);
        }
        printf("\n");
        std::printf("=====\n");
    }

    /* device memory */
    double *d_A = nullptr;
    double *d_tau = nullptr;
    double *d_B = nullptr;
    int *d_info = nullptr;
    double *d_work = nullptr;

    int lwork_geqrf = 0;
    int lwork_ormqr = 0;
    int lwork = 0;
    int info = 0;

    const double one = 1;

    /* step 1: create cudense/cublas handle */
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
    CUBLAS_CHECK(cublasCreate(&cublasH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    /* step 2: copy A and B to device */
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(double) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_tau), sizeof(double) * m));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(double) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));

    CUDA_CHECK(
        cudaMemcpyAsync(d_A, A.data(), sizeof(double) * A.size(), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(d_B, B.data(), sizeof(double) * B.size(), cudaMemcpyHostToDevice, stream));

    /* step 3: query working space of geqrf and ormqr */
    CUSOLVER_CHECK(cusolverDnDgeqrf_bufferSize(cusolverH, m, m, d_A, lda, &lwork_geqrf));

    CUSOLVER_CHECK(cusolverDnDormqr_bufferSize(cusolverH, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, m, nrhs, m,
                                               d_A, lda, d_tau, d_B, ldb, &lwork_ormqr));

    lwork = std::max(lwork_geqrf, lwork_ormqr);

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(double) * lwork));

    printf("Starting QR factorization (geqrf)...\n");

    /* step 4: compute QR factorization */
    CUSOLVER_CHECK(cusolverDnDgeqrf(cusolverH, m, m, d_A, lda, d_tau, d_work, lwork, d_info));

    /* check if QR is good or not */
    CUDA_CHECK(cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::printf("after geqrf: info = %d\n", info);
    if (0 > info) {
        std::printf("%d-th parameter is wrong \n", -info);
        exit(1);
    }

    printf("Starting ormqr (Q^T * B)...\n");

    /* step 5: compute Q^T*B */
    CUSOLVER_CHECK(cusolverDnDormqr(cusolverH, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, m, nrhs, m, d_A, lda,
                                    d_tau, d_B, ldb, d_work, lwork, d_info));

    /* check if QR is good or not */
    CUDA_CHECK(cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::printf("after ormqr: info = %d\n", info);
    if (0 > info) {
        std::printf("%d-th parameter is wrong \n", -info);
        exit(1);
    }

    printf("Computing x = R \\ Q^T*B (triangular solve)...\n");

    /* step 6: compute x = R \ Q^T*B */
    CUBLAS_CHECK(cublasDtrsm(cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                             CUBLAS_DIAG_NON_UNIT, m, nrhs, &one, d_A, lda, d_B, ldb));

    CUDA_CHECK(cudaMemcpyAsync(XC.data(), d_B, sizeof(double) * XC.size(), cudaMemcpyDeviceToHost,
                               stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    // For small matrices, print solution
    if (m <= 16) {
        std::printf("X = (matlab base-1)\n");
        for (int i = 0; i < m; i++) {
            printf("%.6f ", XC[i]);
        }
        printf("\n");
    } else {
        std::printf("X = (first 8 elements)\n");
        for (int i = 0; i < std::min(m, 8); i++) {
            printf("%.6f ", XC[i]);
        }
        printf("\n");
    }

    printf("\n==============================================\n");
    printf("SUCCESS: QR factorization and solve completed\n");
    printf("==============================================\n");

    /* free resources */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_tau));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_info));
    CUDA_CHECK(cudaFree(d_work));

    CUBLAS_CHECK(cublasDestroy(cublasH));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}
