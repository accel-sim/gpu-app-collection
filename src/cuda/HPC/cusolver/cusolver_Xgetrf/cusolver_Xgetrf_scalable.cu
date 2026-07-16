/*
 * Modified cuSOLVER Xgetrf example with scalable input sizes
 * Based on NVIDIA's cusolver_Xgetrf_example.cu
 *
 * Accepts command-line arguments for matrix size:
 *   small:  16x16
 *   medium: 128x128
 *   large:  512x512
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>

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
    printf("  medium             128x128 matrix\n");
    printf("  large              512x512 matrix\n");
    printf("\n");
    printf("Examples:\n");
    printf("  %s --m 1024        # 1024x1024 matrix\n", prog_name);
    printf("  %s -m 2048         # 2048x2048 matrix\n", prog_name);
    printf("  %s small           # 16x16 matrix\n", prog_name);
    printf("  %s medium          # 128x128 matrix\n", prog_name);
    printf("\n");
}

int main(int argc, char *argv[]) {
    // Parse command line arguments
    int64_t m = 16;  // Default: small
    const char* size_name = "small";
    bool custom_m = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "--m") == 0 || strcmp(argv[i], "-m") == 0) {
            if (i + 1 < argc) {
                m = atoll(argv[++i]);
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
            m = 128;
            size_name = "medium";
        } else if (strcmp(argv[i], "large") == 0) {
            m = 512;
            size_name = "large";
        } else {
            fprintf(stderr, "Error: Unknown argument '%s'\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    printf("==============================================\n");
    printf("cuSOLVER Xgetrf Example (Scalable)\n");
    printf("==============================================\n");
    if (custom_m) {
        printf("Matrix size: %ldx%ld\n", m, m);
    } else {
        printf("Matrix size: %s (%ldx%ld)\n", size_name, m, m);
    }
    printf("Pivot: ON (compute P*A = L*U)\n");
    printf("==============================================\n\n");

    cusolverDnHandle_t cusolverH = NULL;
    cudaStream_t stream = NULL;

    using data_type = double;

    const int64_t lda = m;
    const int64_t ldb = m;
    const int64_t nrhs = 1;  // number of right-hand sides

    // Generate random matrix A and vector B
    std::vector<data_type> A(m * m);
    std::vector<data_type> B(m);
    std::vector<data_type> X(m, 0);
    std::vector<data_type> LU(lda * m, 0);
    std::vector<int64_t> Ipiv(m, 0);
    int info = 0;

    // Initialize with random values for reproducibility
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<data_type> dist(0.0, 10.0);

    for (int64_t i = 0; i < m * m; i++) {
        A[i] = dist(gen);
    }

    for (int64_t i = 0; i < m; i++) {
        B[i] = dist(gen);
    }

    // For small matrices, print them
    if (m <= 16) {
        std::printf("A = (matlab base-1)\n");
        for (int64_t row = 0; row < m; row++) {
            for (int64_t col = 0; col < m; col++) {
                printf("%.2f ", A[col * m + row]);
            }
            printf("\n");
        }
        std::printf("=====\n");
        std::printf("B = (matlab base-1)\n");
        for (int64_t i = 0; i < m; i++) {
            printf("%.2f ", B[i]);
        }
        printf("\n");
        std::printf("=====\n");
    }

    data_type *d_A = nullptr;  /* device copy of A */
    data_type *d_B = nullptr;  /* device copy of B */
    int64_t *d_Ipiv = nullptr; /* pivoting sequence */
    int *d_info = nullptr;     /* error info */

    size_t workspaceInBytesOnDevice = 0; /* size of workspace */
    void *d_work = nullptr;              /* device workspace for getrf */
    size_t workspaceInBytesOnHost = 0;   /* size of workspace */
    void *h_work = nullptr;              /* host workspace for getrf */

    const int pivot_on = 1;
    const int algo = 0;

    printf("Using New Algo\n");

    /* step 1: create cusolver handle, bind a stream */
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));

    /* step 2: copy A to device */
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_Ipiv), sizeof(int64_t) * Ipiv.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));

    CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, B.data(), sizeof(data_type) * B.size(),
                               cudaMemcpyHostToDevice, stream));

    /* step 3: query working space of Xgetrf */
    cusolverDnParams_t params;
    CUSOLVER_CHECK(cusolverDnCreateParams(&params));

    CUSOLVER_CHECK(cusolverDnXgetrf_bufferSize(cusolverH, params, m, m,
                                                CUDA_R_64F, d_A, lda,
                                                CUDA_R_64F, &workspaceInBytesOnDevice,
                                                &workspaceInBytesOnHost));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), workspaceInBytesOnDevice));

    if (workspaceInBytesOnHost > 0) {
        h_work = malloc(workspaceInBytesOnHost);
        if (h_work == nullptr) {
            throw std::runtime_error("Error: h_work not allocated.");
        }
    }

    printf("Starting LU factorization (Xgetrf)...\n");
    printf("Workspace: device=%zu bytes, host=%zu bytes\n",
           workspaceInBytesOnDevice, workspaceInBytesOnHost);

    /* step 4: LU factorization */
    CUSOLVER_CHECK(cusolverDnXgetrf(cusolverH, params, m, m,
                                     CUDA_R_64F, d_A, lda, d_Ipiv,
                                     CUDA_R_64F, d_work, workspaceInBytesOnDevice,
                                     h_work, workspaceInBytesOnHost, d_info));

    CUDA_CHECK(cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(LU.data(), d_A, sizeof(data_type) * A.size(),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(Ipiv.data(), d_Ipiv, sizeof(int64_t) * Ipiv.size(),
                               cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::printf("after Xgetrf: info = %d\n", info);
    if (0 > info) {
        std::printf("%d-th parameter is wrong \n", -info);
        exit(1);
    } else if (0 < info) {
        std::printf("WARNING: matrix is singular, U(%d,%d) = 0\n", info, info);
    }

    // For small matrices, print pivoting sequence
    if (m <= 16) {
        printf("pivoting sequence, matlab base-1\n");
        for (int64_t i = 0; i < m; i++) {
            printf("Ipiv(%ld) = %ld\n", i + 1, Ipiv[i]);
        }
    } else {
        printf("pivoting sequence (first 8), matlab base-1\n");
        for (int i = 0; i < std::min((int64_t)8, m); i++) {
            printf("Ipiv(%d) = %ld\n", i + 1, Ipiv[i]);
        }
    }

    // For small matrices, print L and U
    if (m <= 16) {
        printf("L and U = (matlab base-1)\n");
        for (int64_t row = 0; row < m; row++) {
            for (int64_t col = 0; col < m; col++) {
                printf("%.2f ", LU[col * m + row]);
            }
            printf("\n");
        }
        std::printf("=====\n");
    }

    printf("Starting solve (Xgetrs)...\n");

    /* step 5: solve A*X = B */
    CUSOLVER_CHECK(cusolverDnXgetrs(cusolverH, params, CUBLAS_OP_N, m, nrhs,
                                     CUDA_R_64F, d_A, lda, d_Ipiv,
                                     CUDA_R_64F, d_B, ldb, d_info));

    CUDA_CHECK(cudaMemcpyAsync(X.data(), d_B, sizeof(data_type) * X.size(),
                               cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Print solution
    if (m <= 16) {
        std::printf("X = (matlab base-1)\n");
        for (int64_t i = 0; i < m; i++) {
            printf("%.6f ", X[i]);
        }
        printf("\n");
    } else {
        std::printf("X = (first 8 elements)\n");
        for (int i = 0; i < std::min((int64_t)8, m); i++) {
            printf("%.6f ", X[i]);
        }
        printf("\n");
    }

    printf("\n==============================================\n");
    printf("SUCCESS: LU factorization and solve completed\n");
    printf("==============================================\n");

    /* free resources */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_Ipiv));
    CUDA_CHECK(cudaFree(d_info));
    CUDA_CHECK(cudaFree(d_work));

    CUSOLVER_CHECK(cusolverDnDestroyParams(params));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    if (h_work) {
        free(h_work);
    }

    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}
