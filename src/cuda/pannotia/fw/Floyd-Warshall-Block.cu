/************************************************************************************\ 
 *                                                                                  *
 * Copyright © 2014 Advanced Micro Devices, Inc.                                    *
 * Copyright (c) 2015 Mark D. Hill and David A. Wood                                *
 * All rights reserved.                                                             *
 *                                                                                  *
 * Redistribution and use in source and binary forms, with or without               *
 * modification, are permitted provided that the following are met:                 *
 *                                                                                  *
 * You must reproduce the above copyright notice.                                   *
 *                                                                                  *
 * Neither the name of the copyright holder nor the names of its contributors       *
 * may be used to endorse or promote products derived from this software            *
 * without specific, prior, written permission from at least the copyright holder.  *
 *                                                                                  *
 * You must include the following terms in your license and/or other materials      *
 * provided with the software.                                                      *
 *                                                                                  *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"      *
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE        *
 * IMPLIED WARRANTIES OF MERCHANTABILITY, NON-INFRINGEMENT, AND FITNESS FOR A       *
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER        *
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,         *
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT  *
 * OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS      *
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN          *
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING  *
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY   *
 * OF SUCH DAMAGE.                                                                  *
 *                                                                                  *
 * Without limiting the foregoing, the software may implement third party           *
 * technologies for which you must obtain licenses from parties other than AMD.     *
 * You agree that AMD has not obtained or conveyed to you, and that you shall       *
 * be responsible for obtaining the rights to use and/or distribute the applicable  *
 * underlying intellectual property rights related to the third party technologies. *
 * These third party technologies are not licensed hereunder.                       *
 *                                                                                  *
 * If you use the software (in whole or in part), you shall adhere to all           *
 * applicable U.S., European, and other export laws, including but not limited to   *
 * the U.S. Export Administration Regulations ("EAR"�) (15 C.F.R Sections 730-774),  *
 * and E.U. Council Regulation (EC) No 428/2009 of 5 May 2009.  Further, pursuant   *
 * to Section 740.6 of the EAR, you hereby certify that, except pursuant to a       *
 * license granted by the United States Department of Commerce Bureau of Industry   *
 * and Security or as otherwise permitted pursuant to a License Exception under     *
 * the U.S. Export Administration Regulations ("EAR"), you will not (1) export,     *
 * re-export or release to a national of a country in Country Groups D:1, E:1 or    *
 * E:2 any restricted technology, software, or source code you receive hereunder,   *
 * or (2) export to Country Groups D:1, E:1 or E:2 the direct product of such       *
 * technology or software, if such foreign produced direct product is subject to    *
 * national security controls as identified on the Commerce Control List (currently *
 * found in Supplement 1 to Part 774 of EAR).  For the most current Country Group   *
 * listings, or for additional information about the EAR or your obligations under  *
 * those regulations, please refer to the U.S. Bureau of Industry and Security's    *
 * website at http://www.bis.doc.gov/.                                              *
 *                                                                                  *
\************************************************************************************/

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <omp.h>
#include "../graph_parser/util.h"
#include "kernel_block.cu"
#include "parse.h"

#ifdef GEM5_FUSION
#include <stdint.h>
extern "C" {
void m5_work_begin(uint64_t workid, uint64_t threadid);
void m5_work_end(uint64_t workid, uint64_t threadid);
}
#endif

#ifdef GEM5_FUSION
#define MAX_ITERS 36
#else
#include <stdint.h>
#define MAX_ITERS INT32_MAX
#endif

#define BIGNUM 999999
#define TRUE 1
#define FALSE 0

int main(int argc, char **argv)
{
    char *tmpchar;
    bool verify_results = false;

    int num_nodes;
    int num_edges;

    cudaError_t err = cudaSuccess;

    // Get program input
    if (argc >= 2) {
        tmpchar = argv[1];  // Graph input file
    } else {
        fprintf(stderr, "You did something wrong!\n");
        exit(1);
    }

    if (argc >= 3) {
        if (atoi(argv[2]) == 1) {
            verify_results = true;
        }
    }

    // Parse the adjacency matrix
    int *adjmatrix = parse_graph_file(&num_nodes, &num_edges, tmpchar);
    int dim = num_nodes;

    // Initialize the distance matrix
    int *distmatrix = (int *)malloc(dim * dim * sizeof(int));
    if (!distmatrix) fprintf(stderr, "malloc failed - distmatrix\n");

    // Initialize the result matrix
    int *result = (int *)malloc(dim * dim * sizeof(int));
    if (!result) fprintf(stderr, "malloc failed - result\n");

    // TODO: Now only supports integer weights
    // Setup the input matrix
    for (int i = 0 ; i < dim; i++) {
        for (int j = 0 ; j < dim; j++) {
            if (i == j) {
                // Diagonal
                distmatrix[i * dim + j] = 0;
            } else if (adjmatrix[i * dim + j] == -1) {
                // Without edge
                distmatrix[i * dim + j] = BIGNUM;
            } else {
                // With edge
                distmatrix[i * dim + j] = adjmatrix[i * dim + j];
            }
        }
    }

    int *dist_d;

    // Create device-side FW buffers
    err = cudaMalloc(&dist_d, dim * dim * sizeof(int));
    if (err != cudaSuccess) {
        printf("ERROR: cudaMalloc dist_d (size:%d) => %d\n",  dim * dim , err);
        return -1;
    }

    double timer1 = gettime();

#ifdef GEM5_FUSION
    m5_work_begin(0, 0);
#endif

    // Copy the dist matrix to the device
    err = cudaMemcpy(dist_d, distmatrix, dim * dim * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMemcpy feature_d (size:%d) => %d\n", dim * dim, err);
        return -1;
    }

    // Work dimension
    int block_size = 16;
    int num_blk_per_dim = num_nodes / block_size;
    dim3 threads(block_size, block_size, 1);
    dim3 grid_dia(block_size, block_size, 1);
    dim3 grid_strip_x(num_blk_per_dim, block_size, 1);
    dim3 grid_strip_y(block_size, num_blk_per_dim, 1);
    dim3 grid_remain(num_blk_per_dim, num_blk_per_dim, 1);

    double timer3 = gettime();
    // Main computation loop
    for (int blk = 0; blk < num_blk_per_dim && blk < MAX_ITERS; blk++) {
        floydwarshall_dia_block<<<grid_dia, threads>>>(dist_d, blk, dim);
        floydwarshall_strip_blocks_x<<<grid_strip_x, threads>>>(dist_d, blk, dim);
        floydwarshall_strip_blocks_y<<<grid_strip_y, threads>>>(dist_d, blk, dim);
        floydwarshall_remaining_blocks<<<grid_remain, threads>>>(dist_d, blk, dim);
    }
    cudaThreadSynchronize();

    double timer4 = gettime();
    err = cudaMemcpy(result, dist_d, dim * dim * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: read back dist_d %d failed\n", err);
        return -1;
    }

#ifdef GEM5_FUSION
    m5_work_end(0, 0);
#endif

    double timer2 = gettime();

    printf("kernel time = %lf ms\n", (timer4 - timer3) * 1000);
    printf("kernel + memcpy time = %lf ms\n", (timer2 - timer1) * 1000);

    if (verify_results) {
        // Below is the verification part
        // Calculate on the CPU
        int *dist = distmatrix;
        for (int k = 0; k < dim; k++) {
            for (int i = 0; i < dim; i++) {
                for (int j = 0; j < dim; j++) {
                    if (dist[i * dim + k] + dist[k * dim + j] < dist[i * dim + j]) {
                        dist[i * dim + j] = dist[i * dim + k] + dist[k * dim + j];
                    }
                }
            }
        }

        // Compare results
        bool check_flag = 0;
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                if (dist[i * dim + j] !=  result[i * dim + j]) {
                    printf("mismatch at (%d, %d)\n", i, j);
                    check_flag = 1;
                }
            }
        }
        // If there is mismatch, report
        if (check_flag) {
            fprintf(stderr, "WARNING: Produced incorrect results!\n");
        } else {
            printf("Results are correct!\n");
        }
    }

    printf("Finishing Floyd-Warshall\n");

    // Free host-side buffers
    free(adjmatrix);
    free(result);
    free(distmatrix);

    // Free CUDA buffers
    cudaFree(dist_d);

    return 0;

}
