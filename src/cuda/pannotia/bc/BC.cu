/************************************************************************************\ 
 *                                                                                  *
 * Copyright � 2014 Advanced Micro Devices, Inc.                                    *
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
 * the U.S. Export Administration Regulations ("EAR") (15 C.F.R Sections 730-774),  *
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
#include <algorithm>
#include "BC.h"
#include "../graph_parser/util.h"
#include "kernel.cu"

#ifdef GEM5_FUSION
#include <stdint.h>
extern "C" {
void m5_work_begin(uint64_t workid, uint64_t threadid);
void m5_work_end(uint64_t workid, uint64_t threadid);
}
#endif

#ifdef GEM5_FUSION
#define MAX_ITERS 150
#else
#include <stdint.h>
#define MAX_ITERS INT32_MAX
#endif

void print_vector(int *vector, int num);
void print_vectorf(float *vector, int num);

int main(int argc, char **argv)
{

    char *tmpchar;

    int num_nodes;
    int num_edges;
    bool directed = 1;

    cudaError_t err;

    if (argc == 2) {
        tmpchar     = argv[1];       //graph inputfile
    } else {
        fprintf(stderr, "You did something wrong!\n");
        exit(1);
    }

    // Parse graph and store it in a CSR format
    csr_array *csr = parseCOO(tmpchar, &num_nodes, &num_edges, directed);

    // Allocate the bc host array
    float *bc_h = (float *)malloc(num_nodes * sizeof(float));
    if (!bc_h) fprintf(stderr, "malloc failed bc_h\n");

    // Create device-side buffers
    float *bc_d, *sigma_d, *rho_d;
    int *dist_d, *stop_d;
    int *row_d, *col_d, *row_trans_d, *col_trans_d;

    // Create betweenness centrality buffers
    err = cudaMalloc(&bc_d, num_nodes * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMalloc bc_d %s\n", cudaGetErrorString(err));
        return -1;
    }
    err = cudaMalloc(&dist_d, num_nodes * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMalloc dist_d %s\n", cudaGetErrorString(err));
        return -1;
    }
    err = cudaMalloc(&sigma_d, num_nodes * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMalloc sigma_d %s\n", cudaGetErrorString(err));
        return -1;
    }
    err = cudaMalloc(&rho_d, num_nodes * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMalloc rho_d %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Create termination variable buffer
    err = cudaMalloc(&stop_d, sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMalloc stop_d %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Create graph buffers
    err = cudaMalloc(&row_d, (num_nodes + 1) * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMalloc row_d %s\n", cudaGetErrorString(err));
        return -1;
    }
    err = cudaMalloc(&col_d, num_edges * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMalloc col_d %s\n", cudaGetErrorString(err));
        return -1;
    }
    err = cudaMalloc(&row_trans_d, (num_nodes + 1) * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMalloc row_trans_d %s\n", cudaGetErrorString(err));
        return -1;
    }
    err = cudaMalloc(&col_trans_d, num_edges * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMalloc col_trans_d %s\n", cudaGetErrorString(err));
        return -1;
    }

    double timer1, timer2;
    double timer3, timer4;

    timer1 = gettime();

#ifdef GEM5_FUSION
    m5_work_begin(0, 0);
#endif

    // Copy data to device-side buffers
    err = cudaMemcpy(row_d, csr->row_array, (num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMemcpy row_d (size:%d) => %s\n", num_nodes, cudaGetErrorString(err));
        return -1;
    }

    err = cudaMemcpy(col_d, csr->col_array, num_edges * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMemcpy col_d (size:%d) => %s\n", num_nodes, cudaGetErrorString(err));
        return -1;
    }

    // Copy data to device-side buffers
    err = cudaMemcpy(row_trans_d, csr->row_array_t, (num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMemcpy row_trans_d (size:%d) => %s\n", num_nodes, cudaGetErrorString(err));
        return -1;
    }

    err = cudaMemcpy(col_trans_d, csr->col_array_t, num_edges * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMemcpy col_trans_d (size:%d) => %s\n", num_nodes, cudaGetErrorString(err));
        return -1;
    }

    timer3 = gettime();

    // Set up kernel dimensions
    int local_worksize = 128;
    dim3 threads(local_worksize, 1, 1);
    int num_blocks = (num_nodes + local_worksize - 1) / local_worksize;
    dim3 grid(num_blocks, 1, 1);

    // Initialization
    clean_bc<<< grid, threads >>>(bc_d, num_nodes);

    // Main computation loop
    for (int i = 0; i < num_nodes && i < MAX_ITERS; i++) {

        clean_1d_array<<< grid, threads >>>(i, dist_d, sigma_d, rho_d,
                                            num_nodes);

        // Depth of the traversal
        int dist = 0;
        // Termination variable
        int stop = 1;

        // Traverse the graph from the source node i
        do {
            stop = 0;

            // Copy the termination variable to the device
            cudaMemcpy(stop_d, &stop, sizeof(int), cudaMemcpyHostToDevice);

            bfs_kernel<<< grid, threads >>>(row_d, col_d, dist_d, rho_d, stop_d,
                                            num_nodes, num_edges, dist);

            // Copy back the termination variable from the device
            cudaMemcpy(&stop, stop_d, sizeof(int), cudaMemcpyDeviceToHost);

            // Another level
            dist++;

        } while (stop) ;

        cudaThreadSynchronize();

        // Traverse back from the deepest part of the tree
        while (dist) {
            backtrack_kernel<<< grid, threads >>>(row_trans_d, col_trans_d,
                                                dist_d, rho_d, sigma_d,
                                                num_nodes, num_edges, dist, i,
                                                bc_d);

            // Back one level
            dist--;
        }
        cudaThreadSynchronize();

    }
    cudaThreadSynchronize();
    timer4 = gettime();

    // Copy back the results for the bc array
    err = cudaMemcpy(bc_h, bc_d, num_nodes * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: read buffer bc_d (%s)\n", cudaGetErrorString(err));
        return -1;
    }

#ifdef GEM5_FUSION
    m5_work_end(0, 0);
#endif

    timer2 = gettime();

    printf("kernel + memcopy time = %lf ms\n", (timer4 - timer3) * 1000);
    printf("kernel execution time = %lf ms\n", (timer2 - timer1) * 1000);

#if 0
    //dump the results to the file
    print_vectorf(bc_h, num_nodes);
#endif

    // Clean up the host-side buffers
    free(bc_h);
    free(csr->row_array);
    free(csr->col_array);
    free(csr->data_array);
    free(csr->row_array_t);
    free(csr->col_array_t);
    free(csr->data_array_t);
    free(csr);

    // Clean up the device-side buffers
    cudaFree(bc_d);
    cudaFree(dist_d);
    cudaFree(sigma_d);
    cudaFree(rho_d);
    cudaFree(stop_d);
    cudaFree(row_d);
    cudaFree(col_d);
    cudaFree(row_trans_d);
    cudaFree(col_trans_d);

    return 0;

}

void print_vector(int *vector, int num)
{
    for (int i = 0; i < num; i++)
        printf("%d: %d \n", i + 1, vector[i]);
    printf("\n");
}

void print_vectorf(float *vector, int num)
{

    FILE * fp = fopen("result.out", "w");
    if (!fp) {
        printf("ERROR: unable to open result.txt\n");
    }

    for (int i = 0; i < num; i++) {
        fprintf(fp, "%f\n", vector[i]);
    }

    fclose(fp);

}
