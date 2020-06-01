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
#include "../graph_parser/parse.h"
#include "../graph_parser/util.h"
#include "kernel_spmv.cu"

#ifdef GEM5_FUSION
#include <stdint.h>
extern "C" {
void m5_work_begin(uint64_t workid, uint64_t threadid);
void m5_work_end(uint64_t workid, uint64_t threadid);
}
#endif

// Iteration count
#define ITER 20

void print_vectorf(float *vector, int num);

int main(int argc, char **argv)
{
    char *tmpchar;

    int num_nodes;
    int num_edges;
    int file_format = 1;
    bool directed = 0;

    cudaError_t err = cudaSuccess;

    if (argc == 3) {
        tmpchar = argv[1]; // Graph inputfile
        file_format = atoi(argv[2]);
    } else {
        fprintf(stderr, "You did something wrong!\n");
        exit(1);
    }

    // Allocate the csr structure
    csr_array *csr;

    // Parse graph files into csr structure
    if (file_format == 1) {
       csr = parseMetis_transpose(tmpchar, &num_nodes, &num_edges, directed);
    } else if (file_format == 0) {
       csr = parseCOO_transpose(tmpchar, &num_nodes, &num_edges, directed);
    } else {
       printf("reserve for future");
       exit(1);
    }

    // Allocate rank_arrays
    float *pagerank_array = (float *)malloc(num_nodes * sizeof(float));
    if (!pagerank_array) fprintf(stderr, "malloc failed page_rank_array\n");
    float *pagerank_array2 = (float *)malloc(num_nodes * sizeof(float));
    if (!pagerank_array2) fprintf(stderr, "malloc failed page_rank_array2\n");

    int *row_d;
    int *col_d;
    float *data_d;

    float *pagerank1_d;
    float *pagerank2_d;
    int *col_cnt_d;

    // Create device-side buffers for the graph
    err = cudaMalloc(&row_d, (num_nodes + 1) * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMalloc row_d (size:%d) => %s\n",  num_nodes, cudaGetErrorString(err));
        return -1;
    }
    err = cudaMalloc(&col_d, num_edges * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMalloc col_d (size:%d) => %s\n",  num_edges, cudaGetErrorString(err));
        return -1;
    }
    err = cudaMalloc(&data_d, num_edges * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMalloc data_d (size:%d) => %s\n", num_edges, cudaGetErrorString(err));
        return -1;
    }

    // Create buffers for pagerank
    err = cudaMalloc(&pagerank1_d, num_nodes * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMalloc pagerank1_d (size:%d) => %s\n", num_nodes, cudaGetErrorString(err));
        return -1;
    }
    err = cudaMalloc(&pagerank2_d, num_nodes * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMalloc pagerank2_d (size:%d) => %s\n", num_nodes, cudaGetErrorString(err));
        return -1;
    }
    err = cudaMalloc(&col_cnt_d, num_nodes * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMalloc col_cnt_d (size:%d) => %s\n", num_nodes, cudaGetErrorString(err));
        return -1;
    }

    double timer1 = gettime();

#ifdef GEM5_FUSION
    m5_work_begin(0, 0);
#endif

    // Copy the data to the device-side buffers
    err = cudaMemcpy(row_d, csr->row_array, (num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR:#endif cudaMemcpy row_d (size:%d) => %s\n", num_nodes, cudaGetErrorString(err));
        return -1;
    }

    err = cudaMemcpy(col_d, csr->col_array, num_edges * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMemcpy col_d (size:%d) => %s\n", num_nodes, cudaGetErrorString(err));
        return -1;
    }

    err = cudaMemcpy(col_cnt_d, csr->col_cnt, num_nodes * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMemcpy col_cnt_d (size:%d) => %s\n", num_nodes, cudaGetErrorString(err));
        return -1;
    }

    // Set up work dimensions
    int block_size = 64;
    int num_blocks = (num_nodes + block_size - 1) / block_size;

    dim3 threads(block_size, 1, 1);
    dim3 grid(num_blocks, 1, 1);

    double timer3 = gettime();

    // Launch the initialization kernel
    inibuffer <<<grid, threads>>>(pagerank1_d, pagerank2_d, num_nodes);
    cudaThreadSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaLaunch failed (%s)\n", cudaGetErrorString(err));
        return -1;
    }

    // Initialize the CSR
    inicsr <<<grid, threads>>>(row_d, col_d, data_d, col_cnt_d, num_nodes,
                               num_edges);
    cudaThreadSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaLaunch failed (%s)\n", cudaGetErrorString(err));
        return -1;
    }

    // Run PageRank for some iter. TO: convergence determination
    for (int i = 0; i < ITER; i++) {
        // Launch pagerank kernel 1
        spmv_csr_scalar_kernel <<<grid, threads>>>(num_nodes, row_d, col_d,
                                                   data_d, pagerank1_d,
                                                   pagerank2_d);

        // Launch pagerank kernel 2
        pagerank2 <<<grid, threads>>>(pagerank1_d, pagerank2_d, num_nodes);
    }
    cudaThreadSynchronize();

    double timer4 = gettime();

    // Copy the rank buffer back
    err = cudaMemcpy(pagerank_array, pagerank1_d, num_nodes * sizeof(float), cudaMemcpyDeviceToHost);

    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMemcpy() failed (%s)\n", cudaGetErrorString(err));
        return -1;
    }

#ifdef GEM5_FUSION
    m5_work_end(0, 0);
#endif

    double timer2 = gettime();

    // Report timing characteristics
    printf("kernel time = %lf ms\n", (timer4 - timer3) * 1000);
    printf("kernel + memcpy time = %lf ms\n", (timer2 - timer1) * 1000);

#if 1
    // Print rank array
    print_vectorf(pagerank_array, num_nodes);
#endif

    // Free the host-side arrays
    free(pagerank_array);
    free(pagerank_array2);
    csr->freeArrays();
    free(csr);

    // Free the device buffers
    cudaFree(row_d);
    cudaFree(col_d);
    cudaFree(data_d);

    cudaFree(pagerank1_d);
    cudaFree(pagerank2_d);

    return 0;

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

