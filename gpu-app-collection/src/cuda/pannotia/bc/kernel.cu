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

/**
 * @brief   Breadth-first traversal
 * @param   row       CSR pointer array
 * @param   col       CSR column  array
 * @param   d         Distance array
 * @param   rho       Rho array
 * @param   p         Dependency array
 * @param   cont      Termination variable
 * @param   num_nodes Termination variable
 * @param   num_edges Termination variable
 * @param   dist      Current traversal layer
 */

__global__ void
bfs_kernel(int *row, int *col, int *d, float *rho, int *cont,
           const int num_nodes, const int num_edges, const int dist)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    //navigate the current layer
    if (tid < num_nodes && d[tid] == dist) {

        //get the starting and ending pointers
        //of the neighbor list

        int start = row[tid];
        int end;
        if (tid + 1 < num_nodes)
            end = row[tid + 1];
        else
            end = num_edges;

        //navigate through the neighbor list
        for (int edge = start; edge < end; edge++) {
            int w = col[edge];
            if (d[w] < 0) {
                *cont = 1;
                //traverse another layer
                d[w] = dist + 1;
            }
            //transfer the rho value to the neighbor
            if (d[w] == (dist + 1)) {
                atomicAdd(&rho[w], rho[tid]);
            }
        }
    }
}

/**
 * @brief   Back traversal
 * @param   row       CSR pointer array
 * @param   col       CSR column  array
 * @param   d         Distance array
 * @param   rho       Rho array
 * @param   sigma     Sigma array
 * @param   p         Dependency array
 * @param   cont      Termination variable
 * @param   num_nodes Termination variable
 * @param   num_edges Termination variable
 * @param   dist      Current traversal layer
 * @param   s         Source vertex
 * @param   bc        Betweeness Centrality array
 */

__global__ void
backtrack_kernel(int *row, int *col, int *d, float *rho, float *sigma,
                 const int num_nodes, const int num_edges, const int dist,
                 const int s, float* bc)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Navigate the current layer
    if (tid < num_nodes && d[tid] == dist - 1) {

        int start = row[tid];
        int end;
        if (tid + 1 < num_nodes)
            end = row[tid + 1];
        else
            end = num_edges;

        // Get the starting and ending pointers
        // of the neighbor list in the reverse graph
        for (int edge = start; edge < end; edge++) {
            int w = col[edge];
            // Update the sigma value traversing back
            if (d[w] == dist - 2)
                atomicAdd(&sigma[w], rho[w] / rho[tid] * (1 + sigma[tid]));
        }

        // Update the BC value
        if (tid != s)
            bc[tid] = bc[tid] + sigma[tid];
    }

}

/**
 * @brief   back_sum_kernel (not used)
 * @param   s         Source vertex
 * @param   dist      Current traversal layer
 * @param   d         Distance array
 * @param   sigma     Sigma array
 * @param   bc        Betweeness Centrality array
 * @param   num_nodes Termination variable
 * @param   num_edges Termination variable
 */
__global__ void
back_sum_kernel(const int s, const int dist, int *d, float *sigma, float *bc,
                const int num_nodes)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_nodes) {
        // If it is not the source
        if (s != tid && d[tid] == dist - 1) {
            bc[tid] = bc[tid] + sigma[tid];
        }
    }
}

/**
 * @brief   array set 1D
 * @param   s           Source vertex
 * @param   dist_array  Distance array
 * @param   sigma       Sigma array
 * @param   rho         Rho array
 * @param   num_nodes Termination variable
 */
__global__ void
clean_1d_array(const int source, int *dist_array, float *sigma, float *rho,
               const int num_nodes)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_nodes) {

        sigma[tid] = 0;

        if (tid == source) {
            // If source vertex rho = 1, dist = 0
            rho[tid] = 1;
            dist_array[tid] = 0;
        } else {
            // If other vertices rho = 0, dist = -1
            rho[tid] = 0;
            dist_array[tid] = -1;
        }
    }
}

/**
 * @brief   array set 2D
 * @param   p           Dependency array
 * @param   num_nodes   Number of vertices
 */
__global__ void clean_2d_array(int *p, const int num_nodes)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_nodes * num_nodes)
        p[tid] = 0;
}

/**
 * @brief   clean BC
 * @param   bc_d        Betweeness Centrality array
 * @param   num_nodes   Number of vertices
 */
__global__ void clean_bc(float *bc_d, const int num_nodes)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_nodes)
        bc_d[tid] = 0;
}
