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



#define BIGNUM 99999999

/**
* init kernel
* @param s_array   set array
* @param c_array   status array
* @param cu_array  status update array
* @param num_nodes number of vertices
* @param num_edges number of edges
*/
__global__ void
init(int *s_array, int *c_array, int *cu_array, int num_nodes, int num_edges)
{
    // Get my workitem id
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < num_nodes) {
        // Set the status array: not processed
        c_array[tid] = -1;
        cu_array[tid] = -1;
        s_array[tid] = 0;
    }
}

/**
* mis1 kernel
* @param row          csr pointer array
* @param col          csr column index array
* @param node_value   node value array
* @param s_array      set array
* @param c_array node status array
* @param min_array    node value array
* @param stop node    value array
* @param num_nodes    number of vertices
* @param num_edges    number of edges
*/
__global__ void
mis1(int *row, int *col, int *node_value, int *s_array, int *c_array,
     int *min_array, int *stop, int num_nodes, int num_edges)
{
    // Get workitem id
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < num_nodes) {
        // If the vertex is not processed
        if (c_array[tid] == -1) {
            *stop = 1;
            // Get the start and end pointers
            int start = row[tid];
            int end;
            if (tid + 1 < num_nodes) {
                end = row[tid + 1];
            } else {
                end = num_edges;
            }

            // Navigate the neighbor list and find the min
            int min = BIGNUM;
            for (int edge = start; edge < end; edge++) {
                if (c_array[col[edge]] == -1) {
                    if (node_value[col[edge]] < min) {
                        min = node_value[col[edge]];
                    }
                }
            }
            min_array[tid] = min;
        }
    }
}

/**
* mis2 kernel
* @param row          csr pointer array
* @param col          csr column index array
* @param node_value   node value array
* @param s_array      set array
* @param c_array      status array
* @param cu_array     status update array
* @param min_array    node value array
* @param num_nodes    number of vertices
* @param num_edges    number of edges
*/
__global__ void
mis2(int *row, int *col, int *node_value, int *s_array, int *c_array,
     int *cu_array, int *min_array, int num_nodes, int num_edges)
{
    // Get my workitem id
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < num_nodes) {
        if (node_value[tid] <= min_array[tid]  && c_array[tid] == -1) {
            // -1: Not processed -2: Inactive 2: Independent set
            // Put the item into the independent set
            s_array[tid] = 2;

            // Get the start and end pointers
            int start = row[tid];
            int end;

            if (tid + 1 < num_nodes) {
                end = row[tid + 1];
            } else {
                end = num_edges;
            }

            // Set the status to inactive
            c_array[tid] = -2;

            // Mark all the neighbors inactive
            for (int edge = start; edge < end; edge++) {
                if (c_array[col[edge]] == -1) {
                    //use status update array to avoid race
                    cu_array[col[edge]] = -2;
                }
            }
        }
    }
}

/**
* mis3 kernel
* @param cu_array     status update array
* @param  c_array     status array
* @param num_nodes    number of vertices
*/
__global__ void
mis3(int *cu_array, int *c_array, int num_nodes)
{
    //get my workitem id
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    //set the status array
    if (tid < num_nodes && cu_array[tid] == -2) {
        c_array[tid] = cu_array[tid];
    }
}


