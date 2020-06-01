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

#define BLOCK_SIZE 16

/**
 * @brief   floydwarshall -- diagonal block
 * @param   dist       distance array
 * @param   blk_iter   block iteration
 * @param   dim        matrix dim
 */
__global__ void
floydwarshall_dia_block(int *dist, int blk_iter, int dim)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int base_x = blk_iter * BLOCK_SIZE;
    int base_y = blk_iter * BLOCK_SIZE;
    int base   = base_y * dim + base_x;

    __shared__ int dia_block[BLOCK_SIZE * BLOCK_SIZE];

    dia_block[ty * BLOCK_SIZE + tx] = dist[base + ty * dim + tx];

    __syncthreads();

    for (int k = 0; k < BLOCK_SIZE; k++) {
        if (dia_block[ty * BLOCK_SIZE + k] + dia_block[k * BLOCK_SIZE + tx] < dia_block[ty * BLOCK_SIZE + tx]) {
            dia_block[ty * BLOCK_SIZE + tx] = dia_block[ty * BLOCK_SIZE + k] + dia_block[k * BLOCK_SIZE + tx];
        }
        __syncthreads();
    }

    dist[base + ty * dim + tx] = dia_block[ty * BLOCK_SIZE + tx];
}

/**
 * @brief   floydwarshall -- a strip of blocks (x-dim)
 * @param   dist       distance array
 * @param   blk_iter   block iteration
 * @param   dim        matrix dim
 */
__global__ void
floydwarshall_strip_blocks_x(int *dist, int blk_iter, int dim)
{
    int bx = blockIdx.x;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    __shared__ int dia_block[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ int strip_block[BLOCK_SIZE * BLOCK_SIZE];

    if (bx != blk_iter) {

        int base_x = blk_iter * BLOCK_SIZE;
        int base_y = blk_iter * BLOCK_SIZE;
        int base   = base_y * dim + base_x;

        dia_block[ty * BLOCK_SIZE + tx] = dist[base + ty * dim + tx];

        __syncthreads();

        int strip_base_y = blk_iter * BLOCK_SIZE;
        int strip_base   = strip_base_y * dim;

        int index = strip_base + ty * dim + bx * BLOCK_SIZE + tx;

        strip_block[ty * BLOCK_SIZE + tx] = dist[index];

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; k++) {
            if (dia_block[ty * BLOCK_SIZE + k] + strip_block[k * BLOCK_SIZE + tx] < strip_block[ty * BLOCK_SIZE + tx]) {
                strip_block[ty * BLOCK_SIZE + tx] = dia_block[ty * BLOCK_SIZE + k] + strip_block[k * BLOCK_SIZE + tx];
            }
            __syncthreads();
        }

        dist[index] = strip_block[ty * BLOCK_SIZE + tx];
    }
}

/**
 * @brief   floydwarshall -- a strip of blocks (y-dim)
 * @param   dist       distance array
 * @param   blk_iter   block iteration
 * @param   dim        matrix dim
 */
__global__ void
floydwarshall_strip_blocks_y(int *dist, int blk_iter, int dim)
{
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    __shared__ int dia_block[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ int strip_block[BLOCK_SIZE * BLOCK_SIZE];

    if (by != blk_iter) {

        int base_x = blk_iter * BLOCK_SIZE;
        int base_y = blk_iter * BLOCK_SIZE;
        int base   = base_y * dim + base_x;

        dia_block[ty * BLOCK_SIZE + tx] = dist[base + ty * dim + tx];

        __syncthreads();

        int strip_base_x = blk_iter * BLOCK_SIZE;
        int strip_base   = strip_base_x;

        int index = strip_base + (by * BLOCK_SIZE + ty) * dim + tx;

        strip_block[ty * BLOCK_SIZE + tx] = dist[index];

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; k++) {
            if (strip_block[ty * BLOCK_SIZE + k] + dia_block[k * BLOCK_SIZE + tx] < strip_block[ty * BLOCK_SIZE + tx])
                strip_block[ty * BLOCK_SIZE + tx] = strip_block[ty * BLOCK_SIZE + k] + dia_block[k * BLOCK_SIZE + tx];
            __syncthreads();
        }

        dist[index] = strip_block[ty * BLOCK_SIZE + tx];
    }
}

/**
 * @brief   floydwarshall -- the remaining blocks
 * @param   dist       distance array
 * @param   blk_iter   block iteration
 * @param   dim        matrix dim
 */
__global__ void
floydwarshall_remaining_blocks(int *dist, int blk_iter, int dim)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    __shared__ int block_y_iter[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ int block_iter_x[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ int strip_block[BLOCK_SIZE * BLOCK_SIZE];

    if (by != blk_iter && bx != blk_iter) {
        int base_Y_iter_y = by * BLOCK_SIZE;
        int base_Y_iter_x = blk_iter * BLOCK_SIZE;
        int base_Y = base_Y_iter_y * dim + base_Y_iter_x;
        block_y_iter[ty * BLOCK_SIZE + tx] = dist[base_Y + ty * dim + tx];

        __syncthreads();

        int base_X_iter_y = blk_iter * BLOCK_SIZE;
        int base_X_iter_x = bx * BLOCK_SIZE;
        int base_X = base_X_iter_y * dim + base_X_iter_x;

        block_iter_x[ty * BLOCK_SIZE + tx] = dist[base_X + ty * dim + tx];
        __syncthreads();

        int index = dim * BLOCK_SIZE * by + BLOCK_SIZE * bx + dim * ty + tx;
        strip_block[ty * BLOCK_SIZE + tx] = dist[index];
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; k++) {
            if (block_y_iter[ty * BLOCK_SIZE + k] + block_iter_x[k * BLOCK_SIZE + tx] < strip_block[ty * BLOCK_SIZE + tx]) {
                strip_block[ty * BLOCK_SIZE + tx] = block_y_iter[ty * BLOCK_SIZE + k] + block_iter_x[k * BLOCK_SIZE + tx];
            }
            __syncthreads();
        }

        dist[index] = strip_block[ty * BLOCK_SIZE + tx];
    }
}
