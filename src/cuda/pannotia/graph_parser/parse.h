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

#include <stdlib.h>

typedef struct csr_arrays_t {
    int *row_array;
    int *col_array;
    int *data_array;
    int *col_cnt;

    void freeArrays() {
        if (row_array) {
            free(row_array);
            row_array = NULL;
        }
        if (col_array) {
            free(col_array);
            col_array = NULL;
        }
        if (data_array) {
            free(data_array);
            data_array = NULL;
        }
        if (col_cnt) {
            free(col_cnt);
            col_cnt = NULL;
        }
    }
} csr_array;

typedef struct ell_arrays_t {
    int max_height;
    int num_nodes;
    int *col_array;
    int *data_array;
    int *col_cnt;
} ell_array;

typedef struct double_edges_t {
    int *edge_array1;
    int *edge_array2;
} double_edges;

typedef struct cooedgetuple {
    int row;
    int col;
    int val;
} CooTuple;

csr_array *parseCOO(char* tmpchar, int *p_num_nodes, int *p_num_edges, bool directed);
csr_array *parseMetis(char* tmpchar, int *p_num_nodes, int *p_num_edges, bool directed);
csr_array *parseMM(char* tmpchar, int *p_num_nodes, int *p_num_edges, bool directed, bool weight_flag);
ell_array *csr2ell(csr_array *csr, int num_nodes, int num_edges, int fill);

double_edges *parseCOO_doubleEdge(char* tmpchar, int *p_num_nodes, int *p_num_edges, bool directed);
double_edges *parseMetis_doubleEdge(char* tmpchar, int *p_num_nodes, int *p_num_edges, bool directed);

csr_array *parseCOO_transpose(char* tmpchar, int *p_num_nodes, int *p_num_edges, bool directed);
csr_array *parseMetis_transpose(char* tmpchar, int *p_num_nodes, int *p_num_edges, bool directed);
