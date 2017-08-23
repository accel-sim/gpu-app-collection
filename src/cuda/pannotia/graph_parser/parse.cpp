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

#include "parse.h"
#include "stdlib.h"
#include "stdio.h"
#include <string.h>
#include <algorithm>
#include <sys/time.h>
#include "util.h"

bool doCompare(CooTuple elem1, CooTuple elem2)
{
    if (elem1.row < elem2.row) {
        return true;
    }
    return false;
}

ell_array *csr2ell(csr_array *csr, int num_nodes, int num_edges, int fill)
{
    int size, maxheight = 0;
    for (int i = 0; i < num_nodes; i++) {
        size = csr->row_array[i + 1] - csr->row_array[i];
        if (size > maxheight)
            maxheight = size;
    }

    ell_array *ell = (ell_array *)malloc(sizeof(ell_array));
    if (!ell) printf("malloc failed");

    ell->max_height = maxheight;
    ell->num_nodes = num_nodes;

    ell->col_array = (int*)malloc(sizeof(int) * maxheight * num_nodes);
    ell->data_array = (int*)malloc(sizeof(int) * maxheight * num_nodes);


    for (int i = 0; i < maxheight * num_nodes; i++) {
        ell->col_array[i] = 0;
        ell->data_array[i] = fill;
    }

    for (int i = 0; i < num_nodes; i++) {
        int start = csr->row_array[i];
        int end = csr->row_array[i + 1];
        int lastcolid = 0;
        for (int j = start; j < end; j++) {
            int colid = csr->col_array[j];
            int data = csr->data_array[j];
            ell->col_array[i + (j - start) * num_nodes] = colid;
            ell->data_array[i + (j - start) * num_nodes] = data;
            lastcolid = colid;
        }
        for (int j = end; j < start + maxheight; j++) {
            ell->col_array[i + (j - start) * num_nodes] = lastcolid;
            ell->data_array[i + (j - start) * num_nodes] = fill;
        }
    }

    return ell;

}

csr_array *parseMetis(char* tmpchar, int *p_num_nodes, int *p_num_edges, bool directed)
{

    int cnt = 0;
    unsigned int lineno = 0;
    char *line = (char *)malloc(8192);
    int num_edges = 0, num_nodes = 0;

    FILE *fptr;
    CooTuple *tuple_array = NULL;

    fptr = fopen(tmpchar, "r");
    if (!fptr) {
        fprintf(stderr, "Error when opening file: %s\n", tmpchar);
        exit(1);
    }

    printf("Opening file: %s\n", tmpchar);

    while (fgets(line, 8192, fptr)) {
        int head, tail, weight = 0;
        CooTuple temp;

        if (line[0] == '%') continue; // skip comment lines

        if (lineno == 0) { //the first line

            sscanf(line, "%d %d", p_num_nodes, p_num_edges);
            if (!directed) {
                *p_num_edges = *p_num_edges * 2;
                printf("This is an undirected graph\n");
            } else {
                printf("This is a directed graph\n");
            }
            num_nodes = *p_num_nodes;
            num_edges = *p_num_edges;


            printf("Read from file: num_nodes = %d, num_edges = %d\n", num_nodes, num_edges);
            tuple_array = (CooTuple *)malloc(sizeof(CooTuple) * num_edges);
        } else if (lineno > 0) { //from the second line

            char *pch;
            pch = strtok(line , " ,.-");
            while (pch != NULL) {
                head = lineno;
                tail = atoi(pch);
                if (tail <= 0)  break;

                if (tail == head) printf("reporting self loop: %d, %d\n", lineno + 1, lineno);

                temp.row = head - 1;
                temp.col = tail - 1;
                temp.val = weight;

                tuple_array[cnt++] = temp;

                pch = strtok(NULL, " ,.-");

            }
        }

#ifdef VERBOSE
        printf("Adding edge: %d ==> %d ( %d )\n", head, tail, weight);
#endif

        lineno++;

    }

    // Metis files are stored in row-order, so sorting is unnecessary
    // std::stable_sort(tuple_array, tuple_array + num_edges, doCompare);

#ifdef VERBOSE
    for (int i = 0 ; i < num_edges; i++) {
        printf("%d: %d, %d, %d\n", i, tuple_array[i].row, tuple_array[i].col, tuple_array[i].val);
    }
#endif

    int *row_array = (int *)malloc((num_nodes + 1) * sizeof(int));
    int *col_array = (int *)malloc(num_edges * sizeof(int));
    int *data_array = (int *)malloc(num_edges * sizeof(int));

    int row_cnt = 0;
    int prev = -1;
    int idx;
    for (idx = 0; idx < num_edges; idx++) {
        int curr = tuple_array[idx].row;
        if (curr != prev) {
            row_array[row_cnt++] = idx;
            prev = curr;
        }
        col_array[idx] = tuple_array[idx].col;
        data_array[idx] = tuple_array[idx].val;

    }
    row_array[row_cnt] = idx;

    csr_array *csr = (csr_array *)malloc(sizeof(csr_array));
    memset(csr, 0, sizeof(csr_array));
    csr->row_array = row_array;
    csr->col_array = col_array;
    csr->data_array = data_array;

    fclose(fptr);
    free(tuple_array);

    return csr;

}


csr_array *parseCOO(char* tmpchar, int *p_num_nodes, int *p_num_edges, bool directed)
{
    int cnt = 0;
    unsigned int lineno = 0;
    char line[128], sp[2], a, p;
    int num_nodes = 0, num_edges = 0;

    FILE *fptr;
    CooTuple *tuple_array = NULL;

    fptr = fopen(tmpchar, "r");
    if (!fptr) {
        fprintf(stderr, "Error when opening file: %s\n", tmpchar);
        exit(1);
    }

    printf("Opening file: %s\n", tmpchar);

    while (fgets(line, 100, fptr)) {
        int head, tail, weight;
        switch (line[0]) {
        case 'c':
            break;
        case 'p':
            sscanf(line, "%c %s %d %d", &p, sp, p_num_nodes, p_num_edges);

            if (!directed) {
                *p_num_edges = *p_num_edges * 2;
                printf("This is an undirected graph\n");
            } else {
                printf("This is a directed graph\n");
            }

            num_nodes = *p_num_nodes;
            num_edges = *p_num_edges;

            printf("Read from file: num_nodes = %d, num_edges = %d\n", num_nodes, num_edges);
            tuple_array = (CooTuple *)malloc(sizeof(CooTuple) * num_edges);
            break;

        case 'a':
            sscanf(line, "%c %d %d %d", &a, &head, &tail, &weight);
            if (tail == head) printf("reporting self loop\n");
            CooTuple temp;
            temp.row = head - 1;
            temp.col = tail - 1;
            temp.val = weight;
            tuple_array[cnt++] = temp;
            if (!directed) {
                temp.row = tail - 1;
                temp.col = head - 1;
                temp.val = weight;
                tuple_array[cnt++] = temp;
            }

#ifdef VERBOSE
            printf("Adding edge: %d ==> %d ( %d )\n", head, tail, weight);
#endif
            break;
        default:
            fprintf(stderr, "exiting loop\n");
            break;
        }
        lineno++;
    }

    std::stable_sort(tuple_array, tuple_array + num_edges, doCompare);

#ifdef VERBOSE
    for (int i = 0 ; i < num_edges; i++) {
        printf("%d: %d, %d, %d\n", i, tuple_array[i].row, tuple_array[i].col, tuple_array[i].val);
    }
#endif

    int *row_array = (int *)malloc((num_nodes + 1) * sizeof(int));
    int *col_array = (int *)malloc(num_edges * sizeof(int));
    int *data_array = (int *)malloc(num_edges * sizeof(int));

    int row_cnt = 0;
    int prev = -1;
    int idx;
    for (idx = 0; idx < num_edges; idx++) {
        int curr = tuple_array[idx].row;
        if (curr != prev) {
            row_array[row_cnt++] = idx;
            prev = curr;
        }

        col_array[idx] = tuple_array[idx].col;
        data_array[idx] = tuple_array[idx].val;
    }

    row_array[row_cnt] = idx;

    fclose(fptr);
    free(tuple_array);

    csr_array *csr = (csr_array *)malloc(sizeof(csr_array));
    memset(csr, 0, sizeof(csr_array));
    csr->row_array = row_array;
    csr->col_array = col_array;
    csr->data_array = data_array;

    return csr;

}

// Parse Metis file with double edges
double_edges *parseMetis_doubleEdge(char* tmpchar, int *p_num_nodes, int *p_num_edges, bool directed)
{
    int cnt = 0;
    unsigned int lineno = 0;
    char line[4096];
    int num_edges = 0, num_nodes = 0;
    FILE *fptr;
    CooTuple *tuple_array = NULL;

    fptr = fopen(tmpchar, "r");
    if (!fptr) {
        fprintf(stderr, "Error when opening file: %s\n", tmpchar);
        exit(1);
    }

    printf("Opening file: %s\n", tmpchar);

    while (fgets(line, 4096, fptr)) {
        int head, tail, weight = 0;
        CooTuple temp;

        if (line[0] == '%') continue; // skip comment lines

        if (lineno == 0) { //the first line

            sscanf(line, "%d %d", p_num_nodes, p_num_edges);
            if (!directed) {
                *p_num_edges = *p_num_edges * 2;
                printf("This is an undirected graph\n");
            } else {
                printf("This is a directed graph\n");
            }

            num_nodes = *p_num_nodes;
            num_edges = *p_num_edges;

            printf("Read from file: num_nodes = %d, num_edges = %d\n", num_nodes, num_edges);
            tuple_array = (CooTuple *)malloc(sizeof(CooTuple) * num_edges);
            if (!tuple_array) printf("xxxxxxxx\n");

        } else if (lineno > 0) { //from the second line
            char *pch;
            pch = strtok(line , " ,.-");
            while (pch != NULL) {
                head = lineno;
                tail = atoi(pch);
                if (tail <= 0) break;

                if (tail == head) printf("reporting self loop: %d, %d\n", lineno + 1, lineno);

                temp.row = head - 1;
                temp.col = tail - 1;
                temp.val = weight;

                tuple_array[cnt++] = temp;

                pch = strtok(NULL, " ,.-");
            }
        }

#ifdef VERBOSE
        printf("Adding edge: %d ==> %d ( %d )\n", head, tail, weight);
#endif

        lineno++;
    }

    // Metis files are stored in row-order, so sorting is unnecessary
    // std::stable_sort(tuple_array, tuple_array + num_edges, doCompare);

#ifdef VERBOSE
    for (int i = 0 ; i < num_edges; i++) {
        printf("%d: %d, %d, %d\n", i, tuple_array[i].row, tuple_array[i].col, tuple_array[i].val);
    }
#endif

    int *edge_array1 = (int *)malloc(num_edges * sizeof(int));
    int *edge_array2 = (int *)malloc(num_edges * sizeof(int));

    for (int i = 0; i < num_edges; i++) {
        edge_array1[i] = tuple_array[i].row;
        edge_array2[i] = tuple_array[i].col;
    }

    fclose(fptr);
    free(tuple_array);

    double_edges *de = (double_edges *)malloc(sizeof(double_edges));
    de->edge_array1 = edge_array1;
    de->edge_array2 = edge_array2;

    return de;

}

// Parse COO file with double edges
double_edges *parseCOO_doubleEdge(char* tmpchar, int *p_num_nodes, int *p_num_edges, bool directed)
{
    int cnt = 0;
    unsigned int lineno = 0;
    char line[128], sp[2], a, p;
    int num_nodes = 0, num_edges = 0;

    FILE *fptr;
    CooTuple *tuple_array = NULL;

    fptr = fopen(tmpchar, "r");
    if (!fptr) {
        fprintf(stderr, "Error when opening file: %s\n", tmpchar);
        exit(1);
    }

    printf("Opening file: %s\n", tmpchar);

    while (fgets(line, 100, fptr)) {
        int head, tail, weight;
        switch (line[0]) {
        case 'c':
            break;
        case 'p':
            sscanf(line, "%c %s %d %d", &p, sp, p_num_nodes, p_num_edges);

            if (!directed) {
                *p_num_edges = *p_num_edges * 2;
                printf("This is an undirected graph\n");
            } else {
                printf("This is a directed graph\n");
            }

            num_nodes = *p_num_nodes;
            num_edges = *p_num_edges;

            printf("Read from file: num_nodes = %d, num_edges = %d\n", num_nodes, num_edges);
            tuple_array = (CooTuple *)malloc(sizeof(CooTuple) * num_edges);
            break;
        case 'a':
            sscanf(line, "%c %d %d %d", &a, &head, &tail, &weight);
            if (tail == head) printf("reporting self loop\n");
            CooTuple temp;
            temp.row = head - 1;
            temp.col = tail - 1;
            temp.val = weight;
            tuple_array[cnt++] = temp;
            if (!directed) {
                temp.row = tail - 1;
                temp.col = head - 1;
                temp.val = weight;
                tuple_array[cnt++] = temp;
            }

#ifdef VERBOSE
            printf("Adding edge: %d ==> %d ( %d )\n", head, tail, weight);
#endif
            break;
        default:
            fprintf(stderr, "exiting loop\n");
            break;

        }
        lineno++;
    }

    std::stable_sort(tuple_array, tuple_array + num_edges, doCompare);

#ifdef VERBOSE
    for (int i = 0 ; i < num_edges; i++) {
        printf("%d: %d, %d, %d\n", i, tuple_array[i].row, tuple_array[i].col, tuple_array[i].val);
    }
#endif

    int *edge_array1 = (int *)malloc(num_edges * sizeof(int));
    int *edge_array2 = (int *)malloc(num_edges * sizeof(int));

    for (int i = 0; i < num_edges; i++) {
        edge_array1[i] = tuple_array[i].row;
        edge_array2[i] = tuple_array[i].col;
    }

    fclose(fptr);
    free(tuple_array);

    double_edges *de = (double_edges *)malloc(sizeof(double_edges));
    de->edge_array1 = edge_array1;
    de->edge_array2 = edge_array2;

    return de;
}

// Parse matrix market file
csr_array *parseMM(char* tmpchar, int *p_num_nodes, int *p_num_edges, bool directed, bool weight_flag)
{
    int cnt = 0;
    unsigned int lineno = 0;
    char line[128];
    int num_nodes = 0, num_edges = 0, num_nodes2 = 0;

    FILE *fptr;
    CooTuple *tuple_array = NULL;

    fptr = fopen(tmpchar, "r");
    if (!fptr) {
        fprintf(stderr, "Error when opening file: %s\n", tmpchar);
        exit(1);
    }

    printf("Opening file: %s\n", tmpchar);

    while (fgets(line, 100, fptr)) {
        int head, tail, weight;
        if (line[0] == '%') continue;
        if (lineno == 0) {
            sscanf(line, "%d %d %d", p_num_nodes, &num_nodes2, p_num_edges);
            if (!directed) {
                *p_num_edges = *p_num_edges * 2;
                printf("This is an undirected graph\n");
            } else {
                printf("This is a directed graph\n");
            }

            num_nodes = *p_num_nodes;
            num_edges = *p_num_edges;

            printf("Read from file: num_nodes = %d, num_edges = %d\n", num_nodes, num_edges);
            tuple_array = (CooTuple *)malloc(sizeof(CooTuple) * num_edges);
            if (!tuple_array) {
                printf("tuple array not allocated succesfully\n");
                exit(1);
            }

        }
        if (lineno > 0) {

            if (weight_flag) {
                sscanf(line, "%d %d %d", &head, &tail, &weight);
            } else {
                sscanf(line, "%d %d",  &head, &tail);
                printf("(%d, %d)\n", head, tail);
                weight = 0;
            }

            if (tail == head) {
                printf("reporting self loop\n");
                continue;
            };

            CooTuple temp;
            temp.row = head - 1;
            temp.col = tail - 1;
            temp.val = weight;
            tuple_array[cnt++] = temp;

            if (!directed) {
                temp.row = tail - 1;
                temp.col = head - 1;
                temp.val = weight;
                tuple_array[cnt++] = temp;
            }

#ifdef VERBOSE
            printf("Adding edge: %d ==> %d ( %d )\n", head, tail, weight);
#endif
        }
        lineno++;
    }

    std::stable_sort(tuple_array, tuple_array + num_edges, doCompare);

#ifdef VERBOSE
    for (int i = 0 ; i < num_edges; i++) {
        printf("%d: %d, %d, %d\n", i, tuple_array[i].row, tuple_array[i].col, tuple_array[i].val);
    }
#endif

    int *row_array = (int *)malloc((num_nodes + 1) * sizeof(int));
    int *col_array = (int *)malloc(num_edges * sizeof(int));
    int *data_array = (int *)malloc(num_edges * sizeof(int));

    int row_cnt = 0;
    int prev = -1;
    int idx;
    for (idx = 0; idx < num_edges; idx++) {
        int curr = tuple_array[idx].row;
        if (curr != prev) {
            row_array[row_cnt++] = idx;
            prev = curr;
        }

        col_array[idx] = tuple_array[idx].col;
        data_array[idx] = tuple_array[idx].val;
    }
    row_array[row_cnt] = idx;

    fclose(fptr);
    free(tuple_array);

    csr_array *csr = (csr_array *)malloc(sizeof(csr_array));
    memset(csr, 0, sizeof(csr_array));
    csr->row_array = row_array;
    csr->col_array = col_array;
    csr->data_array = data_array;

    return csr;
}

// Parse Metis file with transpose
csr_array *parseMetis_transpose(char* tmpchar, int *p_num_nodes, int *p_num_edges, bool directed)
{
    int cnt = 0;
    unsigned int lineno = 0;
    char *line = (char *)malloc(8192);
    int num_edges = 0, num_nodes = 0;
    int *col_cnt = NULL;

    FILE *fptr;
    CooTuple *tuple_array = NULL;

    fptr = fopen(tmpchar, "r");
    if (!fptr) {
        fprintf(stderr, "Error when opening file: %s\n", tmpchar);
        exit(1);
    }

    printf("Opening file: %s\n", tmpchar);
    while (fgets(line, 8192, fptr)) {
        int head, tail, weight = 0;
        CooTuple temp;

        if (line[0] == '%') continue; // skip comment lines

        if (lineno == 0) { //the first line

            sscanf(line, "%d %d", p_num_nodes, p_num_edges);

            col_cnt = (int *)malloc(*p_num_nodes * sizeof(int));
            if (!col_cnt) {
                printf("memory allocation failed for col_cnt\n");
                exit(1);
            }
            memset(col_cnt, 0, *p_num_nodes * sizeof(int));

            if (!directed) {
                *p_num_edges = *p_num_edges * 2;
                printf("This is an undirected graph\n");
            } else {
                printf("This is a directed graph\n");
            }
            num_nodes = *p_num_nodes;
            num_edges = *p_num_edges;

            printf("Read from file: num_nodes = %d, num_edges = %d\n", num_nodes, num_edges);
            tuple_array = (CooTuple *)malloc(sizeof(CooTuple) * num_edges);
        } else if (lineno > 0) { //from the second line
            char *pch;
            pch = strtok(line , " ,.-");
            while (pch != NULL) {
                head = lineno;
                tail = atoi(pch);
                if (tail <= 0) {
                    break;
                }

                if (tail == head) printf("reporting self loop: %d, %d\n", lineno + 1, lineno);

                if (directed) {
                    temp.row = tail - 1;
                    temp.col = head - 1;
                } else {
                    // Undirected matrices are symmetric, so there is no need
                    // to transpose and then re-sort the edges
                    temp.row = head - 1;
                    temp.col = tail - 1;
                }
                temp.val = weight;

                col_cnt[head - 1]++;
                if (cnt >= num_edges) {
                    fprintf(stderr, "Error when opening file: %s.\n" \
                            "    Check if graph is undirected Metis format\n", tmpchar);
                    exit(1);
                }
                tuple_array[cnt++] = temp;

                pch = strtok(NULL, " ,.-");
            }
        }
#ifdef VERBOSE
        printf("Adding edge: %d ==> %d ( %d )\n", head, tail, weight);
#endif
        lineno++;
    }

    if (directed) {
        // Metis files are stored in row-order, so transposed, directed
        // matrices must be re-sorted!
        std::stable_sort(tuple_array, tuple_array + num_edges, doCompare);
    }

#ifdef VERBOSE
    for (int i = 0 ; i < num_edges; i++) {
        printf("%d: %d, %d, %d\n", i, tuple_array[i].row, tuple_array[i].col, tuple_array[i].val);
    }
#endif

    int *row_array = (int *)malloc((num_nodes + 1) * sizeof(int));
    int *col_array = (int *)malloc(num_edges * sizeof(int));
    int *data_array = (int *)malloc(num_edges * sizeof(int));

    int row_cnt = 0;
    int prev = -1;
    int idx;
    for (idx = 0; idx < num_edges; idx++) {
        int curr = tuple_array[idx].row;
        if (curr != prev) {
            row_array[row_cnt++] = idx;
            prev = curr;
        }
        col_array[idx] = tuple_array[idx].col;
        data_array[idx] = tuple_array[idx].val;
    }
    row_array[row_cnt] = idx;

    csr_array *csr = (csr_array *)malloc(sizeof(csr_array));
    memset(csr, 0, sizeof(csr_array));
    csr->row_array = row_array;
    csr->col_array = col_array;
    csr->data_array = data_array;
    csr->col_cnt = col_cnt;

    fclose(fptr);
    free(tuple_array);

    return csr;
}

// Parse COO file with transpose
csr_array *parseCOO_transpose(char* tmpchar, int *p_num_nodes, int *p_num_edges, bool directed)
{
    int cnt = 0;
    unsigned int lineno = 0;
    char line[128], sp[2], a, p;
    int num_nodes = 0, num_edges = 0;

    FILE *fptr;
    CooTuple *tuple_array = NULL;

    fptr = fopen(tmpchar, "r");
    if (!fptr) {
        fprintf(stderr, "Error when opening file: %s\n", tmpchar);
        exit(1);
    }

    printf("Opening file: %s\n", tmpchar);

    while (fgets(line, 100, fptr)) {
        int head, tail, weight;
        switch (line[0]) {
        case 'c':
            break;
        case 'p':
            fflush(stdout);

            sscanf(line, "%c %s %d %d", &p, sp, p_num_nodes, p_num_edges);

            if (!directed) {
                *p_num_edges = *p_num_edges * 2;
                printf("This is an undirected graph\n");
            } else {
                printf("This is a directed graph\n");
            }

            num_nodes = *p_num_nodes;
            num_edges = *p_num_edges;

            printf("Read from file: num_nodes = %d, num_edges = %d\n", num_nodes, num_edges);
            tuple_array = (CooTuple *)malloc(sizeof(CooTuple) * num_edges);
            break;

        case 'a':
            sscanf(line, "%c %d %d %d", &a, &head, &tail, &weight);
            if (tail == head) printf("reporting self loop\n");
            CooTuple temp;
            temp.val = weight;
            temp.row = tail - 1;
            temp.col = head - 1;
            tuple_array[cnt++] = temp;
            if (!directed) {
                temp.val = weight;
                temp.row = tail - 1;
                temp.col = head - 1;
                tuple_array[cnt++] = temp;
            }

#ifdef VERBOSE
            printf("Adding edge: %d ==> %d ( %d )\n", head, tail, weight);
#endif
            break;
        default:
            fprintf(stderr, "exiting loop\n");
            break;
        }
        lineno++;
    }

    std::stable_sort(tuple_array, tuple_array + num_edges, doCompare);

#ifdef VERBOSE
    for (int i = 0 ; i < num_edges; i++) {
        printf("%d: %d, %d, %d\n", i, tuple_array[i].row, tuple_array[i].col, tuple_array[i].val);
    }
#endif

    int *row_array = (int *)malloc((num_nodes + 1) * sizeof(int));
    int *col_array = (int *)malloc(num_edges * sizeof(int));
    int *data_array = (int *)malloc(num_edges * sizeof(int));

    int row_cnt = 0;
    int prev = -1;
    int idx;
    for (idx = 0; idx < num_edges; idx++) {
        int curr = tuple_array[idx].row;
        if (curr != prev) {
            row_array[row_cnt++] = idx;
            prev = curr;
        }
        col_array[idx] = tuple_array[idx].col;
        data_array[idx] = tuple_array[idx].val;
    }
    while (row_cnt <= num_nodes) {
        row_array[row_cnt++] = idx;
    }

    csr_array *csr = (csr_array *)malloc(sizeof(csr_array));
    memset(csr, 0, sizeof(csr_array));
    csr->row_array = row_array;
    csr->col_array = col_array;
    csr->data_array = data_array;

    fclose(fptr);
    free(tuple_array);

    return csr;
}

