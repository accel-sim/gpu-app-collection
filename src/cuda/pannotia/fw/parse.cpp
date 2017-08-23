
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Test value
bool test_value(int* array, int dim, int i, int j)
{

    // TODO: Current does not support multiple edges between two vertices
    if (array[i * dim + j] != -1) {
        // fprintf(stderr, "Possibly duplicate records at (%d, %d)\n", i, j);
        return 0;
    } else
        return 1;
}

// Set value (i, j) = value
void set_value(int* array, int dim, int i, int j, int value)
{
    array[i * dim + j] = value;
}

int* parse_graph_file(int *num_nodes, int *num_edges, char* tmpchar)
{

    int *adjmatrix;
    int cnt = 0;
    unsigned int lineno = 0;
    char line[128], sp[2], a, p;

    FILE *fptr;

    fptr = fopen(tmpchar, "r");

    if (!fptr) {
        fprintf(stderr, "Error when opening file: %s\n", tmpchar);
        exit(1);
    }

    printf("Opening file: %s\n", tmpchar);

    while (fgets(line, 100, fptr)) {
        int head, tail, weight;
        long long unsigned size;
        switch (line[0]) {
        case 'c':
            break;
        case 'p':
            sscanf(line, "%c %s %d %d", &p, sp, num_nodes, num_edges);
            printf("Read from file: num_nodes = %d, num_edges = %d\n", *num_nodes, *num_edges);
            size = (long long unsigned)(*num_nodes + 1) * (long long unsigned)(*num_nodes + 1);
            if (size > UINT_MAX) {
                fclose(fptr);
                fprintf(stderr, "ERROR: Too many nodes, huge adjacency matrix\n");
                exit(0);
            }
            adjmatrix = (int *)malloc(size * sizeof(int));
            memset(adjmatrix, -1 , size * sizeof(int));
            break;
        case 'a':
            sscanf(line, "%c %d %d %d", &a, &head, &tail, &weight);
            if (tail == head) printf("reporting self loop\n");
            if (test_value(adjmatrix, *num_nodes + 1, head, tail)) {
                set_value(adjmatrix, *num_nodes + 1, head, tail, weight);
                cnt++;
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

    *num_edges = cnt;
    printf("Actual added edges: %d\n", cnt);

    fclose(fptr);

    return adjmatrix;

}
