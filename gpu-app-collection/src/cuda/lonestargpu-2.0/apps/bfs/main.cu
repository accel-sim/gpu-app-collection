/** Breadth-first search -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2013, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 *
 * @section Description
 *
 * Example breadth-first search application for demoing Galois system.
 *
 * @author Rupesh Nasre <nasre@ices.utexas.edu>
*  @author Sreepathi Pai <sreepai@ices.utexas.edu>
 */

#include "lonestargpu.h"
#include "variants.h"
#include "cutil_subset.h"

__global__
void dverifysolution(foru *dist, Graph graph, unsigned *nerr) {
	unsigned int nn = (blockIdx.x * blockDim.x + threadIdx.x);
	  if (nn < graph.nnodes) {
		unsigned int nsrcedges = graph.getOutDegree(nn);
		for (unsigned ii = 0; ii < nsrcedges; ++ii) {
			unsigned int u = nn;
			unsigned int v = graph.getDestination(u, ii);
			foru wt = 1;

			if (wt > 0 && dist[u] + wt < dist[v]) {
			  //printf("%d %d %d %d\n", u, v, dist[u], dist[v]);
			  ++*nerr;
			}
		}
	  }	
}

// __global__
// void dverifysolution(foru *dist, Graph graph, unsigned *nerr) {
// 	unsigned int nn = (blockIdx.x * blockDim.x + threadIdx.x);
// 	  if (nn < graph.nnodes) {
// 		unsigned int nsrcedges = graph.getOutDegree(nn);
// 		for (unsigned ii = 0; ii < nsrcedges; ++ii) {
// 			unsigned int u = nn;
// 			unsigned int v = graph.getDestination(u, ii);
// 			foru wt = 1;

// 			// if(dist[v] == dist[u] && dist[v] == 0)
// 			//   {
// 			//     // uninitialized dist?
// 			//     ++*nerr; // should be atomic!
// 			//     return;
// 			//   }

// 			if (wt > 0 && dist[u] + wt < dist[v]) {
// 			  ++*nerr; // should be atomic!
// 			}
// 		}
// 	  }	
// }

void write_solution(const char *fname, Graph &graph, foru *dist)
{
  foru *h_dist;
  h_dist = (foru *) malloc(graph.nnodes * sizeof(foru));
  assert(h_dist != NULL);

  CUDA_SAFE_CALL(cudaMemcpy(h_dist, dist, graph.nnodes * sizeof(foru), cudaMemcpyDeviceToHost));
  
  printf("Writing solution to %s\n", fname);
  FILE *f = fopen(fname, "w");
  // formatted like Merrill's code for comparison
  fprintf(f, "Computed solution (source dist): [");

  for(int node = 0; node < graph.nnodes; node++)
    {
      fprintf(f, "%d:%d\n ", node, h_dist[node]);
    }

  fprintf(f, "]");

  free(h_dist);
}

int main(int argc, char *argv[]) {
	unsigned intzero = 0;
	Graph hgraph, graph;
	unsigned *nerr, hnerr;
	KernelConfig kconf;
	foru *dist;

	if (argc != 2) {
		printf("Usage: %s <graph>\n", argv[0]);
		exit(1);
	}

	cudaSetDeviceFlags(cudaDeviceMapHost);
	cudaGetLastError();

	hgraph.read(argv[1]);
	hgraph.cudaCopy(graph);

	if (cudaMalloc((void **)&dist, graph.nnodes * sizeof(foru)) != cudaSuccess) CudaTest("allocating dist failed");
	cudaMemset(dist, 0, graph.nnodes * sizeof(foru));

#if VARIANT==BFS_MERRILL
	bfs_merrill(graph, dist);
#else
	bfs(graph, dist);
#endif
	if (cudaMalloc((void **)&nerr, sizeof(unsigned)) != cudaSuccess) CudaTest("allocating nerr failed");

	CUDA_SAFE_CALL(cudaMemcpy(nerr, &intzero, sizeof(intzero), cudaMemcpyHostToDevice));

	kconf.setProblemSize(graph.nnodes);
	kconf.setMaxThreadsPerBlock();
	printf("verifying.\n");
	dverifysolution<<<kconf.getNumberOfBlocks(), kconf.getNumberOfBlockThreads()>>> (dist, graph, nerr);
	CudaTest("dverifysolution failed");
	CUDA_SAFE_CALL(cudaMemcpy(&hnerr, nerr, sizeof(hnerr), cudaMemcpyDeviceToHost));
	printf("\tno of errors = %d.\n", hnerr);

	write_solution("bfs-output.txt", graph, dist);

	// cleanup left to the OS.

	return 0;
}
