/** Single source shortest paths -*- C++ -*-
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
 * Single source shortest paths.
 *
 * @author Rupesh Nasre <nasre@ices.utexas.edu>
 * @author Sreepathi Pai <sreepai@ices.utexas.edu>
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
			foru wt = graph.getWeight(u, ii);
			if (wt > 0 && dist[u] + wt < dist[v]) {
				++*nerr;
			}
		}
	  }	
}

void print_output(const char *filename, foru *hdist, foru *dist, Graph graph)
{
  CUDA_SAFE_CALL(cudaMemcpy(hdist, dist, graph.nnodes * sizeof(foru), cudaMemcpyDeviceToHost));

  printf("Writing output to %s\n", filename);
  FILE *o = fopen(filename, "w");

  for(int i = 0; i < graph.nnodes; i++) {
    fprintf(o, "%d: %d\n", i, hdist[i]);
  }

  fclose(o);
}

int main(int argc, char *argv[]) {
	foru *dist, *hdist;
	unsigned intzero = 0;
	Graph hgraph, graph;
	unsigned *nerr, hnerr;
	KernelConfig kconf;

	cudaFuncSetCacheConfig(drelax, cudaFuncCachePreferShared);
	if (argc != 2) {
		printf("Usage: %s <graph>\n", argv[0]);
		exit(1);
	}

	cudaGetLastError();

	hgraph.read(argv[1]);
	//hgraph.optimize();


	long unsigned totalcommu = hgraph.cudaCopy(graph);

	kconf.setProblemSize(graph.nnodes);

	if (cudaMalloc((void **)&nerr, sizeof(unsigned)) != cudaSuccess) CudaTest("allocating nerr failed");
	if (cudaMalloc((void **)&dist, graph.nnodes * sizeof(foru)) != cudaSuccess) CudaTest("allocating dist failed");
	hdist = (foru *)malloc(graph.nnodes * sizeof(foru));

	kconf.setMaxThreadsPerBlock();
	printf("initializing.\n");
	initialize <<<kconf.getNumberOfBlocks(), kconf.getNumberOfBlockThreads()>>> (dist, graph.nnodes);
	CudaTest("initializing failed");

	sssp(hdist, dist, graph, totalcommu);

	cudaMemcpy(nerr, &intzero, sizeof(intzero), cudaMemcpyHostToDevice);
	kconf.setMaxThreadsPerBlock();
	printf("verifying.\n");
	dverifysolution<<<kconf.getNumberOfBlocks(), kconf.getNumberOfBlockThreads()>>> (dist, graph, nerr);
	CudaTest("dverifysolution failed");
	cudaMemcpy(&hnerr, nerr, sizeof(hnerr), cudaMemcpyDeviceToHost);
	printf("\tno of errors = %d.\n", hnerr);
	
	print_output("sssp-output.txt", hdist, dist, graph);
	// cleanup left to the OS.

	return 0;
}
