#pragma once
#define SSSP_VARIANT "lonestar"
#include "cutil_subset.h"

__global__
void initialize(foru *dist, unsigned int nv) {
	unsigned int ii = blockIdx.x * blockDim.x + threadIdx.x;
	if (ii < nv) {
		dist[ii] = MYINFINITY;
	}
}

__device__
bool processedge(foru *dist, Graph &graph, unsigned src, unsigned ii, unsigned &dst) {
	//unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	dst = graph.getDestination(src, ii);
	if (dst >= graph.nnodes) return 0;

	foru wt = graph.getWeight(src, ii);
	if (wt >= MYINFINITY) return 0;

	foru altdist = dist[src] + wt;
	if (altdist < dist[dst]) {
	 	foru olddist = atomicMin(&dist[dst], altdist);
		if (altdist < olddist) {
			return true;
		} 
		// someone else updated distance to a lower value.
	}
	return false;
}
__device__
bool processnode(foru *dist, Graph &graph, unsigned work) {
	//unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned nn = work;
	if (nn >= graph.nnodes) return 0;
	bool changed = false;
	
	unsigned neighborsize = graph.getOutDegree(nn);
	for (unsigned ii = 0; ii < neighborsize; ++ii) {
		unsigned dst = graph.nnodes;
		foru olddist = processedge(dist, graph, nn, ii, dst);
		if (olddist) {
			changed = true;
		}
	}
	return changed;
}

__global__
void drelax(foru *dist, Graph graph, bool *changed) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned start = id * (MAXBLOCKSIZE / blockDim.x), end = (id + 1) * (MAXBLOCKSIZE / blockDim.x);

	for (unsigned ii = start; ii < end; ++ii) {
		if (processnode(dist, graph, ii)) {
			*changed = true;
		}
	}
}

void sssp(foru *hdist, foru *dist, Graph &graph, long unsigned totalcommu)
{
	foru foruzero = 0.0;
	bool *changed, hchanged;
	int iteration = 0;
	double starttime, endtime;
	KernelConfig kconf;

	kconf.setProblemSize(graph.nnodes);
	cudaMemcpy(&dist[0], &foruzero, sizeof(foruzero), cudaMemcpyHostToDevice);

	if (cudaMalloc((void **)&changed, sizeof(bool)) != cudaSuccess) CudaTest("allocating changed failed");

	printf("solving.\n");
	starttime = rtclock();
	do {
		++iteration;
		hchanged = false;

		cudaMemcpy(changed, &hchanged, sizeof(hchanged), cudaMemcpyHostToDevice);

		drelax <<<kconf.getNumberOfBlocks(), kconf.getNumberOfBlockThreads()>>> (dist, graph, changed);
		CudaTest("solving failed");

		CUDA_SAFE_CALL(cudaMemcpy(&hchanged, changed, sizeof(hchanged), cudaMemcpyDeviceToHost));
	} while (hchanged);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	endtime = rtclock(); // changed from lsg (for now) which included memcopies of graph too.

	CUDA_SAFE_CALL(cudaMemcpy(hdist, dist, graph.nnodes * sizeof(foru), cudaMemcpyDeviceToHost));
	totalcommu += graph.nnodes * sizeof(foru);
	
	printf("\titerations = %d communication = %.3lf MB.\n", iteration, totalcommu * 1.0 / 1000000);

	printf("\truntime [%s] = %f ms\n", SSSP_VARIANT, 1000 * (endtime - starttime));
}
