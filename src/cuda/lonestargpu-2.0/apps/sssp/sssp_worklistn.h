#define SSSP_VARIANT "worklistN"

/* 
 * copied from wl6, corrects pushRange in worklist7.
 */

#include "worklist.h"
#include "cutil_subset.h"

#define WORKPERTHREAD	1

unsigned int NVERTICES;

__global__
void initialize(foru *dist, unsigned int nv) {
	unsigned int ii = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("ii=%d, nv=%d.\n", ii, *nv);
	if (ii < nv) {
		dist[ii] = MYINFINITY;
	}
}


__device__
foru processedge(foru *dist, Graph &graph, unsigned src, unsigned ii, unsigned &dst) {
	//unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	dst = graph.getDestination(src, ii);
	if (dst >= graph.nnodes) return 0;

	foru wt = graph.getWeight(src, ii);
	if (wt >= MYINFINITY) return 0;

	 foru altdist = dist[src] + wt;
	 if (altdist < dist[dst]) {
	 	foru olddist = atomicMin(&dist[dst], altdist);
		if (altdist < olddist) {
			return olddist;
		} 
		// someone else updated distance to a lower value.
	 }
	  return 0;
}
__device__
unsigned processnode(foru *dist, Graph &graph, Worklist &outwl, unsigned work) {
	//unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned nn = work;	// wl.getItem(wlii);
	if (nn >= graph.nnodes) return 0;
	
	unsigned neighborsize = graph.getOutDegree(nn);
	#define PERTHREAD	8
	//unsigned newwork[PERTHREAD];
	//unsigned *newwork = new unsigned[PERTHREAD];
	//unsigned mystart = 0;
	//unsigned nwii = mystart;
	/*__shared__ unsigned newwork[PERTHREAD*BLOCKSIZE];
	unsigned mystart = PERTHREAD*threadIdx.x;
	unsigned nwii = mystart;
	*/

	for (unsigned ii = 0; ii < neighborsize; ++ii) {
		unsigned dst = graph.nnodes;
		foru olddist = processedge(dist, graph, nn, ii, dst);
		if (olddist) {
			if (outwl.push(dst)) {	// buffer oveflow.
				// critical: reset the distance, since this node couldn't be added to the worklist.
				dist[dst] = olddist;	// no atomicMax required.
				return 1;
			}
		}
	}
	//free(newwork);
	return 0;
}

__global__
void drelax(foru *dist, Graph graph, Worklist inwl, Worklist outwl, unsigned *gerrno) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	//if (id == 0) { printf("\tstart=%d, end=%d.\n", *inwl.start, *inwl.end); }
	unsigned start, end;
	inwl.myItems(start, end);
	unsigned work;

	if (id == 0) {
		//printf("\t start=%d, end=%d.\n", start, end);
	}
	for (unsigned ii = start; ii < end; ++ii) {
		//wl.pop(work);
		work = inwl.getItem(ii);
		//if (work == MYNODE) { printf("relax: dist[%d] = %u.\n", work, dist[work]); }
		if (processnode(dist, graph, outwl, work)) {	// buffer oveflow.
			*gerrno = 1;
			return;
		}
	}
}
#define SWAP(a, b)	{ tmp = a; a = b; b = tmp; }

void sssp(foru *hdist, foru *dist, Graph &graph, long unsigned totalcommu)

{
	foru foruzero = 0.0;
	unsigned intzero = 0;
	unsigned int NBLOCKS, FACTOR = 128;
	bool *changed;
	int iteration = 0;
	Worklist inwl, outwl, *inwlptr, *outwlptr, *tmp;
	unsigned *nerr, hnerr;

	double starttime, endtime;
	double runtime;

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	NBLOCKS = deviceProp.multiProcessorCount;

	NVERTICES = graph.nnodes;

	FACTOR = (NVERTICES + MAXBLOCKSIZE * NBLOCKS - 1) / (MAXBLOCKSIZE * NBLOCKS);

	//printf("computing stats...\n");
	//graph.printStats();

	cudaMemcpy(&dist[0], &foruzero, sizeof(foruzero), cudaMemcpyHostToDevice);

	if (cudaMalloc((void **)&changed, sizeof(bool)) != cudaSuccess) CudaTest("allocating changed failed");

	unsigned wlsz = 0;

	inwl.ensureSpace(graph.nnodes / 5);
	outwl.ensureSpace(graph.nnodes / 5);
	inwl.pushHost(0);	// source.
	inwlptr = &inwl;
	outwlptr = &outwl;
	if (cudaMalloc((void **)&nerr, sizeof(unsigned)) != cudaSuccess) CudaTest("allocating nerr failed");

	printf("solving.\n");
	starttime = rtclock();

	do {
		++iteration;

		/*unsigned nblocks = (inwlptr->getSize() + BLOCKSIZE - 1) / BLOCKSIZE;
		if (nblocks > MAXNBLOCKS) {
			nblocks = MAXNBLOCKS;
		}*/
		//unsigned nblocks = 1;
		unsigned nblocks = MAXNBLOCKS;
		CUDA_SAFE_CALL(cudaMemcpy(nerr, &intzero, sizeof(intzero), cudaMemcpyHostToDevice));
		//printf("invoking drelax with %d blocks, blocksize=%d, wlsize=%d, outwlsize=%d, iteration=%d.\n", nblocks, BLOCKSIZE, inwlptr->getSize(), outwlptr->getSize(), iteration);
		//inwlptr->printHost();
		drelax <<<nblocks, BLOCKSIZE>>> (dist, graph, *inwlptr, *outwlptr, nerr);
		CudaTest("solving failed");
		//outwlptr->printHost();
		CUDA_SAFE_CALL(cudaMemcpy(&hnerr, nerr, sizeof(hnerr), cudaMemcpyDeviceToHost));
		wlsz = outwlptr->getSize();
		if (hnerr == 0) {
			if (iteration % 500 == 0) printf("iteration=%d, outwl.size=%d.\n", iteration, wlsz);
			SWAP(inwlptr, outwlptr);
			outwlptr->noverflows = inwlptr->noverflows;
		} else {	// error: currently only buffer oveflow.
			//printf("compressing (iteration=%d): outwlsize=%d, inwlsize=%d.\n", iteration, wlsz, inwlptr->getSize());
			//outwlptr->compressHost(wlsz, graph.nnodes);
			if (++outwlptr->noverflows == MAXOVERFLOWS) {
				unsigned cap = inwlptr->getCapacity();
				//inwlptr->printHost();
				inwlptr->ensureSpace(2 * cap);	// double the capacity.
				//inwlptr->printHost();
				outwlptr->ensureSpace(2 * cap);
				inwlptr->appendHost(outwlptr);
				outwlptr->noverflows = 0;
			} else {
				// defer increasing worklist capacity.
				printf("\tdeferred increasing worklist capacity.\n");
			}
			//printf("\tinwlsz=%d, outwlsz=%d.\n", inwlptr->getSize(), outwlptr->getSize());
		}
		outwlptr->clearHost();	// clear it whether overflow or not.
		//printf("\tcleared: inwlsz=%d, outwlsz=%d.\n", inwlptr->getSize(), outwlptr->getSize());
		//getchar();
	} while (wlsz);
	endtime = rtclock();
	
	printf("\titerations = %d.\n", iteration);
	runtime = (1000.0f * (endtime - starttime));
	printf("\truntime [%s] = %f ms.\n", SSSP_VARIANT, runtime);
}
 
