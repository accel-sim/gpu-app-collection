#define BFS_VARIANT "worklista"

#define MAXDIST		100

#define AVGDEGREE	2.5
#define WORKPERTHREAD	1

unsigned int NVERTICES;

#include "cutil_subset.h"

//#include "blktiming.h"

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

	foru wt = 1;	//graph.getWeight(src, ii);
	if (wt >= MYINFINITY) return 0;

	//printf("%d %d %d %d\n", src, dst, dist[src], dist[dst]);


	 foru altdist = dist[src] + wt;
	 if(dist[dst] == MYINFINITY)
	   {
	     dist[dst] = altdist;
	     return MYINFINITY;
	   }

	 /* if (altdist < dist[dst]) { */
	 /* 	foru olddist = atomicMin(&dist[dst], altdist); */
	 /* 	if (altdist < olddist) { */
	 /* 		return olddist; */
	 /* 	}  */
	 /* 	// someone else updated distance to a lower value. */
	 /* } */
	  return 0;
}
__device__
unsigned processnode(foru *dist, Graph &graph, unsigned work, bool *active) {
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

	bool changed = false;

	for (unsigned ii = 0; ii < neighborsize; ++ii) {
		unsigned dst = graph.nnodes;
		foru olddist = processedge(dist, graph, nn, ii, dst);
		if (olddist) {
		  //printf("%d pushing %d\n", nn, dst);
		  active[dst] = true;
		  /* if (outwl.push(graph, dst)) {	// buffer oveflow. */
		  /* 		// critical: reset the distance, since this node couldn't be added to the worklist. */
		  /* 		dist[dst] = olddist;	// no atomicMax required. */
		  /* 		return 1; */
		  /* 	} */
		  changed = 1;
		}
	}
	//free(newwork);
	return int(changed);
}

__global__
void drelax(foru *dist, Graph graph, unsigned *gerrno, bool *active1, bool *active2) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id >= graph.nnodes)
	  return;

	if(!active1[id])
	  return;

	active1[id] = false;

	//printf("%d degree %d\n", id, graph.getOutDegree(id));
	//printf("%d work %d\n", id, end - start);

	if (processnode(dist, graph, id, active2)) {	// buffer oveflow.
	  *gerrno = 1;
	  return;
	}
}

/* __global__ */
/* void drelax_instr(foru *dist, Graph graph, unsigned *gerrno, bool *active1, bool *active2, TIMING_ARGS) { */


/*   TIME_START(); */


/* 	unsigned id = blockIdx.x * blockDim.x + threadIdx.x; */

/* 	if(id >= graph.nnodes) */
/* 	  return; */

/* 	if(!active1[id]) */
/* 	  return; */

/* 	active1[id] = false; */

/* 	//printf("%d degree %d\n", id, graph.getOutDegree(id)); */
/* 	//printf("%d work %d\n", id, end - start); */

/* 	if (processnode(dist, graph, id, active2)) {	// buffer oveflow. */
/* 	  *gerrno = 1; */
/* 	  return; */
/* 	} */
/*    TIME_END(); */
/* } */

int get_count(bool *wl, int nlength)
{
  int count = 0;
  for(int i = 0; i < nlength; i++)
    {
      if(wl[i]) 
	count++;
    }
      
  return count;
}

void display_items(bool *wl, int nlength)
{
  printf("WL: ");
  for(int i = 0, k = 0; i < nlength; i++)
    {
      if(wl[i])
	printf("%d %d, ", k++, i);
    }
      
  printf("\n");
}

void bfs(Graph &graph, foru *dist)
{
	foru foruzero = 0.0;
	unsigned intzero = 0;
	unsigned int NBLOCKS, FACTOR = 128;
	bool *changed;
	int iteration = 0;
	unsigned *nerr, hnerr;

	double starttime, endtime;
	double runtime;

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	NBLOCKS = deviceProp.multiProcessorCount;

	NVERTICES = graph.nnodes;

	FACTOR = (NVERTICES + MAXBLOCKSIZE * NBLOCKS - 1) / (MAXBLOCKSIZE * NBLOCKS);

	printf("initializing (nblocks=%d, blocksize=%d).\n", NBLOCKS*FACTOR, MAXBLOCKSIZE);
	initialize <<<NBLOCKS*FACTOR, MAXBLOCKSIZE>>> (dist, graph.nnodes);
	CudaTest("initializing failed");
	cudaMemcpy(&dist[0], &foruzero, sizeof(foruzero), cudaMemcpyHostToDevice);

	if (cudaMalloc((void **)&changed, sizeof(bool)) != cudaSuccess) CudaTest("allocating changed failed");

	if (cudaMalloc((void **)&nerr, sizeof(unsigned)) != cudaSuccess) CudaTest("allocating nerr failed");

	bool *d_active1, *d_active2, *active;

	active = (bool *) calloc(graph.nnodes, sizeof(bool));
	cudaMalloc(&d_active1, graph.nnodes * sizeof(bool));
	cudaMalloc(&d_active2, graph.nnodes * sizeof(bool));
	CudaTest("malloc failed!");
	cudaMemset(d_active1, 0, graph.nnodes * sizeof(bool));
	cudaMemset(d_active2, 0, graph.nnodes * sizeof(bool));
	CudaTest("memtest failed!");
	bool truevalue = true;
	cudaMemcpy(d_active1, &truevalue, sizeof(truevalue), cudaMemcpyHostToDevice);	
	
	printf("solving.\n");
	starttime = rtclock();

	const int BLKSIZE = 384;
	do {
		++iteration;

		/*unsigned nblocks = (inwlptr->getSize() + BLOCKSIZE - 1) / BLOCKSIZE;
		if (nblocks > MAXNBLOCKS) {
			nblocks = MAXNBLOCKS;
		}*/
		unsigned nblocks = (graph.nnodes + BLKSIZE - 1) / BLKSIZE; 
		CUDA_SAFE_CALL(cudaMemcpy(nerr, &intzero, sizeof(intzero), cudaMemcpyHostToDevice));
		//printf("invoking drelax with %d blocks, blocksize=%d, wlsize=%d, outwlsize=%d, iteration=%d.\n", nblocks, BLOCKSIZE, inwlptr->getSize(), outwlptr->getSize(), iteration);
		//inwlptr->printHost();

		CUDA_SAFE_CALL(cudaMemcpy(active, d_active1, graph.nnodes * sizeof(bool), cudaMemcpyDeviceToHost));
		//display_items(active, graph.nnodes);
		//printf("%d %d\n", iteration, get_count(active, graph.nnodes));

		drelax <<<nblocks, BLKSIZE>>> (dist, graph, nerr, d_active1, d_active2);

		bool *tmp1;
		tmp1 = d_active1;
		d_active1 = d_active2;
		d_active2 = tmp1;


		CudaTest("solving failed");
		//outwlptr->printHost();
		CUDA_SAFE_CALL(cudaMemcpy(&hnerr, nerr, sizeof(hnerr), cudaMemcpyDeviceToHost));
		//wlsz = outwlptr->getSize();
	} while (hnerr); // hnerr is actually not_done
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	endtime = rtclock();
	
	printf("\titerations = %d.\n", iteration);
	runtime = (1000.0f * (endtime - starttime));
	printf("\truntime [%s] = %f ms.\n", BFS_VARIANT, runtime);

	return;
}
