/* from optimizations/bfs/topology-atomic.cu */

#define BFS_VARIANT "topology-atomic"

#define WORKPERTHREAD	1
#define VERTICALWORKPERTHREAD	12	// max value, see relax.
#define BLKSIZE 1024
#define BANKSIZE	BLKSIZE

__global__
void initialize(foru *dist, unsigned int *nv) {
	unsigned int ii = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("ii=%d, nv=%d.\n", ii, *nv);
	if (ii < *nv) {
		dist[ii] = MYINFINITY;
	}
}

__global__ 
void dprocess(float *matrix, foru *dist, unsigned int *prev, bool *relaxed) {
	__shared__ volatile unsigned int dNVERTICES;
	unsigned int ii = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int v;
	unsigned int jj;

	if (ii < dNVERTICES) {
		unsigned int u = dNVERTICES;
		foru mm = MYINFINITY;
		//minimum <<< NVERTICES / BLKSIZE, BLKSIZE>>> (dist, relaxed, &u);
		for (jj = 0; jj < dNVERTICES; ++jj) {
			if (relaxed[jj] == false && dist[jj] < mm) {
				mm = dist[jj];
				u = jj;
			}
		}
		if (u != dNVERTICES && dist[u] != MYINFINITY) {
			relaxed[u] = true;
			for (v = 0; v < dNVERTICES; ++v) {
				if (matrix[u*dNVERTICES + v] > 0) {
					foru alt = dist[u] + matrix[u*dNVERTICES + v];
					if (alt < dist[v]) {
						dist[v] = alt;
						prev[v] = u;
					}
				}
			}
		}
	}

}
 
__global__
void drelax(foru *dist, unsigned int *edgessrcdst, foru *edgessrcwt, unsigned int *psrc, unsigned int *noutgoing, unsigned int *nedges, unsigned int *nv, bool *changed, unsigned int *srcsrc, unsigned unroll) {
	unsigned int workperthread = WORKPERTHREAD;
 	unsigned nn = workperthread * (blockIdx.x * blockDim.x + threadIdx.x);
	unsigned int ii;
	__shared__ int changedv[VERTICALWORKPERTHREAD * BLKSIZE]; 
	int iichangedv = threadIdx.x;
	int anotheriichangedv = iichangedv;
	unsigned int nprocessed = 0;

	// collect the work to be performed.
	for (unsigned node = 0; node < workperthread; ++node, ++nn) {
		changedv[iichangedv] = nn;
		iichangedv += BANKSIZE;
	}

	// go over the worklist and keep updating it in a BFS manner.
	while (anotheriichangedv < iichangedv) {
	    nn = changedv[anotheriichangedv];
		anotheriichangedv += BANKSIZE;
	    if (nn < *nv) {
		unsigned src = nn;					// source node.
		//if (src < *nv && psrc[srcsrc[src]]) {
			unsigned int start = psrc[srcsrc[src]];
			unsigned int end = start + noutgoing[src];
			// go over all the target nodes for the source node.
			for (ii = start; ii < end; ++ii) {
				unsigned int u = src;
				unsigned int v = edgessrcdst[ii];	// target node.
				float wt = 1;
				//if (wt > 0 && v < *nv) {
					foru alt = dist[u] + wt;
					if (alt < dist[v]) {
							//dist[v] = alt;
							atomicMin((unsigned *)&dist[v], (unsigned )alt);
							if (++nprocessed < unroll) {
								// add work to the worklist.
								changedv[iichangedv] = v;
								iichangedv += BANKSIZE;
							}
					}
				//}
			}
		//}
	    }
	}
	if (nprocessed) {
		*changed = true;
	}
}

void bfs(Graph &graph, foru *dist)
{
	cudaFuncSetCacheConfig(drelax, cudaFuncCachePreferShared);

	foru zero = 0;
	//unsigned int intzero = 0, intone = 1;
	unsigned int NBLOCKS, FACTOR = 128;
	unsigned int *nedges, hnedges;
	unsigned int *nv;
	bool *changed, hchanged;
	int iteration = 0;

	double starttime, endtime;
	double runtime;
	cudaEvent_t start, stop;
	float time;

	cudaDeviceProp deviceProp;
	unsigned int NVERTICES;
 
	cudaGetDeviceProperties(&deviceProp, 0);
	NBLOCKS = deviceProp.multiProcessorCount;

	NVERTICES = graph.nnodes;
	hnedges = graph.nedges;

	FACTOR = (NVERTICES + BLKSIZE * NBLOCKS - 1) / (BLKSIZE * NBLOCKS);

	if (cudaMalloc((void **)&nv, sizeof(unsigned int)) != cudaSuccess) CudaTest("allocating nv failed");
	cudaMemcpy(nv, &NVERTICES, sizeof(NVERTICES), cudaMemcpyHostToDevice);

	if (cudaMalloc((void **)&nedges, sizeof(unsigned int)) != cudaSuccess) CudaTest("allocating nedges failed");
	cudaMemcpy(nedges, &hnedges, sizeof(unsigned int), cudaMemcpyHostToDevice);

 	if (cudaMalloc((void **)&changed, sizeof(bool)) != cudaSuccess) CudaTest("allocating changed failed");

	for (unsigned unroll=VERTICALWORKPERTHREAD; unroll <= VERTICALWORKPERTHREAD; ++unroll) {
	  //printf("initializing (nblocks=%d, blocksize=%d).\n", NBLOCKS*FACTOR, BLKSIZE);
	  initialize <<<NBLOCKS*FACTOR, BLKSIZE>>> (dist, nv);
	  CudaTest("initializing failed");
	  cudaMemcpy(&dist[0], &zero, sizeof(zero), cudaMemcpyHostToDevice);

	  //printf("solving.\n");
	  starttime = rtclock();
	  cudaEventCreate(&start);
	  cudaEventCreate(&stop);

	  cudaEventRecord(start, 0);
	  do {
	    ++iteration;
	    //printf("iteration %d.\n", iteration);
	    hchanged = false;
	    cudaMemcpy(changed, &hchanged, sizeof(bool), cudaMemcpyHostToDevice);
	    unsigned nblocks = NBLOCKS*FACTOR;

	    //sree: why are nedges and nv pointers?

	    drelax <<<nblocks/WORKPERTHREAD, BLKSIZE>>> (dist, graph.edgessrcdst, graph.edgessrcwt, graph.psrc, graph.noutgoing, nedges, nv, changed, graph.srcsrc, unroll);
	    CudaTest("solving failed");
	    cudaMemcpy(&hchanged, changed, sizeof(bool), cudaMemcpyDeviceToHost);
	  } while (hchanged);
	  cudaEventRecord(stop, 0);  cudaEventSynchronize(stop);  cudaEventElapsedTime(&time, start, stop);
	  endtime = rtclock();

	  runtime = 1000*(endtime - starttime);
	  //printf("%d ms.\n", runtime);
	  printf("\n");
	  printf("\truntime [%s] = %.3lf ms.\n", 
		 BFS_VARIANT, 1000 * (endtime - starttime));
	  
	  printf("%d\t%f\n", unroll, runtime);
	}

	printf("iterations = %d.\n", iteration);
}
