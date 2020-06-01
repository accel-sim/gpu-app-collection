#ifndef LSG_GRAPH
#define LSG_GRAPH

#define MYINFINITY	1000000000
#define DISTANCETHRESHOLD	150
#define THRESHOLDDEGREE		10

typedef struct Graph {
	enum {NotAllocated, AllocatedOnHost, AllocatedOnDevice} memory;

	unsigned read(char file[]);
	long unsigned cudaCopy(struct Graph &copygraph);
	unsigned optimize();
	unsigned printStats();
	void     print();

	Graph();
	~Graph();
	unsigned init();
	unsigned allocOnHost();
	unsigned allocOnDevice();
	unsigned dealloc();
	unsigned deallocOnHost();
	unsigned deallocOnDevice();
	unsigned optimizeone();
	unsigned optimizetwo();
	void allocLevels();
	void freeLevels();
	void progressPrint(unsigned maxii, unsigned ii);
	unsigned readFromEdges(char file[]);
	unsigned readFromGR(char file[]);

	__device__ void printStats1x1();
	__device__ void print1x1();
	__device__ unsigned getOutDegree(unsigned src);
	__device__ unsigned getInDegree(unsigned src);
	__device__ unsigned getDestination(unsigned src, unsigned nthedge);
	__device__ foru    getWeight(unsigned src, unsigned nthedge);
	__device__ unsigned getMinEdge(unsigned src);

	__device__ unsigned getFirstEdge(unsigned src);
	__device__ unsigned findStats();
	__device__ void computeStats();
	__device__ bool computeLevels();
	__device__ unsigned findMaxLevel();
	__device__ void computeDiameter();
	__device__ void computeInOut();
	__device__ void initLevels();


	unsigned nnodes, nedges;
	unsigned *noutgoing, *nincoming, *srcsrc, *psrc, *edgessrcdst;
	foru *edgessrcwt;
	unsigned *levels;
	unsigned source;

	unsigned *maxOutDegree, *maxInDegree;
	unsigned diameter;
	bool foundStats;

} Graph;

static unsigned CudaTest(char *msg);

__device__ unsigned Graph::getOutDegree(unsigned src) {
	if (src < nnodes) {
		return noutgoing[src];
	}
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	printf("Error: %s(%d): thread %d, node %d out of bounds %d.\n", __FILE__, __LINE__, id, src, nnodes); 
	return 0;
}
__device__ unsigned Graph::getInDegree(unsigned dst) {
	if (dst < nnodes) {
		return nincoming[dst];
	}
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	printf("Error: %s(%d): thread %d, node %d out of bounds %d.\n", __FILE__, __LINE__, id, dst, nnodes); 
	return 0;
}
__device__ unsigned Graph::getDestination(unsigned src, unsigned nthedge) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if (src < nnodes && nthedge < getOutDegree(src)) {
		unsigned edge = getFirstEdge(src) + nthedge;
		if (edge && edge < nedges + 1) {
			return edgessrcdst[edge];
		}
		////printf("Error: %s(%d): thread %d, node %d: edge %d out of bounds %d.\n", __FILE__, __LINE__, id, src, edge, nedges + 1);
		return nnodes;
	}
	if (src < nnodes) {
		printf("Error: %s(%d): thread %d, node %d: edge %d out of bounds %d.\n", __FILE__, __LINE__, id, src, nthedge, getOutDegree(src));
	} else {
		printf("Error: %s(%d): thread %d, node %d out of bounds %d.\n", __FILE__, __LINE__, id, src, nnodes);
	}
	return nnodes;
}
__device__ foru Graph::getWeight(unsigned src, unsigned nthedge) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if (src < nnodes && nthedge < getOutDegree(src)) {
		unsigned edge = getFirstEdge(src) + nthedge;
		if (edge && edge < nedges + 1) {
			return edgessrcwt[edge];
		}
		////printf("Error: %s(%d): thread %d, node %d: edge %d out of bounds %d.\n", __FILE__, __LINE__, id, src, edge, nedges + 1);
		return MYINFINITY;
	}
	if (src < nnodes) {
		printf("Error: %s(%d): thread %d, node %d: edge %d out of bounds %d.\n", __FILE__, __LINE__, id, src, nthedge, getOutDegree(src));
	} else {
		printf("Error: %s(%d): thread %d, node %d out of bounds %d.\n", __FILE__, __LINE__, id, src, nnodes);
	}
	return MYINFINITY;
}

__device__ unsigned Graph::getFirstEdge(unsigned src) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if (src < nnodes) {
		unsigned srcnout = getOutDegree(src);
		//if (src == 368) printf("nout[368]=%d, psrc[srcsrc[368]]=%d, psrc[368]=%d, srcsrc[368]=%d.\n", srcnout, psrc[srcsrc[src]], psrc[src], srcsrc[src]);
		if (srcnout > 0 && srcsrc[src] < nnodes) {
			return psrc[srcsrc[src]];
		}
		printf("Error: %s(%d): thread %d, edge %d out of bounds %d.\n", __FILE__, __LINE__, id, 0, srcnout);
		return 0;
	}
	printf("Error: %s(%d): thread %d, node %d out of bounds %d.\n", __FILE__, __LINE__, id, src, nnodes);
	return 0;
}
__device__ unsigned Graph::getMinEdge(unsigned src) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if (src < nnodes) {
		unsigned srcnout = getOutDegree(src);
		if (srcnout > 0) {
			unsigned minedge = 0;
			foru    minwt   = getWeight(src, 0);
			for (unsigned ii = 1; ii < srcnout; ++ii) {
				foru wt = getWeight(src, ii);
				if (wt < minwt) {
					minedge = ii;
					minwt = wt;
				}
			}
			return minedge;
		}
		printf("Error: %s(%d): thread %d, edge %d out of bounds %d.\n", __FILE__, __LINE__, id, 0, srcnout);
		return 0;
	}
	printf("Error: %s(%d): thread %d, node %d out of bounds %d.\n", __FILE__, __LINE__, id, src, nnodes);
	return 0;
}
__device__ void Graph::print1x1() {
	unsigned edgescounted = 0;
	printf("%d %d\n", nnodes, nedges);
	for (unsigned ii = 0; ii < nnodes; ++ii) {
		unsigned nout = getOutDegree(ii);
		for (unsigned ee = 0; ee < nout; ++ee) {
			unsigned dst = getDestination(ii, ee);
			foru wt = getWeight(ii, ee);
			printf("%d %d %d\n", ii, dst, wt);
			++edgescounted;
		}
	}
	if (nedges != edgescounted) {
		printf("Error: nedges=%d, edgescounted=%d.\n", nedges, edgescounted);
	}
}
unsigned Graph::init() {
	noutgoing = nincoming = srcsrc = psrc = edgessrcdst = NULL;
	edgessrcwt = NULL;
	source = 0;
	nnodes = nedges = 0;
	memory = NotAllocated;

	maxOutDegree = maxInDegree = NULL;
	diameter = 0;
	foundStats = false;

	return 0;
}
unsigned Graph::allocOnHost() {
	edgessrcdst = (unsigned int *)malloc((nedges+1) * sizeof(unsigned int));	// first entry acts as null.
	edgessrcwt = (foru *)malloc((nedges+1) * sizeof(foru));	// first entry acts as null.
	psrc = (unsigned int *)calloc(nnodes+1, sizeof(unsigned int));	// init to null.
	psrc[nnodes] = nedges;	// last entry points to end of edges, to avoid thread divergence in drelax.
	noutgoing = (unsigned int *)calloc(nnodes, sizeof(unsigned int));	// init to 0.
	nincoming = (unsigned int *)calloc(nnodes, sizeof(unsigned int));	// init to 0.
	srcsrc = (unsigned int *)malloc(nnodes * sizeof(unsigned int));

	maxOutDegree = (unsigned *)malloc(sizeof(unsigned));
	maxInDegree = (unsigned *)malloc(sizeof(unsigned));
	*maxOutDegree = 0;
	*maxInDegree = 0;

	memory = AllocatedOnHost;
	return 0;
}
unsigned Graph::allocOnDevice() {
	if (cudaMalloc((void **)&edgessrcdst, (nedges+1) * sizeof(unsigned int)) != cudaSuccess) 
		CudaTest("allocating edgessrcdst failed");
	if (cudaMalloc((void **)&edgessrcwt, (nedges+1) * sizeof(foru)) != cudaSuccess) 
		CudaTest("allocating edgessrcwt failed");
	if (cudaMalloc((void **)&psrc, (nnodes+1) * sizeof(unsigned int)) != cudaSuccess) 
		CudaTest("allocating psrc failed");
	if (cudaMalloc((void **)&noutgoing, nnodes * sizeof(unsigned int)) != cudaSuccess) 
		CudaTest("allocating noutgoing failed");
	if (cudaMalloc((void **)&nincoming, nnodes * sizeof(unsigned int)) != cudaSuccess) 
		CudaTest("allocating nincoming failed");
	if (cudaMalloc((void **)&srcsrc, nnodes * sizeof(unsigned int)) != cudaSuccess) 
		CudaTest("allocating srcsrc failed");

	if (cudaMalloc((void **)&maxOutDegree, 1 * sizeof(unsigned int)) != cudaSuccess) 
		CudaTest("allocating maxOutDegree failed");
	if (cudaMalloc((void **)&maxInDegree, 1 * sizeof(unsigned int)) != cudaSuccess) 
		CudaTest("allocating maxInDegree failed");

	memory = AllocatedOnDevice;
	return 0;
}
unsigned Graph::deallocOnHost() {
	free(noutgoing);
	free(nincoming);
	free(srcsrc);
	free(psrc);
	free(edgessrcdst);
	free(edgessrcwt);

	free(maxOutDegree);
	free(maxInDegree);
	return 0;
}
unsigned Graph::deallocOnDevice() {
	cudaFree(noutgoing);
	cudaFree(nincoming);
	cudaFree(srcsrc);
	cudaFree(psrc);
	cudaFree(edgessrcdst);
	cudaFree(edgessrcwt);

	cudaFree(maxOutDegree);
	cudaFree(maxInDegree);
	return 0;
}
unsigned Graph::dealloc() {
	switch (memory) {
		case AllocatedOnHost:
			printf("dealloc on host.\n");
			deallocOnHost();
			break;
		case AllocatedOnDevice:
			printf("dealloc on device.\n");
			deallocOnDevice();
			break;
	}
	return 0;
}
Graph::Graph() {
	init();
}
Graph::~Graph() {
	//// The destructor seems to be getting called at unexpected times.
	//dealloc();
	//init();
}
//TODO: make optimizations use the graph api.
unsigned Graph::optimizeone() {
	unsigned int nvv = nnodes;	// no of vertices to be optimized.
	unsigned int insertindex = 1;	// because ii starts with 0.

	for (unsigned ii = 0; ii < nvv; ++ii) {
		unsigned src = srcsrc[ii];
		unsigned dstindex = psrc[src];
		unsigned degree = noutgoing[src];
		if (degree && srcsrc[edgessrcdst[dstindex]] > src + DISTANCETHRESHOLD) {
			unsigned int nee = degree;
			for (unsigned ee = 0; ee < nee; ++ee) {
				unsigned dst = edgessrcdst[dstindex + ee];
				unsigned dstentry = srcsrc[dst];
				// swap insertindex and dst.
				unsigned temp = psrc[insertindex];
				psrc[insertindex] = psrc[dstentry];
				psrc[dstentry] = temp;

				temp = srcsrc[ii];
				srcsrc[ii] = srcsrc[dst];
				srcsrc[dst] = temp;

				if (++insertindex >= nnodes) {
					break;
				}
			}
			if (insertindex >= nnodes) {
				break;
			}
		}
	}
	return 0;
}
unsigned Graph::optimizetwo() {
	// load balance.
	unsigned int nvv = nnodes / 2;
	bool firsthalfsmaller = true;
	unsigned int temp;

	for (unsigned ii = 0; ii < nvv; ++ii) {
		unsigned one = ii;
		unsigned two = nvv + ii;
		unsigned degreeone = noutgoing[one];
		unsigned degreetwo = noutgoing[two];

		if (degreeone > degreetwo && degreeone - degreetwo > THRESHOLDDEGREE && !firsthalfsmaller || degreetwo > degreeone && degreetwo - degreeone > THRESHOLDDEGREE && firsthalfsmaller) {
			temp = srcsrc[one];
			srcsrc[one] = srcsrc[two];
			srcsrc[two] = temp;

			temp = psrc[one];
			psrc[one] = psrc[two];
			psrc[two] = temp;
			firsthalfsmaller = !firsthalfsmaller;
		}
	}
	return 0;
}
unsigned Graph::optimize() {
	optimizeone();
	optimizetwo();
	return 0;
}
void Graph::progressPrint(unsigned maxii, unsigned ii) {
	const unsigned nsteps = 10;
	unsigned ineachstep = (maxii / nsteps);
	if(ineachstep == 0) ineachstep = 1;
	/*if (ii == maxii) {
		printf("\t100%%\n");
	} else*/ if (ii % ineachstep == 0) {
		printf("\t%3d%%\r", ii*100/maxii + 1);
		fflush(stdout);
	}
}
unsigned Graph::readFromEdges(char file[]) {
	std::ifstream cfile;
	cfile.open(file);

	std::string str;
	getline(cfile, str);
	sscanf(str.c_str(), "%d %d", &nnodes, &nedges);

	allocOnHost();
	for (unsigned ii = 0; ii < nnodes; ++ii) {
		srcsrc[ii] = ii;
	}


	unsigned int prevnode = 0;
	unsigned int tempsrcnode;
	unsigned int ncurroutgoing = 0;
	for (unsigned ii = 0; ii < nedges; ++ii) {
		getline(cfile, str);
		sscanf(str.c_str(), "%d %d %d", &tempsrcnode, &edgessrcdst[ii+1], &edgessrcwt[ii+1]);
		if (prevnode == tempsrcnode) {
			if (ii == 0) {
				psrc[tempsrcnode] = ii + 1;
			}
			++ncurroutgoing;
		} else {
			psrc[tempsrcnode] = ii + 1;
			if (ncurroutgoing) {
				noutgoing[prevnode] = ncurroutgoing;
			}
			prevnode = tempsrcnode;
			ncurroutgoing = 1;	// not 0.
		}
		++nincoming[edgessrcdst[ii+1]];

		progressPrint(nedges, ii);
	}
	noutgoing[prevnode] = ncurroutgoing;	// last entries.

	cfile.close();
	return 0;
}

unsigned Graph::readFromGR(char file[]) {
	std::ifstream cfile;
	cfile.open(file);

	// copied from GaloisCpp/trunk/src/FileGraph.h
	int masterFD = open(file, O_RDONLY);
  	if (masterFD == -1) {
	printf("FileGraph::structureFromFile: unable to open %s.\n", file);
	return 1;
  	}

  	struct stat buf;
	int f = fstat(masterFD, &buf);
  	if (f == -1) {
    		printf("FileGraph::structureFromFile: unable to stat %s.\n", file);
    		abort();
  	}
  	size_t masterLength = buf.st_size;

  	int _MAP_BASE = MAP_PRIVATE;
//#ifdef MAP_POPULATE
//  _MAP_BASE  |= MAP_POPULATE;
//#endif

  	void* m = mmap(0, masterLength, PROT_READ, _MAP_BASE, masterFD, 0);
  	if (m == MAP_FAILED) {
    		m = 0;
    		printf("FileGraph::structureFromFile: mmap failed.\n");
    		abort();
  	}

	double starttime, endtime;
	starttime = rtclock();

  	//parse file
  	uint64_t* fptr = (uint64_t*)m;
  	__attribute__((unused)) uint64_t version = le64toh(*fptr++);
  	assert(version == 1);
  	uint64_t sizeEdgeTy = le64toh(*fptr++);
  	uint64_t numNodes = le64toh(*fptr++);
  	uint64_t numEdges = le64toh(*fptr++);
  	uint64_t *outIdx = fptr;
  	fptr += numNodes;
  	uint32_t *fptr32 = (uint32_t*)fptr;
  	uint32_t *outs = fptr32; 
  	fptr32 += numEdges;
  	if (numEdges % 2) fptr32 += 1;
  	unsigned  *edgeData = (unsigned *)fptr32;

	
	// cuda.
	nnodes = numNodes;
	nedges = numEdges;

	printf("nnodes=%d, nedges=%d.\n", nnodes, nedges);
	allocOnHost();

	for (unsigned ii = 0; ii < nnodes; ++ii) {
		// fill unsigned *noutgoing, *nincoming, *srcsrc, *psrc, *edgessrcdst; foru *edgessrcwt;
		srcsrc[ii] = ii;
		if (ii > 0) {
			psrc[ii] = le64toh(outIdx[ii - 1]) + 1;
			noutgoing[ii] = le64toh(outIdx[ii]) - le64toh(outIdx[ii - 1]);
		} else {
			psrc[0] = 1;
			noutgoing[0] = le64toh(outIdx[0]);
		}
		for (unsigned jj = 0; jj < noutgoing[ii]; ++jj) {
			unsigned edgeindex = psrc[ii] + jj;
			unsigned dst = le32toh(outs[edgeindex - 1]);
			if (dst >= nnodes) printf("\tinvalid edge from %d to %d at index %d(%d).\n", ii, dst, jj, edgeindex);
			edgessrcdst[edgeindex] = dst;
			edgessrcwt[edgeindex] = edgeData[edgeindex - 1];

			++nincoming[dst];
			//if (ii == 194 || ii == 352) {
			//	printf("edge %d: %d->%d, wt=%d.\n", edgeindex, ii, dst, edgessrcwt[edgeindex]);
			//}
		}
		progressPrint(nnodes, ii);
	}

	cfile.close();	// probably galois doesn't close its file due to mmap.

	endtime = rtclock();

	printf("read %lld bytes in %0.2f ms (%0.2f MB/s)\n", masterLength, 1000 * (endtime - starttime), (masterLength / 1048576) / (endtime - starttime));

	return 0;
}
unsigned Graph::read(char file[]) {
	if (strstr(file, ".edges")) {
		return readFromEdges(file);
	} else if (strstr(file, ".gr")) {
		return readFromGR(file);
	}
	return 0;
}
long unsigned Graph::cudaCopy(struct Graph &copygraph) {
	long unsigned totalcommu = 0;
	copygraph.nnodes = nnodes;
	copygraph.nedges = nedges;

	copygraph.allocOnDevice();

	cudaMemcpy(copygraph.edgessrcdst, edgessrcdst, (nedges+1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(copygraph.edgessrcwt, edgessrcwt, (nedges+1) * sizeof(foru), cudaMemcpyHostToDevice);
	cudaMemcpy(copygraph.psrc, psrc, (nnodes+1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(copygraph.noutgoing, noutgoing, nnodes * sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(copygraph.nincoming, nincoming, nnodes * sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(copygraph.srcsrc, srcsrc, nnodes * sizeof(unsigned int), cudaMemcpyHostToDevice);
	totalcommu += nedges * sizeof(unsigned)
		    + nedges * sizeof(foru)
		    + nnodes * sizeof(unsigned)
		    + nnodes * sizeof(unsigned)
		    + nnodes * sizeof(unsigned);

	cudaMemcpy(copygraph.maxOutDegree, maxOutDegree, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(copygraph.maxInDegree, maxInDegree, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice);
	copygraph.diameter = diameter;
	copygraph.foundStats = foundStats;

	return totalcommu;
}
__device__ void Graph::computeStats() {
	computeInOut();
	computeDiameter();
}
__device__ bool Graph::computeLevels() {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	bool changed = false;

	if (id < nnodes) {
		unsigned iilevel = levels[id];
		unsigned noutii = getOutDegree(id);

		//printf("level[%d] = %d.\n", id, iilevel);
		for (unsigned jj = 0; jj < noutii; ++jj) {
			unsigned dst = getDestination(id, jj);

			if (dst < nnodes && levels[dst] > iilevel + 1) {
				levels[dst] = iilevel + 1;
				changed = true;
			} else if (dst >= nnodes) {
				printf("\t%s(%d): dst %d >= nnodes %d.\n", __FILE__, __LINE__, dst, nnodes);
			}
		}
	}
	return changed;
}

#define MAX(a, b)	(a < nnodes && a > b ? a : b)

__device__ unsigned Graph::findMaxLevel() {
	unsigned maxlevel = 0;
	for (unsigned ii = 0; ii < nnodes; ++ii) {
		maxlevel = MAX(levels[ii], maxlevel);
	}
	return maxlevel;
}
__device__  void Graph::computeDiameter() {
	diameter = findMaxLevel();
}
__device__ void Graph::computeInOut() {
	for (unsigned ii = 0; ii < nnodes; ++ii) {
		// process outdegree.
		unsigned noutii = getOutDegree(ii);
		if (noutii > *maxOutDegree) {
			*maxOutDegree = noutii;
		}
		// process indegree.
		unsigned ninii = getInDegree(ii);
		if (ninii > *maxInDegree) {
			*maxInDegree = ninii;
		}
	}
}

__device__ void Graph::printStats1x1() {	// 1x1.
	char prefix[] = "\t";

	computeStats();

	printf("%snnodes             = %d.\n",   prefix, nnodes);
	printf("%snedges             = %d.\n",   prefix, nedges);
	printf("%savg, max outdegree = %.2f, %d.\n", prefix, nedges*1.0 / nnodes, *maxOutDegree);
	printf("%savg, max indegree  = %.2f, %d.\n", prefix, nedges*1.0 / nnodes, *maxInDegree);
	printf("%sdiameter           = %d.\n",   prefix, diameter);
	return;
}
void Graph::allocLevels() {
	if (cudaMalloc((void **)&levels, nnodes * sizeof(unsigned)) != cudaSuccess) CudaTest("allocating levels failed");
}
void Graph::freeLevels() {
	//printf("freeing levels.\n");
	cudaFree(levels);
}
__device__ void Graph::initLevels() {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < nnodes) levels[id] = nnodes;
}
/*
 * Global functions.
 */
__global__ void dprintstats(Graph graph) {
	graph.printStats1x1();
}
__global__ void dcomputelevels(Graph graph, bool *changed) {
	if (graph.computeLevels()) {
		*changed = true;
	}
}
__global__ void dinitlevels(Graph graph) {
	graph.initLevels();
}
__global__ void dprint1x1(Graph graph) {
	graph.print1x1();
}
void Graph::print() {
	dprint1x1<<<1,1>>>(*this);
	CudaTest("print1x1 failed");
}
unsigned Graph::printStats() {
	allocLevels();
	dinitlevels<<<(nnodes+MAXBLOCKSIZE-1)/MAXBLOCKSIZE, MAXBLOCKSIZE>>>(*this);
	CudaTest("dinitlevels failed");

	unsigned intzero = 0;
	cudaMemcpy(&levels[source], &intzero, sizeof(intzero), cudaMemcpyHostToDevice);
	bool *changed;
	if (cudaMalloc((void **)&changed, sizeof(bool)) != cudaSuccess) CudaTest("allocating changed failed");

	printf("\tnot computing levels, diameter will be zero.\n");
	/*unsigned iteration = 0;
	bool hchanged;
	do {
		++iteration;
		hchanged = false;
		cudaMemcpy(changed, &hchanged, sizeof(bool), cudaMemcpyHostToDevice);
		printf("computelevels: iteration %d.\n", iteration);
		dcomputelevels<<<(nnodes+BLOCKSIZE-1)/BLOCKSIZE, BLOCKSIZE>>>(*this, changed);
		CudaTest("dcomputelevels failed");
		printf("computelevels: iteration %d over.\n", iteration);
		cudaMemcpy(&hchanged, changed, sizeof(bool), cudaMemcpyDeviceToHost);
	} while (hchanged);
	*/
	cudaFree(changed);

	dprintstats<<<1, 1>>>(*this);
	CudaTest("dprintstats failed");
	
	freeLevels();
	return 0;
}
#endif
