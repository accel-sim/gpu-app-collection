#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>
#include <cassert>
#include <inttypes.h>


#define FORMATSTR	"%d %d %d"

unsigned allocOnHost(Graph &gg) {
	gg.destination = (unsigned int *)malloc((gg.nedges+1) * sizeof(unsigned int));	// first entry acts as null.
	gg.weight = (foru *)malloc((gg.nedges+1) * sizeof(foru));	// first entry acts as null.
	gg.psrc = (unsigned int *)calloc(gg.nnodes+1, sizeof(unsigned int));	// init to null.
	gg.psrc[gg.nnodes] = gg.nedges;	// last entry points to end of edges, to avoid thread divergence in drelax.
	gg.noutgoing = (unsigned int *)calloc(gg.nnodes, sizeof(unsigned int));	// init to 0.
	gg.srcsrc = (unsigned int *)malloc(gg.nnodes * sizeof(unsigned int));

	return 0;
}
void progressPrint(unsigned maxii, unsigned ii) {
	const unsigned nsteps = 10;
	unsigned ineachstep = (maxii / nsteps);
	if (ii % ineachstep == 0) {
		printf("\t%3d%%\r", ii*100/maxii + 1);
		fflush(stdout);
	}
}
unsigned readFromEdges(char file[], Graph &gg) {
	std::ifstream cfile;
	cfile.open(file);

	std::string str;
	getline(cfile, str);
	sscanf(str.c_str(), "%d %d", &gg.nnodes, &gg.nedges);

	printf("file %s: nnodes=%d, nedges=%d.\n", file, gg.nnodes, gg.nedges);
	allocOnHost(gg);
	for (unsigned ii = 0; ii < gg.nnodes; ++ii) {
		gg.srcsrc[ii] = ii;
	}


	unsigned int prevnode = 0;
	unsigned int tempsrcnode;
	unsigned int ncurroutgoing = 0;
	unsigned unweightedgraph = 0;
	for (unsigned ii = 0; ii < gg.nedges; ++ii) {
		getline(cfile, str);
		if (unweightedgraph) {
			sscanf(str.c_str(), "%d %d", &tempsrcnode, &gg.destination[ii+1]);
			gg.weight[ii+1] = 0;
		} else {
			sscanf(str.c_str(), FORMATSTR, &tempsrcnode, &gg.destination[ii+1], &gg.weight[ii+1]);
		}
		if (prevnode == tempsrcnode) {
			if (ii == 0) {
				gg.psrc[tempsrcnode] = ii + 1;
			}
			++ncurroutgoing;
		} else {
			gg.psrc[tempsrcnode] = ii + 1;
			if (ncurroutgoing) {
				gg.noutgoing[prevnode] = ncurroutgoing;
			}
			prevnode = tempsrcnode;
			ncurroutgoing = 1;	// not 0.
		}

		progressPrint(gg.nedges, ii);
	}
	gg.noutgoing[prevnode] = ncurroutgoing;	// last entries.

	printf("\n");
	cfile.close();
	return 0;
}

unsigned readFromGR(char file[], Graph &gg) {
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
  	foru  *edgeData = (foru *)fptr32;
	
	// cuda.
	gg.nnodes = numNodes;
	gg.nedges = numEdges;

	printf("file %s: nnodes=%d, nedges=%d.\n", file, gg.nnodes, gg.nedges);
	allocOnHost(gg);

	for (unsigned ii = 0; ii < gg.nnodes; ++ii) {
		// fill unsigned *noutgoing, *nincoming, *srcsrc, *psrc, *destination; unsigned *weight;
		gg.srcsrc[ii] = ii;
		if (ii > 0) {
			gg.psrc[ii] = le64toh(outIdx[ii - 1]) + 1;
			gg.noutgoing[ii] = le64toh(outIdx[ii]) - le64toh(outIdx[ii - 1]);
		} else {
			gg.psrc[0] = 1;
			gg.noutgoing[0] = le64toh(outIdx[0]);
		}
		for (unsigned jj = 0; jj < gg.noutgoing[ii]; ++jj) {
			unsigned edgeindex = gg.psrc[ii] + jj;
			unsigned dst = le32toh(outs[edgeindex - 1]);
			if (dst >= gg.nnodes) printf("\tinvalid edge from %d to %d at index %d(%d).\n", ii, dst, jj, edgeindex);
			gg.destination[edgeindex] = dst;
			gg.weight[edgeindex] = edgeData[edgeindex - 1];	// Weighted.
			//gg.weight[edgeindex] = 1;			// Unweighted like wikipedia.

		}
		progressPrint(gg.nnodes, ii);
	}
	printf("\n");

	cfile.close();	// probably galois doesn't close its file due to mmap.
	return 0;
}
unsigned readInput(char file[], Graph &gg) {
	if (strstr(file, ".edges") || strstr(file, ".undirected")) {
		return readFromEdges(file, gg);
	} else if (strstr(file, ".gr")) {
		return readFromGR(file, gg);
	}
	return 0;
}

