/* 
 * use atomicInc to automatically wrap around.
 */

#include <stdio.h>
#include <cuda.h>
#include <time.h>
#include <fstream>
#include <string>
#include <vector>
#include <iostream>
#include <string.h>

#define MINCAPACITY	65535
#define MAXOVERFLOWS	1

typedef struct Worklist {
	enum {NotAllocated, AllocatedOnHost, AllocatedOnDevice} memory;

	__device__ unsigned pushRange(unsigned *start, unsigned nitems);
	__device__ unsigned push(unsigned work);
	__device__ unsigned popRange(unsigned *start, unsigned nitems);
	__device__ unsigned pop(unsigned &work);
	__device__ void clear();
	__device__ void myItems(unsigned &start, unsigned &end);
	__device__ unsigned getItem(unsigned at);
	__device__ unsigned getItemWithin(unsigned at, unsigned hsize);
	__device__ unsigned count();

	void init();
	void init(unsigned initialcapacity);
	void setSize(unsigned hsize);
	unsigned getSize();
	void setCapacity(unsigned hcapacity);
	unsigned getCapacity();
	void pushRangeHost(unsigned *start, unsigned nitems);
	void pushHost(unsigned work);
	void clearHost();
	void setInitialSize(unsigned hsize);
	unsigned calculateSize(unsigned hstart, unsigned hend);
	void setStartEnd(unsigned hstart, unsigned hend);
	void getStartEnd(unsigned &hstart, unsigned &hend);
	void copyOldToNew(unsigned *olditems, unsigned *newitems, unsigned oldsize, unsigned oldcapacity);
	void compressHost(unsigned wlsize, unsigned sentinel);
	void printHost();
	unsigned appendHost(Worklist *otherwl);

	Worklist();
	~Worklist();
	unsigned ensureSpace(unsigned space);
	unsigned *alloc(unsigned allocsize);
	unsigned realloc(unsigned space);
	unsigned dealloc();
	unsigned freeSize();

	unsigned *items;
	unsigned *start, *end, *capacity;	// since these change, we don't want their copies with threads, hence pointers.

	unsigned noverflows;


} Worklist;

static unsigned CudaTest(char *msg);

Worklist::Worklist() {
	init();
}
void Worklist::init() {
	init(0);
}
void Worklist::init(unsigned initialcapacity) {
	start = alloc(1);
	end = alloc(1);
	capacity = alloc(1);
	setCapacity(initialcapacity);
	setInitialSize(0);

	items = NULL;
	if (initialcapacity) items = alloc(initialcapacity);

	noverflows = 0;
}
unsigned *Worklist::alloc(unsigned allocsize) {
	unsigned *ptr = NULL;
	if (cudaMalloc((void **)&ptr, allocsize * sizeof(unsigned)) != cudaSuccess) {
		//CudaTest("allocating ptr failed");
		printf("%s(%d): Allocating %d failed.\n", __FILE__, __LINE__, allocsize);
		return NULL;
	}
	return ptr;
}
unsigned Worklist::getCapacity() {
	unsigned hcapacity;
	cudaMemcpy(&hcapacity, capacity, sizeof(hcapacity), cudaMemcpyDeviceToHost);
	return hcapacity;
}
unsigned Worklist::calculateSize(unsigned hstart, unsigned hend) {
	if (hend >= hstart) {
		return hend - hstart;
	}
	// circular queue.
	unsigned hcapacity = getCapacity();
	return hend + (hcapacity - hstart + 1);
}
void Worklist::getStartEnd(unsigned &hstart, unsigned &hend) {
	cudaMemcpy(&hstart, start, sizeof(hstart), cudaMemcpyDeviceToHost);
	cudaMemcpy(&hend, end, sizeof(hend), cudaMemcpyDeviceToHost);
}
unsigned Worklist::getSize() {
	unsigned hstart, hend;
	getStartEnd(hstart, hend);
	if (hstart != 0) { printf("\tNOTICE: hstart = %d.\n", hstart); }
	return calculateSize(hstart, hend);
}
void Worklist::setStartEnd(unsigned hstart, unsigned hend) {
	cudaMemcpy(start, &hstart, sizeof(hstart), cudaMemcpyHostToDevice);
	cudaMemcpy(end, &hend, sizeof(hend), cudaMemcpyHostToDevice);
}
void Worklist::setCapacity(unsigned hcapacity) {
	cudaMemcpy(capacity, &hcapacity, sizeof(hcapacity), cudaMemcpyHostToDevice);
}
void Worklist::setInitialSize(unsigned hsize) {
	setStartEnd(0, 0);
}
void Worklist::setSize(unsigned hsize) {
	unsigned hcapacity = getCapacity();
	if (hsize > hcapacity) {
		printf("%s(%d): buffer overflow, setting size=%d, when capacity=%d.\n", __FILE__, __LINE__, hsize, hcapacity);
		return;
	}
	unsigned hstart, hend;
	getStartEnd(hstart, hend);
	if (hstart + hsize < hcapacity) {
		hend   = hstart + hsize;
	} else {
		hsize -= hcapacity - hstart;
		hend   = hsize;
	}
	setStartEnd(hstart, hend);
}
void Worklist::copyOldToNew(unsigned *olditems, unsigned *newitems, unsigned oldsize, unsigned oldcapacity) {
	unsigned mystart, myend;
	getStartEnd(mystart, myend);

	if (mystart < myend) {	// no wrap-around.
		cudaMemcpy(newitems, olditems + mystart, oldsize * sizeof(unsigned), cudaMemcpyDeviceToDevice);
	} else {
		cudaMemcpy(newitems, olditems + mystart, (oldcapacity - mystart) * sizeof(unsigned), cudaMemcpyDeviceToDevice);
		cudaMemcpy(newitems + (oldcapacity - mystart), olditems, myend * sizeof(unsigned), cudaMemcpyDeviceToDevice);
	}
}
unsigned Worklist::realloc(unsigned space) {
	unsigned hcapacity = getCapacity();
	unsigned newcapacity = (space > MINCAPACITY ? space : MINCAPACITY);
	if (hcapacity == 0) {
		setCapacity(newcapacity);
		items = alloc(newcapacity);
		if (items == NULL) {
			return 1;
		}
	} else {
		unsigned *itemsrealloc = alloc(newcapacity);
		if (itemsrealloc == NULL) {
			return 1;
		}
		unsigned oldsize = getSize();
		//cudaMemcpy(itemsrealloc, items, getSize() * sizeof(unsigned), cudaMemcpyDeviceToDevice);
		copyOldToNew(items, itemsrealloc, oldsize, hcapacity);
		dealloc();
		items = itemsrealloc;
		setCapacity(newcapacity);
		setStartEnd(0, oldsize);
	}
	printf("\tworklist capacity set to %d.\n", getCapacity());
	return 0;
}
unsigned Worklist::freeSize() {
	return getCapacity() - getSize();
}
unsigned Worklist::ensureSpace(unsigned space) {
	if (freeSize() >= space) {
		return 0;
	}
	realloc(space);
	// assert freeSize() >= space.
	return 1;
}
unsigned Worklist::dealloc() {
	cudaFree(items);
	setInitialSize(0);
	return 0;
}
Worklist::~Worklist() {
	// dealloc();
	// init();
}
__device__ unsigned Worklist::pushRange(unsigned *copyfrom, unsigned nitems) {
	if (copyfrom == NULL || nitems == 0) return 0;

	unsigned lcap = *capacity;
	unsigned offset = atomicAdd(end, nitems);
	if (offset >= lcap) {	// overflow.
		atomicSub(end, nitems);
		//unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
		//printf("%s(%d): thread %d: buffer overflow, increase capacity beyond %d.\n", __FILE__, __LINE__, id, *capacity);
		return 1;
	}
	for (unsigned ii = 0; ii < nitems; ++ii) {
		items[(offset + ii) % lcap] = copyfrom[ii];
	}
	return 0;
}
__device__ unsigned Worklist::push(unsigned work) {
	return pushRange(&work, 1);
}
__device__ unsigned Worklist::popRange(unsigned *copyto, unsigned nitems) {
	unsigned currsize = count();
	if (currsize < nitems) {
		// popping fewer than requested number of items.
		nitems = currsize;
	}
	//unsigned offset = atomicSub(size, nitems);
	//unsigned offset = atomicCAS(size, currsize, currsize - nitems);
	//unsigned offset = atomicExch(size, currsize - nitems);
	unsigned offset = 0;
	unsigned lcap = *capacity;
	if (nitems) {
		if (*start + nitems < lcap) {
			offset = atomicAdd(start, nitems);
		} else {
			offset = atomicExch(start, *start + nitems - lcap);
		}
	}
	// copy nitems starting from offset.
	for (unsigned ii = 0; ii < nitems; ++ii) {
		copyto[ii] = items[(offset + ii) % lcap];
	}
	return nitems;
}
__device__ unsigned Worklist::pop(unsigned &work) {
	return popRange(&work, 1);
}
void Worklist::pushRangeHost(unsigned *copyfrom, unsigned nitems) {
	ensureSpace(nitems);

	unsigned hsize = getSize();
	cudaMemcpy(items + hsize * sizeof(unsigned), copyfrom, nitems * sizeof(unsigned), cudaMemcpyHostToDevice);
	hsize += nitems;
	setSize(hsize);
}
void Worklist::pushHost(unsigned work) {
	pushRangeHost(&work, 1);

}
__device__ void Worklist::clear() {	// should be invoked by a single thread.
	*end = *start;
}
void Worklist::clearHost() {
	setSize(0);
}
__device__ void Worklist::myItems(unsigned &mystart, unsigned &myend) {
	unsigned nblocks = gridDim.x;
	unsigned nthreadsperblock = blockDim.x;
	unsigned nthreads = nblocks * nthreadsperblock;

	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned hsize = count();

	if (nthreads > hsize) {
		// each thread gets max 1 item.
		if (id < hsize) {
			mystart = id; 
			myend = mystart + 1;	// one item.
		} else {
			mystart = id; 
			myend = mystart;	// no items.
		}
	} else {
		unsigned nitemsperthread = hsize / nthreads;	// every thread gets at least these many.
		unsigned nitemsremaining = hsize % nthreads;	// initial some threads get one more.
		mystart = id * nitemsperthread; 
		myend = mystart + nitemsperthread;

		if (id < nitemsremaining) {
			mystart += id;			// initial few threads get one extra item, due to which
			myend   += id + 1;		// their assignment changes.
		} else {
			mystart += nitemsremaining;	// the others don't get anything extra, but
			myend   += nitemsremaining;	// their assignment changes.
		}
	}
}
__device__ unsigned Worklist::getItem(unsigned at) {
	unsigned hsize = count();
	return getItemWithin(at, hsize);
}
__device__ unsigned Worklist::getItemWithin(unsigned at, unsigned hsize) {
	if (at < hsize) {
		return items[at];
	}
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	printf("%s(%d): thread %d: buffer overflow, extracting %d when buffer size is %d.\n", __FILE__, __LINE__, id, at, hsize);
	return 1;
}
__device__ unsigned Worklist::count() {
	if (*end >= *start) {
		return *end - *start;
	} else {
		return *end + (*capacity - *start + 1);
	}
}

#define SWAPDEV(a, b)	{ unsigned tmp = a; a = b; b = tmp; }

__global__ void compress(Worklist wl, unsigned wlsize, unsigned sentinel) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ unsigned shmem[MAXSHAREDUINT];

	// copy my elements to my ids in shmem.
	unsigned wlstart = MAXSHAREDUINT * blockIdx.x + SHAREDPERTHREAD * threadIdx.x;
	unsigned shstart = SHAREDPERTHREAD * threadIdx.x;

	for (unsigned ii = 0; ii < SHAREDPERTHREAD; ++ii) {
		if (wlstart + ii < wlsize && shstart + ii < MAXSHAREDUINT) {
			shmem[shstart + ii] = wl.getItemWithin(wlstart + ii, wlsize);
		}
	}
	__syncthreads();
	
	
	// sort in shmem.
	for (unsigned s = blockDim.x / 2; s; s >>= 1) {
		if (threadIdx.x < s) {
			if (shmem[threadIdx.x] > shmem[threadIdx.x + s]) {
				SWAPDEV(shmem[threadIdx.x], shmem[threadIdx.x + s]);
			}
		}
		__syncthreads();
	}
	__syncthreads();

	// uniq in shmem.
	// TODO: find out how to do uniq in a hierarchical manner.
	unsigned lastindex = 0;
	if (id == 0) {
		for (unsigned ii = 1; ii < MAXSHAREDUINT; ++ii) {
			if (shmem[ii] != shmem[lastindex]) {
				shmem[++lastindex] = shmem[ii];
			} else {
				shmem[ii] = sentinel;
			}
		}
	}
	__syncthreads();

	// copy back elements to the worklist.
	for (unsigned ii = 0; ii < SHAREDPERTHREAD; ++ii) {
		if (wlstart + ii < wlsize) {
			//shmem[shstart + ii] = getItem(wlstart + ii);
			wl.items[wlstart + ii] = shmem[shstart + ii];
		}
	}
	__syncthreads();

	// update worklist indices.
	if (id == 0) {
		*wl.start = 0;
		*wl.end = lastindex + 1;
	}
}
void Worklist::compressHost(unsigned wlsize, unsigned sentinel) {
	unsigned nblocks = (wlsize + MAXBLOCKSIZE - 1) / MAXBLOCKSIZE;
	compress<<<nblocks, MAXBLOCKSIZE>>>(*this, wlsize, sentinel);
	CudaTest("compress failed");
}
__global__ void printWorklist(Worklist wl) {
	unsigned start, end;
	start = *wl.start;
	end = *wl.end;
	printf("\t");
	for (unsigned ii = start; ii < end; ++ii) {
		printf("%d,", wl.getItem(ii));
	}
	printf("\n");
}
void Worklist::printHost() {
	printWorklist<<<1, 1>>>(*this);
	CudaTest("printWorklist failed");
}
__global__ void appendWorklist(Worklist dst, Worklist src, unsigned dstsize) {
	unsigned start, end;
	src.myItems(start, end);

	for (unsigned ii = start; ii < end; ++ii) {
		dst.items[dstsize + ii] = src.items[ii];
	}
}
unsigned Worklist::appendHost(Worklist *otherlist) {
	unsigned otherlistsize = otherlist->getSize();
	unsigned nblocks = (otherlistsize + MAXBLOCKSIZE - 1) / MAXBLOCKSIZE;
	appendWorklist<<<nblocks, MAXBLOCKSIZE>>>(*this, *otherlist, getSize());
	CudaTest("appendWorklist failed");
	unsigned hstart, hend;
	getStartEnd(hstart, hend);
	setStartEnd(hstart, hend + otherlistsize);

	return hend + otherlistsize;
}
