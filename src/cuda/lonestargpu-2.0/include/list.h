#ifndef LSG_LIST
#define LSG_LIST

typedef struct List {
	__device__ List(unsigned size);
	__device__ void init(unsigned *mem, unsigned size, unsigned cap);
	__device__ void push(unsigned item);
	__device__ unsigned *toArray();
	__device__ void clear();
	__device__ unsigned size();
	__device__ void uniq(unsigned *mark, unsigned maxelement);

	unsigned *array;
	unsigned nitems;
	unsigned capacity;
} List;

__device__ List::List(unsigned size) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	capacity = 0;
	array = NULL;
	nitems = 0;

	if (size) {
		array = (unsigned *)malloc(size * sizeof(unsigned));
		if (array == NULL) {
			printf("%s(%d): thread %d: Error: malloc of %d unsigned returned no memory.\n", __FILE__, __LINE__, id, size);
		} else {
			capacity = size;
		}
	}
}
__device__ void List::init(unsigned *mem, unsigned size, unsigned cap) {
	array = mem;
	nitems = size;
	capacity = cap;
}
__device__ void List::push(unsigned item) {
	if (array && nitems < capacity) {
		array[nitems++] = item;
	} else {
		unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
		printf("%s(%d): thread %d: Error: buffer overflow, capacity=%d.\n", __FILE__, __LINE__, id, capacity);
	}
}
__device__ unsigned *List::toArray() {
	return array;
}
__device__ void List::clear() {
	if (array) free(array);
	nitems = 0;
	capacity = 0;
}
__device__ unsigned List::size() {
	return nitems;
}
__device__ void List::uniq(unsigned *mark, unsigned maxelement) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned mysize = size();
	if (mysize == 0) return;

	unsigned *newarray = (unsigned *)malloc(mysize * sizeof(unsigned));
	if (newarray == NULL) {
		printf("%s(%d): thread %d: Error: malloc of %d unsigned returned no memory.\n", __FILE__, __LINE__, id, mysize);
		return;
	}
	unsigned *insertptr = newarray;

	for (unsigned ii = 0; ii < mysize; ++ii) {
		unsigned element = array[ii];
		if (element < maxelement && mark[element] == id) {	// this thread didn't succeed in marking this element.
			*insertptr++ = element;
		}
	}
	clear();
	init(newarray, insertptr - newarray, mysize);
}
#endif
