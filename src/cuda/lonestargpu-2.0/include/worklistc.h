#pragma once

#include "cutil_subset.h"

static int zero = 0;

struct Worklist {
  int *dwl, *wl;
  int length, *dnsize;
  int *dindex;

  Worklist(size_t nsize)
  {
    wl = (int *) calloc(nsize, sizeof(int));
    CUDA_SAFE_CALL(cudaMalloc(&dwl, nsize * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc(&dnsize, 1 * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc(&dindex, 1 * sizeof(int)));

    CUDA_SAFE_CALL(cudaMemcpy(dnsize, &nsize, 1 * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy((void *) dindex, &zero, 1 * sizeof(zero), cudaMemcpyHostToDevice));
  }

  ~Worklist()
  {
    //CUDA_SAFE_CALL(cudaFree(dwl));
  }

  void update_cpu()
  {
    int nsize = nitems();
    CUDA_SAFE_CALL(cudaMemcpy(wl, dwl, nsize  * sizeof(int), cudaMemcpyDeviceToHost));
  }

  void display_items()
  {
    int nsize = nitems();
    CUDA_SAFE_CALL(cudaMemcpy(wl, dwl, nsize  * sizeof(int), cudaMemcpyDeviceToHost));

    printf("WL: ");
    for(int i = 0; i < nsize; i++)
      printf("%d %d, ", i, wl[i]);

    printf("\n");
    return;
  }

  void reset()
  {
    CUDA_SAFE_CALL(cudaMemcpy((void *) dindex, &zero, 1 * sizeof(zero), cudaMemcpyHostToDevice));
  }

  int nitems()
  {
    int index;
    CUDA_SAFE_CALL(cudaMemcpy(&index, (void *) dindex, 1 * sizeof(index), cudaMemcpyDeviceToHost));

    return index;
  }

  __device__ 
  int push(int item)
  {
    int lindex = atomicAdd((int *) dindex, 1);
    if(lindex >= *dnsize)
      return 0;

    dwl[lindex] = item;
    return 1;
  }

  __device__
  int pop(int &item)
  {
    int lindex = atomicSub((int *) dindex, 1);
    if(lindex <= 0)
      {
	*dindex = 0;
	return 0;
      }

    item = dwl[lindex - 1];
    return 1;
  }
};

struct Worklist2: public Worklist {
  Worklist2(int nsize) : Worklist(nsize) {}

  template <typename T>
  __device__ __forceinline__
    int push_1item(int nitem, int item, int threads_per_block)
  {
    __shared__ typename T::TempStorage temp_storage;
    __shared__ int queue_index;
    int total_items = 0;
    int thread_data = nitem;

    T(temp_storage).ExclusiveSum(thread_data, thread_data, total_items);

    if(threadIdx.x == 0)
      {	
	if(debug) printf("t: %d\n", total_items);
	queue_index = atomicAdd((int *) dindex, total_items);
	//printf("queueindex: %d %d %d %d %d\n", blockIdx.x, threadIdx.x, queue_index, thread_data + n_items, total_items);
      }

    __syncthreads();

    if(nitem == 1)
      {
	if(queue_index + thread_data >= *dnsize)
	  {
	    printf("GPU: exceeded length: %d %d %d\n", queue_index, thread_data, *dnsize);
	    return 0;
	  }

	//dwl[queue_index + thread_data] = item;
	cub::ThreadStore<cub::STORE_CG>(dwl + queue_index + thread_data, item);
      }

    return total_items;
  }

  template <typename T>
  __device__ __forceinline__
  int push_nitems(int n_items, int *items, int threads_per_block)
  {
    __shared__ typename T::TempStorage temp_storage;
    __shared__ int queue_index;
    int total_items;

    int thread_data = n_items;

    T(temp_storage).ExclusiveSum(thread_data, thread_data, total_items);

    if(threadIdx.x == 0)
      {	
	queue_index = atomicAdd((int *) dindex, total_items);
	//printf("queueindex: %d %d %d %d %d\n", blockIdx.x, threadIdx.x, queue_index, thread_data + n_items, total_items);
      }

    __syncthreads();

    
    for(int i = 0; i < n_items; i++)
      {
	//printf("pushing %d to %d\n", items[i], queue_index + thread_data + i);
	if(queue_index + thread_data + i >= *dnsize)
	  {
	    printf("GPU: exceeded length: %d %d %d %d\n", queue_index, thread_data, i, *dnsize);
	    return 0;
	  }

	dwl[queue_index + thread_data + i] = items[i];
      }

    return total_items;
  }

  __device__ 
  int pop_id(int id, int &item)
  {
    if(id < *dindex)
      {
	item = cub::ThreadLoad<cub::LOAD_CG>(dwl + id);
	//item = dwl[id];
	return 1;
      }
    
    return 0;
  }  
};

