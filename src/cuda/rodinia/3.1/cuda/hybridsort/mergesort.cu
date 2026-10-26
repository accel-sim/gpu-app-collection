////////////////////////////////////////////////////////////////////////////////
// Includes
////////////////////////////////////////////////////////////////////////////////
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "mergesort.cuh"
#include "mergesort_kernel.cu"
////////////////////////////////////////////////////////////////////////////////
// Defines
////////////////////////////////////////////////////////////////////////////////
#define BLOCKSIZE	256
#define ROW_LENGTH	BLOCKSIZE * 4
#define ROWS		4096

static cudaTextureObject_t h_tex = 0;

/* The list texture is rebound after every level swap, so the previous
   object has to be destroyed before a new one replaces it. */
static void bindListTexture(float4 *list, cudaChannelFormatDesc channelDesc, size_t bytes)
{
	if (h_tex) {
		cudaDestroyTextureObject(h_tex);
		h_tex = 0;
	}
	cudaResourceDesc resDesc;
	cudaTextureDesc  texDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	memset(&texDesc, 0, sizeof(texDesc));
	resDesc.resType = cudaResourceTypeLinear;
	resDesc.res.linear.devPtr = list;
	resDesc.res.linear.desc = channelDesc;
	resDesc.res.linear.sizeInBytes = bytes;
	texDesc.readMode = cudaReadModeElementType;
	cudaCreateTextureObject(&h_tex, &resDesc, &texDesc, NULL);
	cudaMemcpyToSymbol(tex, &h_tex, sizeof(h_tex));
}

////////////////////////////////////////////////////////////////////////////////
// The mergesort algorithm
////////////////////////////////////////////////////////////////////////////////
float4* runMergeSort(int listsize, int divisions, 
				     float4 *d_origList, float4 *d_resultList, 
				     int *sizes, int *nullElements,
					 unsigned int *origOffsets)
{
	int *startaddr = (int *)malloc((divisions + 1)*sizeof(int)); 
	int largestSize = -1; 
	startaddr[0] = 0; 
	for(int i=1; i<=divisions; i++)
	{
		startaddr[i] = startaddr[i-1] + sizes[i-1];
		if(sizes[i-1] > largestSize) largestSize = sizes[i-1]; 
	}
	largestSize *= 4; 

	// Setup texture
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);

	////////////////////////////////////////////////////////////////////////////
	// First sort all float4 elements internally
	////////////////////////////////////////////////////////////////////////////
	#ifdef MERGE_WG_SIZE_0
	const int THREADS = MERGE_WG_SIZE_0;
	#else
	const int THREADS = 256; 
	#endif
	dim3 threads(THREADS, 1);
	int blocks = ((listsize/4)%THREADS == 0) ? (listsize/4)/THREADS : (listsize/4)/THREADS + 1; 
	dim3 grid(blocks, 1);
	bindListTexture(d_origList, channelDesc, listsize*sizeof(float));
	mergeSortFirst<<< grid, threads >>>(d_resultList, listsize);

	////////////////////////////////////////////////////////////////////////////
	// Then, go level by level
	////////////////////////////////////////////////////////////////////////////
	cudaMemcpyToSymbol(constStartAddr, startaddr, (divisions + 1)*sizeof(int)); 
	cudaMemcpyToSymbol(finalStartAddr, origOffsets, (divisions + 1)*sizeof(int)); 
	cudaMemcpyToSymbol(nullElems, nullElements, (divisions)*sizeof(int)); 
	int nrElems = 2;
	while(true){
		int floatsperthread = (nrElems*4); 
		int threadsPerDiv = (int)ceil(largestSize/(float)floatsperthread); 
		int threadsNeeded = threadsPerDiv * divisions; 
		#ifdef MERGE_WG_SIZE_1
		threads.x = MERGE_WG_SIZE_1;
		#else
		threads.x = 208; 
		#endif
		grid.x = ((threadsNeeded%threads.x) == 0) ?
			threadsNeeded/threads.x : 
			(threadsNeeded/threads.x) + 1; 
		if(grid.x < 8){
			grid.x = 8; 
			threads.x = ((threadsNeeded%grid.x) == 0) ? 
				threadsNeeded / grid.x : 
				(threadsNeeded / grid.x) + 1; 
		}
		// Swap orig/result list
		float4 *tempList = d_origList; 
		d_origList = d_resultList; 
		d_resultList = tempList; 
		bindListTexture(d_origList, channelDesc, listsize*sizeof(float));
		mergeSortPass <<< grid, threads >>>(d_resultList, nrElems, threadsPerDiv); 
		nrElems *= 2; 
		floatsperthread = (nrElems*4); 
		if(threadsPerDiv == 1) break; 
	}
	////////////////////////////////////////////////////////////////////////////
	// Now, get rid of the NULL elements
	////////////////////////////////////////////////////////////////////////////
	#ifdef MERGE_WG_SIZE_0
	threads.x = MERGE_WG_SIZE_0;
	#else
	threads.x = 256; 
	#endif
	grid.x = ((largestSize%threads.x) == 0) ?
			largestSize/threads.x : 
			(largestSize/threads.x) + 1; 
	grid.y = divisions; 
	mergepack <<< grid, threads >>> ((float *)d_resultList, (float *)d_origList);

	free(startaddr);
	if (h_tex) {
		cudaDestroyTextureObject(h_tex);
		h_tex = 0;
	}
	return d_origList; 
}
