#define BFS_VARIANT "worklistc"

#define MAXDIST		100

#define AVGDEGREE	2.5
#define WORKPERTHREAD	1

unsigned int NVERTICES;

#include <cub/cub.cuh>
#include "worklistc.h"
#include "gbar.cuh"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"

const int BLKSIZE = 704;
const int IN_CORE = 1;     // set this to zero to disable global-barrier version

texture <int, 1, cudaReadModeElementType> columns;
texture <int, 1, cudaReadModeElementType> row_offsets;

__global__
void initialize(foru *dist, unsigned int nv) {
	unsigned int ii = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("ii=%d, nv=%d.\n", ii, *nv);
	if (ii < nv) {
		dist[ii] = MYINFINITY;
	}
}

__device__
foru processedge2(foru *dist, Graph &graph, unsigned iteration, unsigned edge, unsigned &dst) {
  
  dst = tex1Dfetch(columns, edge);

  if (dst >= graph.nnodes) return 0;

  foru wt = 1;	//graph.getWeight(src, ii);
  if (wt >= MYINFINITY) return 0;

  if(cub::ThreadLoad<cub::LOAD_CG>(dist + dst) == MYINFINITY)
    {
      cub::ThreadStore<cub::STORE_CG>(dist + dst, iteration);
      return MYINFINITY;
    }
  
  return 0;
}

__device__
foru processedge(foru *dist, Graph &graph, unsigned src, unsigned ii, unsigned &dst) {
	//unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	//dst = graph.getDestination(src, ii);	

  // no bounds checking here
  dst = tex1Dfetch(columns, tex1Dfetch(row_offsets, graph.srcsrc[src]) + ii);

	//printf("%d %d %d %d\n", dst, tex1Dfetch(columns, tex1Dfetch(row_offsets, graph.srcsrc[src]) + ii));


	if (dst >= graph.nnodes) return 0;

	foru wt = 1;	//graph.getWeight(src, ii);
	if (wt >= MYINFINITY) return 0;

	//printf("%d %d %d %d\n", src, dst, dist[src], dist[dst]);

	foru altdist = cub::ThreadLoad<cub::LOAD_CG>(dist + src) + wt;
	 
	 if(cub::ThreadLoad<cub::LOAD_CG>(dist + dst) == MYINFINITY)
	   {
	     cub::ThreadStore<cub::STORE_CG>(dist + dst, altdist);
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

__device__ void expandByCTA(foru *dist, Graph &graph, Worklist2 &inwl, Worklist2 &outwl, unsigned iteration)
{
  unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
  int nn;

  typedef cub::BlockScan<int, BLKSIZE> BlockScan;

  __shared__ int owner;
  __shared__ int shnn;

  int total_inputs = (*inwl.dindex + gridDim.x * blockDim.x - 1)/(gridDim.x * blockDim.x);

  owner = -1;

  while(total_inputs-- > 0)
    {      
      int neighborsize = 0;
      int neighboroffset = 0;
      int nnsize = 0;

      if(inwl.pop_id(id, nn))
	{	  
	  neighborsize = nnsize = graph.getOutDegree(nn);
	  neighboroffset = tex1Dfetch(row_offsets, graph.srcsrc[nn]);	  
	}

      while(true)
	{
	  if(nnsize > BLKSIZE)
	    owner = threadIdx.x;

	  __syncthreads();
	  
	  if(owner == -1)
	    break;

	  if(owner == threadIdx.x)
	    {
	      shnn = nn;
	      cub::ThreadStore<cub::STORE_CG>(inwl.dwl + id, -1);
	      owner = -1;
	      nnsize = 0;
	    }

	  __syncthreads();

	  neighborsize = graph.getOutDegree(shnn);
	  neighboroffset = tex1Dfetch(row_offsets, graph.srcsrc[shnn]);
	  int xy = ((neighborsize + blockDim.x - 1) / blockDim.x) * blockDim.x;
	  
	  for(int i = threadIdx.x; i < xy; i+= blockDim.x)
	    {
	      int ncnt = 0;
	      unsigned to_push = 0;

	      if(i < neighborsize)
		if(processedge2(dist, graph, iteration, neighboroffset + i, to_push))
		  {
		    ncnt = 1;
		  }
	    
	      outwl.push_1item<BlockScan>(ncnt, (int) to_push, BLKSIZE);
	    }
	}

      id += gridDim.x * blockDim.x;
    }
}
__device__
unsigned processnode2(foru *dist, Graph &graph, Worklist2 &inwl, Worklist2 &outwl, unsigned iteration) 
{
  //expandByCTA(dist, graph, inwl, outwl, iteration);

  unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
  int nn;

  typedef cub::BlockScan<int, BLKSIZE> BlockScan;

  const int SCRATCHSIZE = BLKSIZE;
  __shared__ BlockScan::TempStorage temp_storage;
  __shared__ int gather_offsets[SCRATCHSIZE];

  gather_offsets[threadIdx.x] = 0;

  int total_inputs = (*inwl.dindex + gridDim.x * blockDim.x - 1)/(gridDim.x * blockDim.x);
  
  while(total_inputs-- > 0)
    {      
      int neighborsize = 0;
      int neighboroffset = 0;
      int scratch_offset = 0;
      int total_edges = 0;

      if(inwl.pop_id(id, nn))
	{	  
	  if(nn != -1)
	    {
	      neighborsize = graph.getOutDegree(nn);
	      neighboroffset = tex1Dfetch(row_offsets, graph.srcsrc[nn]);
	    }
	}

      BlockScan(temp_storage).ExclusiveSum(neighborsize, scratch_offset, total_edges);
  
      int done = 0;
      int neighborsdone = 0;

      /* if(total_edges) */
      /* 	if(threadIdx.x == 0) */
      /* 	  printf("total edges: %d\n", total_edges); */

      while(total_edges > 0)
	{
	  __syncthreads();

	  int i;
	  for(i = 0; neighborsdone + i < neighborsize && (scratch_offset + i - done) < SCRATCHSIZE; i++)
	    {
	      gather_offsets[scratch_offset + i - done] = neighboroffset + neighborsdone + i;
	    }

	  neighborsdone += i;
	  scratch_offset += i;

	  __syncthreads();

	  int ncnt = 0;
	  unsigned to_push = 0;

	  if(threadIdx.x < total_edges)
	    {
	      if(processedge2(dist, graph, iteration, gather_offsets[threadIdx.x], to_push))
		{
		  ncnt = 1;
		}
	    }

	  outwl.push_1item<BlockScan>(ncnt, (int) to_push, BLKSIZE);
      
	  total_edges -= BLKSIZE;
	  done += BLKSIZE;
	}

      id += blockDim.x * gridDim.x;
    }

  return 0;
}

__device__
unsigned processnode(foru *dist, Graph &graph, Worklist2 &inwl, Worklist2 &outwl) 
{
  unsigned id = blockIdx.x * blockDim.x + threadIdx.x;

  int neighbours[256];
  int ncnt = 0;
  int nn;

  if(inwl.pop_id(id, nn))
	{
	  unsigned neighborsize = graph.getOutDegree(nn);

	  if(neighborsize > 256)
	  	printf("whoa! out of local space");
	  
	  for (unsigned ii = 0; ii < neighborsize; ++ii) {
		unsigned dst = graph.nnodes;
		foru olddist = processedge(dist, graph, nn, ii, dst);
		if (olddist) {
		  //printf("%d pushing %d\n", nn, dst);
		  neighbours[ncnt] = dst;
		  ncnt++;
		}
      }
    }

  typedef cub::BlockScan<int, BLKSIZE> BlockScan;

  return outwl.push_nitems<BlockScan>(ncnt, neighbours, BLKSIZE) == 0 && ncnt > 0;
}

__device__
void drelax(foru *dist, Graph& graph, unsigned *gerrno, Worklist2 &inwl, Worklist2& outwl, int iteration) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;

	if(iteration == 0)
	  {
	    if(id == 0)
	      {
			int item = 0;
			inwl.push(item);
	      }
	    return;	    
	  }
	else
	  {
	    if(processnode2(dist, graph, inwl, outwl, iteration))
	      *gerrno = 1;
	  }
}

__global__ void drelax3(foru *dist, Graph graph, unsigned *gerrno, Worklist2 inwl, Worklist2 outwl, int iteration)
{
  drelax(dist, graph, gerrno, inwl, outwl, iteration);
}

__global__ void drelax2(foru *dist, Graph graph, unsigned *gerrno, Worklist2 inwl, Worklist2 outwl, int iteration, GlobalBarrier gb)
{
  if(iteration == 0)
    drelax(dist, graph, gerrno, inwl, outwl, iteration);
  else
    {
      Worklist2 *in;
      Worklist2 *out;
      Worklist2 *tmp;

      in = &inwl; out = &outwl;

      while(*in->dindex > 0) // && iteration < 30)
	{
	  drelax(dist, graph, gerrno, *in, *out, iteration);

	  //__threadfence_system();
	  gb.Sync();

	  tmp = in;
	  in = out;
	  out = tmp;

	  *out->dindex = 0;

	  iteration++;
	}
    }
}
__global__ void print_array(int *a, int n)
{
  int id = threadIdx.x + blockDim.x * blockIdx.x;
  if(id < n)
    printf("%d %d\n", id, a[id]);
}

__global__ void print_texture(int n)
{
  int id = threadIdx.x + blockDim.x * blockIdx.x;
  if(id < n)
    printf("%d %d\n", id, tex1Dfetch(columns, id));
}

void bfs(Graph &graph, foru *dist)
{
	foru foruzero = 0;
	unsigned int NBLOCKS, FACTOR = 128;
	bool *changed;
	int iteration = 0;
	unsigned *nerr;

	double starttime, endtime;
	double runtime;

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	NBLOCKS = deviceProp.multiProcessorCount;

	NVERTICES = graph.nnodes;

	FACTOR = (NVERTICES + MAXBLOCKSIZE * NBLOCKS - 1) / (MAXBLOCKSIZE * NBLOCKS);

	//printf("initializing (nblocks=%d, blocksize=%d).\n", NBLOCKS*FACTOR, MAXBLOCKSIZE);
	initialize <<<NBLOCKS*FACTOR, MAXBLOCKSIZE>>> (dist, graph.nnodes);
	CudaTest("initializing failed");
	cudaMemcpy(&dist[0], &foruzero, sizeof(foruzero), cudaMemcpyHostToDevice);

	if (cudaMalloc((void **)&changed, sizeof(bool)) != cudaSuccess) CudaTest("allocating changed failed");
	if (cudaMalloc((void **)&nerr, sizeof(unsigned)) != cudaSuccess) CudaTest("allocating nerr failed");

	Worklist2 wl1(graph.nedges * 2), wl2(graph.nedges * 2);
	Worklist2 *inwl = &wl1, *outwl = &wl2;
	int nitems = 1;

	cudaBindTexture(0, columns, graph.edgessrcdst, (graph.nedges + 1) * sizeof(int));
	cudaBindTexture(0, row_offsets, graph.psrc, (graph.nnodes + 1) * sizeof(int));

	//print_array<<<1, graph.nedges + 1>>>((int *) graph.edgessrcdst, graph.nedges + 1);
	//print_texture<<<1, graph.nedges + 1>>>(graph.nedges + 1);
	//return;


	printf("solving.\n");
	printf("starting...\n");
	starttime = rtclock();

	if(IN_CORE) {
	  GlobalBarrierLifetime gb;
	  const size_t drelax2_max_blocks = maximum_residency(drelax2, BLKSIZE, 0);
	  gb.Setup(deviceProp.multiProcessorCount * drelax2_max_blocks);

	  drelax2<<<1, BLKSIZE>>>(dist, graph, nerr, *inwl, *outwl, 0, gb);
	  drelax2 <<<deviceProp.multiProcessorCount * drelax2_max_blocks, BLKSIZE>>> (dist, graph, nerr, *inwl, *outwl, 1, gb);
	}
	else {
	  drelax3<<<1, BLKSIZE>>>(dist, graph, nerr, *inwl, *outwl, 0);
	  nitems = inwl->nitems();

	  while(nitems > 0) {
	    ++iteration;
	    unsigned nblocks = (nitems + BLKSIZE - 1) / BLKSIZE; 
	    //printf("%d %d %d %d\n", nblocks, BLKSIZE, iteration, nitems);
	    //printf("ITERATION: %d\n", iteration);
	    //inwl->display_items();

	    drelax3<<<nblocks, BLKSIZE>>>(dist, graph, nerr, *inwl, *outwl, iteration);
  
	    nitems = outwl->nitems();

	    //printf("worklist size: %d\n", nitems);
		
	    Worklist2 *tmp = inwl;
	    inwl = outwl;
	    outwl = tmp;
	    
	    outwl->reset();
	  };  
	}
	
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	endtime = rtclock();
	
	printf("\titerations = %d.\n", iteration);
	runtime = (1000.0f * (endtime - starttime));
	printf("\truntime [%s] = %f ms.\n", BFS_VARIANT, runtime);

	return;
}
