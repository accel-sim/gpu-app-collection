#define SSSP_VARIANT "worklistc"

#define MAXDIST		100

#define AVGDEGREE	2.5
#define WORKPERTHREAD	1

unsigned int NVERTICES;

#include <cub/cub.cuh>
#include "worklistc.h"
#include "gbar.cuh"
#include "cutil_subset.h"

const int BLKSIZE = 512;

struct workprogress 
{
  Worklist2 *wl[2];
  int in_wl;
  int iteration;
};

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
foru processedge2(foru *dist, Graph &graph, unsigned iteration, unsigned src, unsigned edge, unsigned &dst) {
  
  dst = tex1Dfetch(columns, edge);
  if (dst >= graph.nnodes) return 0;

  foru wt = graph.edgessrcwt[edge];
  if (wt >= MYINFINITY) return 0;

  foru dstwt = cub::ThreadLoad<cub::LOAD_CG>(dist + dst);
  foru altdist = cub::ThreadLoad<cub::LOAD_CG>(dist + src) + wt;  

  //printf("%d %d %d %d %d\n", src, dst, wt, dstwt, altdist);

  if(altdist < dstwt)
    {
      atomicMin(&dist[dst], altdist);
      return 1;

      /* foru olddist = atomicMin(&dist[dst], altdist); */
      /* if (altdist < olddist) { */
      /* 	return olddist; */
      /* }  */
    }
  
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
	  if(nn != -1)
	    {
	      neighborsize = nnsize = graph.getOutDegree(nn);
	      neighboroffset = tex1Dfetch(row_offsets, graph.srcsrc[nn]);
	    }
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
		if(processedge2(dist, graph, iteration, shnn, neighboroffset + i, to_push))
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
  __shared__ int src[SCRATCHSIZE];

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
	      src[scratch_offset + i - done] = nn;
	    }

	  neighborsdone += i;
	  scratch_offset += i;

	  __syncthreads();

	  int ncnt = 0;
	  unsigned to_push = 0;

	  if(threadIdx.x < total_edges)
	    {
	      if(processedge2(dist, graph, iteration, src[threadIdx.x], gather_offsets[threadIdx.x], to_push))
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

__global__ void drelax3(foru *dist, Graph graph, unsigned *gerrno, Worklist2 inwl, Worklist2 outwl, int iteration, GlobalBarrier gb)
{
  drelax(dist, graph, gerrno, inwl, outwl, iteration);
}

__global__ void drelax2(foru *dist, Graph graph, unsigned *gerrno, Worklist2 inwl, Worklist2 outwl, struct workprogress *wp, GlobalBarrier gb)
{
  wp->wl[0] = &inwl;
  wp->wl[1] = &outwl;
  int iteration = wp->iteration;
  int max_iteration = iteration + 200;

  if(wp->iteration == 0)
    {
      drelax(dist, graph, gerrno, *wp->wl[0], *wp->wl[1], wp->iteration);
      wp->iteration++;
    }
  else
    {
      Worklist2 *in;
      Worklist2 *out;
      int in_wl;

      in_wl = wp->in_wl;

      in = wp->wl[in_wl]; out = wp->wl[1 - in_wl];

      while(*in->dindex > 0 && iteration < max_iteration) // && iteration < 30)
	{
	  drelax(dist, graph, gerrno, *in, *out, wp->iteration);

	  //__threadfence_system();
	  gb.Sync();

	  /* tmp = in; */
	  /* in = out; */
	  /* out = tmp; */

	  in_wl = 1 - in_wl;
	  in = wp->wl[in_wl];  
	  out = wp->wl[1 - in_wl];

	  *out->dindex = 0;

	  iteration++;
	}

      wp->iteration = iteration;
      wp->in_wl = in_wl;
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

__global__ void remove_dups(Worklist2 wl, int *node_owner, GlobalBarrier gb)
{
  unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
  int nn;
  
  int total_inputs = (*wl.dindex + gridDim.x * blockDim.x - 1)/(gridDim.x * blockDim.x);
  
  while(total_inputs-- > 0)
    {      
      if(wl.pop_id(id, nn))
	{
	  node_owner[nn] = id;
	}

      id += gridDim.x * blockDim.x;
    }

  id = blockIdx.x * blockDim.x + threadIdx.x;
  total_inputs = (*wl.dindex + gridDim.x * blockDim.x - 1)/(gridDim.x * blockDim.x);

  gb.Sync();
  
  while(total_inputs-- > 0)
    { 
      if(wl.pop_id(id, nn))
	{
	  if(node_owner[nn] != id)
	    wl.dwl[id] = -1;
	}

      id += gridDim.x * blockDim.x;    
    }
}

void sssp(foru *hdist, foru *dist, Graph &graph, unsigned long totalcomm)
{
	foru foruzero = 0.0;
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

	struct workprogress hwp, *dwp;

	hwp.wl[0] = NULL;
	hwp.wl[1] = NULL;
	hwp.in_wl = 0;
	hwp.iteration = 1;

	int *node_owners;
	CUDA_SAFE_CALL(cudaMalloc(&node_owners, graph.nnodes * sizeof(int)));

	CUDA_SAFE_CALL(cudaMalloc(&dwp, sizeof(hwp)));
	CUDA_SAFE_CALL(cudaMemcpy(dwp, &hwp, sizeof(*dwp), cudaMemcpyHostToDevice));


	cudaBindTexture(0, columns, graph.edgessrcdst, (graph.nedges + 1) * sizeof(int));
	cudaBindTexture(0, row_offsets, graph.psrc, (graph.nnodes + 1) * sizeof(int));

	//print_array<<<1, graph.nedges + 1>>>((int *) graph.edgessrcdst, graph.nedges + 1);
	//print_texture<<<1, graph.nedges + 1>>>(graph.nedges + 1);
	//return;


	/* currently not used due to ensuing launch timeouts*/
	GlobalBarrierLifetime gb;
	gb.Setup(28);

	printf("solving.\n");
	printf("starting...\n");
	//printf("worklist size: %d\n", inwl->nitems());
	//printf("WL: 0 0, \n");

	starttime = rtclock();
	drelax3<<<1, BLKSIZE>>>(dist, graph, nerr, *inwl, *outwl, 0, gb);

	do {
	        ++iteration;
		unsigned nblocks = (nitems + BLKSIZE - 1) / BLKSIZE; 
		//printf("%d %d %d %d\n", nblocks, BLKSIZE, iteration, nitems);
		//printf("ITERATION: %d\n", iteration);
		//inwl->display_items();
		//drelax2 <<<14, BLKSIZE>>> (dist, graph, nerr, *inwl, *outwl, dwp, gb);
		
		drelax3 <<<nblocks, BLKSIZE>>> (dist, graph, nerr, *inwl, *outwl, iteration, gb);
		nitems = outwl->nitems();

		//remove_dups<<<14, 1024>>>(*outwl, node_owners, gb);
		
		//printf("%d\n", iteration);
		//outwl->display_items();

		//printf("worklist size: %d\n", nitems);
		
		Worklist2 *tmp = inwl;
		inwl = outwl;
		outwl = tmp;

		outwl->reset();
	} while (nitems > 0);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	endtime = rtclock();

	CUDA_SAFE_CALL(cudaMemcpy(&hwp, dwp, sizeof(hwp), cudaMemcpyDeviceToHost));

	printf("\titerations = %d %d.\n", iteration, hwp.iteration);
	runtime = (1000.0f * (endtime - starttime));
	printf("\truntime [%s] = %f ms.\n", SSSP_VARIANT, runtime);

	return;
}
