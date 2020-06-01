
#include <stdio.h> 
#include <string>
#include <deque>
#include <vector>
#include <iostream>

#define BFS_VARIANT "merrill"
#include "cutil_subset.h"

#include <b40c_test_util.h>

#include <b40c/graph/builder/dimacs.cuh>
/* #include <b40c/graph/builder/grid2d.cuh> */
/* #include <b40c/graph/builder/grid3d.cuh> */
/* #include <b40c/graph/builder/market.cuh> */
/* #include <b40c/graph/builder/metis.cuh> */
/* #include <b40c/graph/builder/rmat.cuh> */
/* #include <b40c/graph/builder/random.cuh> */
/* #include <b40c/graph/builder/rr.cuh> */

// BFS includes
#include <b40c/graph/bfs/csr_problem.cuh>
#include <b40c/graph/csr_graph.cuh>
#include <b40c/graph/bfs/enactor_contract_expand.cuh>
#include <b40c/graph/bfs/enactor_expand_contract.cuh>
#include <b40c/graph/bfs/enactor_two_phase.cuh>
#include <b40c/graph/bfs/enactor_hybrid.cuh>
#include <b40c/graph/bfs/enactor_multi_gpu.cuh>

using namespace b40c;
using namespace graph;

//texture<char, cudaTextureType1D, cudaReadModeElementType> bitmask_tex_ref;

void bfs_merrill(Graph &graph, foru *dist)
{
  typedef int VertexId ;
  typedef foru Value ;
  typedef int SizeT ;

  graph::CsrGraph<VertexId, Value, SizeT> csr_graph;

  csr_graph.FromScratch<true>(graph.nnodes, graph.nedges);
  
  cudaMemcpy(csr_graph.column_indices, graph.edgessrcdst + 1, 
	     sizeof(VertexId) * graph.nedges, cudaMemcpyDeviceToHost);

  CudaTest("copying column indices failed");

  cudaMemcpy(csr_graph.row_offsets, graph.psrc, 
	     sizeof(SizeT) * (graph.nnodes + 1), cudaMemcpyDeviceToHost);

  CudaTest("copying row offsets failed");

  // TODO: avoid maintaining two copies of the graph in GPU memory
  //graph.dealloc();

  for(int i = 0; i < graph.nnodes; i++)
    {
      csr_graph.row_offsets[i] -= 1;
      //printf("%d ", csr_graph.row_offsets[i]);
    }
  printf("Translated back to 0-index \n");

  typedef bfs::CsrProblem<VertexId, SizeT, false> CsrProblem;

  bfs::EnactorHybrid<false> hybrid(false);

  CsrProblem csr_problem;
  if (csr_problem.FromHostProblem(
				  false,
				  csr_graph.nodes,
				  csr_graph.edges,
				  csr_graph.column_indices,
				  csr_graph.row_offsets,
				  1)) exit(1);

  cudaError_t		retval = cudaSuccess;

  VertexId src = 0;
  
  double starttime, endtime;

  starttime = rtclock();
  if (retval = csr_problem.Reset(hybrid.GetFrontierType(), 1.3))
    return; 
  
  if (retval = hybrid.EnactSearch(csr_problem, src, 0))
    {
      if (retval && (retval != cudaErrorInvalidDeviceFunction)) {
  	exit(1);
      }
    }
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  endtime = rtclock();

  printf("\truntime [%s] = %f ms.\n", 
	 BFS_VARIANT, 1000 * (endtime - starttime));


  foru *h_dist;
  h_dist = (foru *) malloc(graph.nnodes * sizeof(foru));
  assert(h_dist != NULL);

  if (csr_problem.ExtractResults((int *) h_dist)) exit(1);

  for(int i = 0; i < graph.nnodes; i++)
    if((signed) h_dist[i] == -1)
      h_dist[i] = MYINFINITY;

  CUDA_SAFE_CALL(cudaMemcpy(dist, h_dist, graph.nnodes * sizeof(foru), cudaMemcpyHostToDevice));

  free(h_dist);

  printf("done!\n");
}
