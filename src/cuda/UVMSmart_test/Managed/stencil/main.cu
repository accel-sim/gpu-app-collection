
/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/
//#include <parboil.h>
#include <stdio.h>
#include <stdlib.h>

#include "file.h"
#include "common.h"
#include "cuerr.h"
#include "kernels.h"

static int read_data(float *A0, int nx,int ny,int nz,FILE *fp) 
{	
  int s=0;
  for(int i=0;i<nz;i++)
    {
      for(int j=0;j<ny;j++)
	{
	  for(int k=0;k<nx;k++)
	    {
	      fread(A0+s,sizeof(float),1,fp);
	      s++;
	    }
	}
    }
  return 0;
}

int main(int argc, char** argv) {
  printf("CUDA accelerated 7 points stencil codes****\n");
  printf("Original version by Li-Wen Chang <lchang20@illinois.edu> and I-Jui Sung<sung10@illinois.edu>\n");
  printf("This version maintained by Chris Rodrigues  ***********\n");
	
  //declaration
  int nx,ny,nz;
  int size;
  int iteration;
  float c0=1.0f/6.0f;
  float c1=1.0f/6.0f/6.0f;

  if (argc<6) 
    {
      printf("Usage: probe nx ny nz tx ty t\n"
	     "nx: the grid size x\n"
	     "ny: the grid size y\n"
	     "nz: the grid size z\n"
	     "t: the iteration time\n");
      return -1;
    }

  nx = atoi(argv[2]);
  if (nx<1)
    return -1;
  ny = atoi(argv[3]);
  if (ny<1)
    return -1;
  nz = atoi(argv[4]);
  if (nz<1)
    return -1;
  iteration = atoi(argv[5]);
  if(iteration<1)
    return -1;
	
  float *h_A0;
  float *h_Anext;

  size=nx*ny*nz;
	
  cudaMallocManaged(&h_A0, sizeof(float)*size);
  cudaMallocManaged(&h_Anext, sizeof(float)*size);
  memset(h_Anext,0.0,size*sizeof(float));  

  FILE *fp = fopen(argv[1], "rb");
  read_data(h_A0, nx,ny,nz,fp);
  fclose(fp);
  FILE *fp2 = fopen(argv[1], "rb");
  read_data(h_Anext, nx,ny,nz,fp2);
  fclose(fp2);

#ifdef PREF
  int device = -1;
  cudaGetDevice(&device);
  
  cudaStream_t stream1;
  cudaStreamCreate(&stream1);

  cudaStream_t stream2;
  cudaStreamCreate(&stream2);

  cudaStream_t stream3;
  cudaStreamCreate(&stream3);

  cudaMemPrefetchAsync(h_A0, sizeof(float)*size, device, stream1);
  cudaMemPrefetchAsync(h_Anext, sizeof(float)*size, device, stream2);
#endif
	
  //only use tx-by-ty threads
  int tx=32;
  int ty=4;
  
  dim3 block (tx, ty, 1);
  //also change threads size maping from tx by ty to 2tx x ty
  dim3 grid ((nx+tx*2-1)/(tx*2), (ny+ty-1)/ty,1);
  int sh_size = tx*2*ty*sizeof(float);	

  //main execution
  for(int t=0;t<iteration;t++)
    {
#ifdef PREF
      block2D_hybrid_coarsen_x<<<grid, block,sh_size, stream3>>>(c0,c1, h_A0, h_Anext, nx, ny,  nz);
#else
      block2D_hybrid_coarsen_x<<<grid, block,sh_size>>>(c0,c1, h_A0, h_Anext, nx, ny,  nz);
#endif
      float *d_temp = h_A0;
      h_A0 = h_Anext;
      h_Anext = d_temp;
      
    }
  CUERR // check and clear any existing errors
  
  cudaDeviceSynchronize();  
#define DEBUG
#ifdef DEBUG
  outputData("res.txt",h_A0,nx,ny,nz);
#endif

  cudaFree(h_A0);
  cudaFree(h_Anext);		

  return 0;

}
