#ifndef __XSBENCH__CUDA_PORT_HEADER_H__
#define __XSBENCH__CUDA_PORT_HEADER_H__

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include "XSbench_header.h"

#define CUDA_CALL(F)  if( (F) != cudaSuccess ) \
  {printf("Error %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()), __FILE__,__LINE__); exit(-1);} 

#define CUDA_CHECK()  if( (cudaPeekAtLastError()) != cudaSuccess )        \
  {printf("Error %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()), __FILE__,__LINE__-1); exit(-1);} 

#define linearize(x,y,w) x*w + y


#define __USE_LDG__ 0


#if __USE_LDG__
#define __LDG(x) __ldg(x)
#else
#define __LDG(x) (*( x ))
#endif



#define GRID_SEARCH 0
#define USE_GRIDPOINT_INDEX 1  // This is meaningless if GRID_SEARCH is 1
#define IDX_TYPE short

#define THREADS_PER_BLOCK 64
#define N_MAT 12

#define PROFILE_MODE 1




typedef struct{
	double energy;
  IDX_TYPE * xs_idx;  // contains the column index in an algorithm-defined row of NuclideGrids
} GridPoint_Index;




C_LINKAGE double cudaDriver(int numLookups, int n_nisotopes, int n_gridpoints, int numMats, 
                            int * numNucs, GridPoint * energyGrid, 
                            double ** concs, NuclideGridPoint ** nuclideGrid,  int ** mats,
                            double * cpuResults, int KernelId);
 



C_LINKAGE int checkResults(int numResults, int numelements, double * referenceResults, double * comparisonResults);
C_LINKAGE int checkGPUResults(int numResults, int numElements, double * referenceResults, double * devGPUResults);
void createDevDistCDF(double ** devDistCDF);

#ifdef __cplusplus



template<class T>
__inline__ __host__ __device__ int devBasicBinarySearch_index(T * A, double q, int n)
{
  int lowerLimit = 0;
  int upperLimit = n-1;
  int examinationPoint;
  int length = upperLimit - lowerLimit;

  while(length > 1)
  {
   //  (length > 1)  ==> ( length/2 >= 1 )  ==> (examinationPoint != lowerLimit )
    examinationPoint = lowerLimit + (length/2); 
    #if !defined(__CUDA_ARCH__) 
    
      double x = A[examinationPoint].energy ;
    #else
//      printf("(%d, %d, %d)\n", lowerLimit, examinationPoint, upperLimit);
      double x = __LDG(&A[examinationPoint].energy) ;
    #endif
    if(x > q)
    {
      upperLimit = examinationPoint;
    }
    else{
      lowerLimit = examinationPoint;
    }
    length = upperLimit - lowerLimit;
  }
  return lowerLimit;
}


template<class T>
__host__ __device__ NuclideGridPoint * devBasicBinarySearch_ptr(T * A, double q, int n)
{
  int lowerLimit = 0;
  int upperLimit = n-1;
  int examinationPoint;
  int length = upperLimit - lowerLimit;

  while(length > 1)
  {
   //  (length > 1)  ==> ( length/2 >= 1 )  ==> (examinationPoint != lowerLimit )
    examinationPoint = lowerLimit + (length/2); 
    double x = __LDG(&A[examinationPoint].energy);
    if(x > q)
    {
      upperLimit = examinationPoint;
    }
    else{
      lowerLimit = examinationPoint;
    }
    length = upperLimit - lowerLimit;
  }
  return &A[lowerLimit];
}

#endif

#endif
