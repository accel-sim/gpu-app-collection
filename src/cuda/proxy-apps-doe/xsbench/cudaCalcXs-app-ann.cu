#include <cudaHeader.h>
#include <stdio.h>
#include <unistd.h>
#include <string>
#include <vector>
#include <algorithm>



__constant__ double dist[12] = {
  0.140,	// fuel
  0.052,	// cladding
  0.275,	// cold, borated water
  0.134,	// hot, borated water
  0.154,	// RPV
  0.064,	// Lower, radial reflector
  0.066,	// Upper reflector / top plate
  0.055,	// bottom plate
  0.008,	// bottom nozzle
  0.015,	// top nozzle
  0.025,	// top of fuel assemblies
  0.153 	// bottom of fuel assemblies
};	




#define NO_BINARY_SEARCH 0
#define RANDOM_CONC_NUC 0

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__device__ double devRn(unsigned long * seed)
{

#if STRIP_RANDOM == 1
	return 0.0;
#else
	/*
	 unsigned int m = 2147483647;
    unsigned int n1 = ( 16807u * (*seed) ) % m;
	(*seed) = n1;
	return (double) n1 / (double) m;
	*/
	double x = (double)seed * (double)31415.9262;
	x -= (double) (int) x;
	return x;
#endif
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__device__ int devPickMat( unsigned long * seed)
{
 double roll = devRn(seed);

  // makes a pick based on the distro
  double running = 0;
  for( int i = 0; i < 12; i++ )
  {
    running += dist[i];
    if( roll < running )
      return i;
  }
  return 11;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void devCalculateXS_integrated_single(int numLookups, int n_isotopes, int n_gridpoints,
                                               GridPoint_Index * energyGrid, int * numNucs, 
                                               int numMats, int maxNumNucs,  NuclideGridPoint * nuclideGrid,
                                               int * mats, double * concs, double * macro_xs_vector, 
                                               double * results, int nResults)
{
  unsigned long seed = 10000*threadIdx.x + 10* blockIdx.x + threadIdx.x + 31415;
  int matIdx; 
  int lin;
  double p_energy;
  double r[5];
  double f;
  int k = threadIdx.x + blockIdx.x * blockDim.x;

#if(STRIP_RANDOM==1)
  p_energy = 0.01 + (((double)(k%10))/10.0) + (((double)(k%1000))/1000.0);
  p_energy -= ((int)(p_energy));
  matIdx = k%12;
#else
  p_energy = devRn(&seed);
  matIdx = devPickMat(&seed);
#endif
  
  r[0] = r[1] = r[2] = r[3]= r[4] = 0.0;   
  int idx = devBasicBinarySearch_index<GridPoint_Index>(energyGrid, p_energy, n_isotopes*n_gridpoints);
  
  for(int i=0; i< numNucs[matIdx]; i++)
  {
    lin = linearize(matIdx, i, maxNumNucs);
    NuclideGridPoint * high;
    NuclideGridPoint * low;
    
    int nuc = mats[lin];
    int ptr = energyGrid[idx].xs_idx[nuc];
    low  = &nuclideGrid[linearize(nuc, ptr, n_gridpoints)];
    high = low + 1 ;
    double c = concs[lin];
    
    // calculate the re-useable interpolation factor
    f = (high->energy - p_energy) / (high->energy - low->energy);
    r[0] += c * (high->total_xs - f * (high->total_xs - low->total_xs));
    r[1] += c * (high->elastic_xs - f * (high->elastic_xs - low->elastic_xs));
    r[2] += c * (high->absorbtion_xs - f * (high->absorbtion_xs - low->absorbtion_xs));
    r[3] += c * (high->fission_xs - f * (high->fission_xs - low->fission_xs));
    r[4] += c * (high->nu_fission_xs - f * (high->nu_fission_xs - low->nu_fission_xs));
    
  } 
  // For some sense of legitimacy, write back the outputs, even though we will overwrite o
#pragma unroll 5
  for(int i=0; i<5; i++)
  {
    macro_xs_vector[i*blockDim.x + threadIdx.x] = r[i];
  }


#if(STRIP_RANDOM==1)
  if(k < nResults)
  {
    memcpy(&results[5*k], r, N_ELEMENTS*sizeof(double));
  }
#endif

}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//  devCalculateXS_d3 is the 'best performing' kernel on the large case. This kernel unwraps the nuclide lookup for two nuclides simultaneously with the 
//    edge case handled inline (instead of having a completely separate code path for the final iteration as in the d2 version of this kernel).
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void devCalculateXS_d4(int numLookups, int n_isotopes, int n_gridpoints,
                                  GridPoint_Index * energyGrid, int * numNucs, 
                                  int numMats, int maxNumNucs,  NuclideGridPoint * nuclideGrid,
                                  int * mats, double * concs, double * macro_xs_vector, double * results, int nResults)
{
  int matIdx;
  double p_energy;
  int k = threadIdx.x + blockIdx.x * blockDim.x;
  if(k > numLookups)
  {
    return;
  }
#if(STRIP_RANDOM==1)
  p_energy = 0.01 + (((double)(k%10))/10.0) + (((double)(k%1000))/1000.0);
  p_energy -= ((int)(p_energy));
  matIdx = k%12;
#else
  unsigned long seed = 10000*threadIdx.x + threadIdx.x + 10* blockIdx.x + 31415;
  p_energy = devRn(&seed);
  matIdx = devPickMat(&seed);
#endif
    double r[5];
    r[0] = r[1] = r[2] = r[3]= r[4] = 0.0;

    #if NO_BINARY_SEARCH
      int idx = k % (n_isotopes * n_gridpoints);
      idx += (((unsigned long)k * (unsigned long)k) % (unsigned long) (n_isotopes * n_gridpoints));
      idx = ((unsigned long)idx * (unsigned long) blockIdx.x * (unsigned long) threadIdx.x) % ((unsigned long) (n_isotopes * n_gridpoints));
    #else
      int idx = devBasicBinarySearch_index<GridPoint_Index>(energyGrid, p_energy, n_isotopes*n_gridpoints);
    #endif
    for(int i=0; i< numNucs[matIdx]; i+=2)
    {
      double2 t[12];   
      int nuc    = __LDG(&mats[linearize(matIdx, i, maxNumNucs)]);
      double c   = __LDG(&concs[linearize(matIdx, i, maxNumNucs)]);      
      IDX_TYPE ptr = __LDG(&energyGrid[idx].xs_idx[nuc]);
      NuclideGridPoint * low = &nuclideGrid[linearize(nuc, ptr, n_gridpoints)];
      double2 * base = (double2*) low;
   
      int nucB;
      IDX_TYPE ptrB;
      double2 * baseB;
      double cB;

      if( i < numNucs[matIdx]-1 )     
      {
        nucB  = __LDG(&mats[linearize(matIdx, i+1, maxNumNucs)]);
        cB    = __LDG(&concs[linearize(matIdx, i+1, maxNumNucs)]);      
        ptrB  = __LDG(&energyGrid[idx].xs_idx[nucB]);
        baseB = (double2*)&nuclideGrid[linearize(nucB, ptrB, n_gridpoints)];
      }
      #pragma unroll 6
      for(int s=0; s<6; s++)
      {
        t[s] = __LDG(&base[s]);
      }

      if( i < numNucs[matIdx]-1 )
      {
        #pragma unroll 6
        for(int s=0; s<6; s++)
        {
          t[s+6] = __LDG(&baseB[s]);          
        }
      }

      {
        double f = (t[3].x - p_energy) / (t[3].x - t[0].x);
        double fB;
        r[0] += c*(t[3].y - f* (t[3].y - t[0].y));
        if( i < numNucs[matIdx]-1 )
        {
          fB = (t[9].x - p_energy) / (t[9].x - t[6].x);
          r[0] += cB*(t[9].y - fB* (t[9].y - t[6].y));
        }
        
        #pragma unroll
        for(int s=0; s<2; s++)
        {
          r[2*s+1] += c*(t[s+4].x - f* (t[s+4].x - t[s+1].x));
          r[2*s+2] += c*(t[s+4].y - f* (t[s+4].y - t[s+1].y));
          if( i < numNucs[matIdx]-1 )
          {
            r[2*s+1] += cB*(t[s+10].x - fB* (t[s+10].x - t[s+7].x));
            r[2*s+2] += cB*(t[s+10].y - fB* (t[s+10].y - t[s+7].y));
          }
        }
      }
    }
  
    // For some sense of legitimacy, write back the outputs, even though we will overwrite o
    #pragma unroll 
    for(int i=0; i<5; i++)
    {
      macro_xs_vector[i* blockDim.x + threadIdx.x] = r[i];
    }

#if(STRIP_RANDOM==1)
    if(k < nResults)
    {
      results[5*k] = r[0];
      results[5*k+1] = r[1];
      results[5*k+2] = r[2];
      results[5*k+3] = r[3];
      results[5*k+4] = r[4];
      memcpy(&results[5*k], r, N_ELEMENTS*sizeof(double));
    }
#endif

}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//  devCalculateXS_d3 is the 'best performing' kernel on the large case. This kernel unwraps the nuclide lookup for two nuclides simultaneously with the 
//    edge case handled inline (instead of having a completely separate code path for the final iteration as in the d2 version of this kernel).
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void devCalculateXS_d3(int numLookups, int n_isotopes, int n_gridpoints,
                                  GridPoint_Index * energyGrid, int * numNucs, 
                                  int numMats, int maxNumNucs,  NuclideGridPoint * nuclideGrid,
                                  int * mats, double * concs, double * macro_xs_vector, double * results, int nResults)
{

  unsigned long seed = 10000*threadIdx.x + threadIdx.x + 10* blockIdx.x + 31415;
  int matIdx;
  double p_energy;
  int workPerBlock = 1 + ( numLookups / (gridDim.x) );
  int lowerLimit = blockIdx.x * workPerBlock + threadIdx.x;
  int upperLimit = lowerLimit + workPerBlock;

  for(int k=lowerLimit; k< upperLimit; k+= blockDim.x)
  {
#if(STRIP_RANDOM==1)
    p_energy = 0.01 + (((double)(k%10))/10.0) + (((double)(k%1000))/1000.0);
    p_energy -= ((int)(p_energy));
    matIdx = k%12;
#else
    
    p_energy = devRn(&seed);
    matIdx = devPickMat(&seed);
#endif
    double r[5];
    r[0] = r[1] = r[2] = r[3]= r[4] = 0.0;

    int idx = devBasicBinarySearch_index<GridPoint_Index>(energyGrid, p_energy, n_isotopes*n_gridpoints);

    for(int i=0; i< numNucs[matIdx]; i+=2)
    {
      double2 t[12];   
      int nuc    = __LDG(&mats[linearize(matIdx, i, maxNumNucs)]);
      double c   = __LDG(&concs[linearize(matIdx, i, maxNumNucs)]);      
      IDX_TYPE ptr = __LDG(&energyGrid[idx].xs_idx[nuc]);
      NuclideGridPoint * low = &nuclideGrid[linearize(nuc, ptr, n_gridpoints)];
      double2 * base = (double2*) low;
   
      int nucB;
      IDX_TYPE ptrB;
      double2 * baseB;
      double cB;

      if( i < numNucs[matIdx]-1 )     
      {
        nucB  = __LDG(&mats[linearize(matIdx, i+1, maxNumNucs)]);
        cB    = __LDG(&concs[linearize(matIdx, i+1, maxNumNucs)]);      
        ptrB  = __LDG(&energyGrid[idx].xs_idx[nucB]);
        baseB = (double2*)&nuclideGrid[linearize(nucB, ptrB, n_gridpoints)];
      }
      #pragma unroll 6
      for(int s=0; s<6; s++)
      {
        t[s] = __LDG(&base[s]);
      }
      if( i < numNucs[matIdx]-1 )
      {
        #pragma unroll 6
        for(int s=0; s<6; s++)
        {
          t[s+6] = __LDG(&baseB[s]);          
        }
      }

      {
        double f = (t[3].x - p_energy) / (t[3].x - t[0].x);
        double fB;
        r[0] += c*(t[3].y - f* (t[3].y - t[0].y));
        if( i < numNucs[matIdx]-1 )
        {
          fB = (t[9].x - p_energy) / (t[9].x - t[6].x);
          r[0] += cB*(t[9].y - fB* (t[9].y - t[6].y));
        }
        
        #pragma unroll
        for(int s=0; s<2; s++)
        {
          r[2*s+1] += c*(t[s+4].x - f* (t[s+4].x - t[s+1].x));
          r[2*s+2] += c*(t[s+4].y - f* (t[s+4].y - t[s+1].y));
          if( i < numNucs[matIdx]-1 )
          {
            r[2*s+1] += cB*(t[s+10].x - fB* (t[s+10].x - t[s+7].x));
            r[2*s+2] += cB*(t[s+10].y - fB* (t[s+10].y - t[s+7].y));
          }
        }
      }
    }
  
    // For some sense of legitimacy, write back the outputs, even though we will overwrite o
    #pragma unroll 
    for(int i=0; i<5; i++)
    {
      macro_xs_vector[i* blockDim.x + threadIdx.x] = (double) idx;// r[i];
    }

#if(STRIP_RANDOM==1)
    if(k < nResults)
    {
      results[5*k] = r[0];
      results[5*k+1] = r[1];
      results[5*k+2] = r[2];
      results[5*k+3] = r[3];
      results[5*k+4] = r[4];
      memcpy(&results[5*k], r, N_ELEMENTS*sizeof(double));
    }
#endif
  }
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void devCalculateXS_opt_single(int numLookups, int n_isotopes, int n_gridpoints,
                                   GridPoint_Index * energyGrid, int * numNucs, 
                                   int numMats, int maxNumNucs,  NuclideGridPoint * nuclideGrid,
                                   int * mats, double * concs, double * macro_xs_vector, 
                                   double * results, int nResults)
{


  int id = threadIdx.x + blockDim.x * blockIdx.x;
  int matIdx; 
  double p_energy;
  double r0, r1, r2, r3, r4;   

#if(STRIP_RANDOM==1)
  int k = threadIdx.x + blockDim.x * blockIdx.x;
  p_energy = 0.01 + (((double)(k%10))/10.0) + (((double)(k%1000))/1000.0);
  p_energy -= ((int)(p_energy));
  matIdx = k%12;
#else
  unsigned long seed = 10000*threadIdx.x + 10* blockIdx.x + threadIdx.x + 31415;
  p_energy = devRn(&seed);
  matIdx = devPickMat(&seed);
#endif

  r0 = r1 = r2 = r3 = r4 = 0.0;
  int idx = devBasicBinarySearch_index<GridPoint_Index>(energyGrid, p_energy, n_isotopes*n_gridpoints);
  for(int i=0; i< numNucs[matIdx]; i++)
  {
    double2 t0, t1, t2, t3, t4, t5;
    int nuc  = __LDG(&mats[linearize(matIdx, i, maxNumNucs)]);
    double c = __LDG(&concs[linearize(matIdx, i, maxNumNucs)]);      
    IDX_TYPE ptr  = __LDG(&energyGrid[idx].xs_idx[nuc]);
    double2 * base = (double2*) &nuclideGrid[linearize(nuc, ptr, n_gridpoints)];
    
    t0 = __LDG(base);
    t1 = __LDG(&base[1]);
    t2 = __LDG(&base[2]);
    t3 = __LDG(&base[3]);
    t4 = __LDG(&base[4]);
    t5 = __LDG(&base[5]);
    
    double f = (t3.x - p_energy) / (t3.x - t0.x);
    r0 += c*(t3.y - f * (t3.y - t0.y));
    r1 += c*(t4.x - f * (t4.x - t1.x));
    r2 += c*(t4.y - f * (t4.y - t1.y));
    r3 += c*(t5.x - f * (t5.x - t2.x));
    r4 += c*(t5.y - f * (t5.y - t2.y));
  }  
  // For some sense of legitimacy, write back the outputs, even though we will overwrite o
  
  macro_xs_vector[threadIdx.x] = r0;
  macro_xs_vector[blockDim.x + threadIdx.x] = r1;
  macro_xs_vector[2*blockDim.x + threadIdx.x] = r2;
  macro_xs_vector[3*blockDim.x + threadIdx.x] = r3;
  macro_xs_vector[4*blockDim.x + threadIdx.x] = r4;
  

#if(STRIP_RANDOM==1)
  if(k < nResults)
  {
    results[5*k] = r0;
    results[5*k+1] = r1;
    results[5*k+2] = r2;
    results[5*k+3] = r3;
    results[5*k+4] = r4;
  }

#endif

}



///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
C_LINKAGE int checkResults(int numResults, int numElements, double * referenceResults, double * comparisonResults)
{
  bool correct = true;
  for(int i=0; i<numResults*numElements; i++)
  {
    if(fabs(referenceResults[i] - comparisonResults[i]) > 0.000001)
    {
      printf("Answers do not match at lookup %d element %d (%.10f, %.10f) \n", i/numElements, i%numElements, referenceResults[i], comparisonResults[i]);
      correct = false;
      break;
    }
  }
  if(correct)
  {
    return 1;
  }
  else
  {
    return 0;
  }
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
C_LINKAGE int checkGPUResults(int numResults, int numElements, double * referenceResults, double * devGPUResults)
{
  double * lResults;
  bool correct;
  CUDA_CALL(cudaMallocHost(&lResults, numResults * numElements*sizeof(double)));
  CUDA_CALL(cudaMemcpy(lResults, devGPUResults, numResults * numElements *sizeof(double), cudaMemcpyDeviceToHost));
  correct = checkResults(numResults, numElements, referenceResults, lResults);
  CUDA_CALL(cudaFreeHost(lResults));
  if(correct)
  {
    return 1;
  }
  else
  {
    return 0;
  }
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double runExperiment( void (*kernel)(int, int, int, GridPoint_Index*, int*, int, int, NuclideGridPoint *, int *, double*, double *,  double*, int),
                      std::string kernelName,
                      int numLookups, int n_isotopes, int n_gridpoints, GridPoint_Index * energyGrid, int * numNucs,  int numMats, int maxNumNucs,  
                      NuclideGridPoint * nuclideGrid, int * mats, double * concs, double * macro_xs_vector, double * searchGrid, double * results, int nResults,
                      double * cpuResults, int blockLimiter, int lookupsPerThread=1, int dshared=0)
{
  
  int numBlocks;
  int threadsPerBlock;
  double performance = -1.0E12;
  cudaEvent_t start, stop;
  //CUDA_CALL(cudaEventCreate(&start));
  //CUDA_CALL(cudaEventCreate(&stop));
  double maxPerf = -1.0;

#if(PROFILE_MODE)
    threadsPerBlock = THREADS_PER_BLOCK;
  
    CUDA_CALL(cudaMemset(results, 0, 5*NUM_RESULTS*sizeof(double)));
    //CUDA_CALL(cudaEventRecord(start));
    
    numBlocks = 1 + numLookups/(threadsPerBlock*lookupsPerThread);
   
    kernel<<<numBlocks, threadsPerBlock, dshared>>>(numLookups, n_isotopes,  n_gridpoints, energyGrid, numNucs,
                                             numMats, maxNumNucs, nuclideGrid, mats, concs, macro_xs_vector,
                                             results, nResults);
    //CUDA_CHECK();
    //CUDA_CALL(cudaEventRecord(stop));
    //CUDA_CALL(cudaEventSynchronize(stop));
    CUDA_CALL(cudaDeviceSynchronize());
    float gpuTime = 0.0;
    //CUDA_CALL(cudaEventElapsedTime(&gpuTime, start, stop));
	gpuTime = 1.0;
    performance = 1000.0 * ( (double) numLookups/ (double) gpuTime); 
    printf("%s <<<%d, %d>>> Lookups/s: %.0lf in %f ms\n", kernelName.c_str(), numBlocks, threadsPerBlock, performance, gpuTime);
    maxPerf = performance;
    
#else
  for(threadsPerBlock=32; threadsPerBlock<=blockLimiter; threadsPerBlock+=32)
  {

    CUDA_CALL(cudaMemset(results, 0, 5*NUM_RESULTS*sizeof(double)));
    //CUDA_CALL(cudaEventRecord(start));
    
    numBlocks = 1 + numLookups/(threadsPerBlock*lookupsPerThread);


    kernel<<<numBlocks, threadsPerBlock, dshared>>>(numLookups, n_isotopes,  n_gridpoints, energyGrid, numNucs,
                                           numMats, maxNumNucs, nuclideGrid, mats, concs, macro_xs_vector,
                                           results, nResults);
    
    //CUDA_CHECK();
    //CUDA_CALL(cudaEventRecord(stop));
    //CUDA_CALL(cudaEventSynchronize(stop));
    CUDA_CALL(cudaDeviceSynchronize());

    float gpuTime = 0.0;
    //CUDA_CALL(cudaEventElapsedTime(&gpuTime, start, stop));
    gpuTime = 1.0;
    performance = 1000.0 * ( (double) numLookups/ (double) gpuTime); 
    printf("%s <<<%d, %d>>> Lookups/s: %.0lf in %f ms\n", kernelName.c_str(), numBlocks, threadsPerBlock, performance, gpuTime);
    
    
#if STRIP_RANDOM == 1
    if(checkGPUResults(nResults, N_ELEMENTS, cpuResults, results))
    {
      printf("%s results match reference CPU Results\n", kernelName.c_str());
    }
    else{
      printf("%s Results are INCORRECT\n", kernelName.c_str());
      performance = -1.0;
    }
#endif
    if(performance > maxPerf)
    {
      maxPerf = performance;
    }
  }
#endif
  return maxPerf;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
C_LINKAGE double cudaDriver(int numLookups,  int n_isotopes, int n_gridpoints, int numMats, 
                            int * numNucs, GridPoint * energyGrid, 
                            double ** concs, NuclideGridPoint ** nuclideGrid,  int ** mats,
                            double * cpuResults, int kernelId)

{

  printf("################################################################################\n");
  printf("                           GPU SIMULATION\n");
	printf("################################################################################\n");


  int * devNumNucs = NULL;
  double * devConcs = NULL;
  int * devMats = NULL;
  GridPoint_Index * devEnergyGrid = NULL;
  NuclideGridPoint * devNuclideGrid = NULL;
  double * devEnergySearchGrid = NULL;
  double * devMacroXSVector = NULL;
  double * devResults = NULL;
  IDX_TYPE * devIndexArray;
  cudaEvent_t start, stop;
  //CUDA_CALL(cudaEventCreate(&start));
  //CUDA_CALL(cudaEventCreate(&stop));


  int threadsPerBlock = THREADS_PER_BLOCK;
  long numBlocks = 1024;

  int maxNumNucs=-1;
  for(int i=0; i<numMats; i++)
    { maxNumNucs = max(maxNumNucs, numNucs[i]); }

  printf("MaxNumNucs is %d\n", maxNumNucs);
  printf(" numIsotopes: %d, n_gridpoints: %d, n_mats: %d\n", n_isotopes, n_gridpoints, numMats);

// Hoisted for doing annotaion of hot structure
    CUDA_CALL(cudaMalloc(&devIndexArray, n_isotopes * n_isotopes * n_gridpoints * sizeof(IDX_TYPE)));
  CUDA_CALL(cudaMalloc(&devMacroXSVector, 5 * numBlocks * threadsPerBlock * sizeof(double)));

  CUDA_CALL(cudaMalloc(&devNumNucs, numMats * sizeof(int)));
  CUDA_CALL(cudaMemcpy(devNumNucs, numNucs, numMats * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMalloc(&devNuclideGrid, n_isotopes * n_gridpoints * sizeof(NuclideGridPoint)));
  CUDA_CALL(cudaMemcpy(devNuclideGrid, nuclideGrid[0], n_isotopes * n_gridpoints * sizeof(NuclideGridPoint), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMalloc(&devResults, 5*NUM_RESULTS*sizeof(double)));

  // Deal with concs and Mats. Notice, when copying these to the GPU we flatten out the structure into a dense matrix
  {
    long totalElements = numMats * maxNumNucs;
    CUDA_CALL(cudaMalloc(&devMats, totalElements * sizeof(int)));
    CUDA_CALL(cudaMalloc(&devConcs, totalElements * sizeof(double)));
    CUDA_CALL(cudaMemset(devMats, 0, totalElements * sizeof(int)));
    CUDA_CALL(cudaMemset(devConcs, 0, totalElements * sizeof(double)));
    for(int i=0; i<numMats; i++)
    {
      CUDA_CALL(cudaMemcpy(&devMats[i*maxNumNucs], mats[i], numNucs[i]*sizeof(int), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(&devConcs[i*maxNumNucs], concs[i], numNucs[i]*sizeof(double), cudaMemcpyHostToDevice));
    }
  }
  // Deal with Energy Grid. 
  {
    GridPoint_Index * lEnergyGrid;

    IDX_TYPE * lIndexArray;
    IDX_TYPE * rowptr;
    

    CUDA_CALL(cudaMalloc(&devEnergyGrid, n_isotopes * n_gridpoints * sizeof(GridPoint_Index)));
    CUDA_CALL(cudaMallocHost(&lEnergyGrid, n_isotopes * n_gridpoints * sizeof(GridPoint_Index)));
    printf("Allocation is %f mb\n",  ((long)n_isotopes * (long)n_isotopes * (long)n_gridpoints * sizeof(IDX_TYPE))/(double) 1E6);

    CUDA_CALL(cudaMallocHost(&lIndexArray, n_gridpoints  *n_isotopes* n_isotopes * sizeof(IDX_TYPE)));
    memset(lIndexArray, 0, n_gridpoints * n_isotopes * n_isotopes * sizeof(IDX_TYPE));

    for(int i=0; i<n_isotopes * n_gridpoints; i++)
    {
      lEnergyGrid[i].energy = energyGrid[i].energy;
      lEnergyGrid[i].xs_idx = &devIndexArray[i*n_isotopes];
      rowptr = &lIndexArray[i*n_isotopes];
      switch(sizeof(IDX_TYPE))
      {
      case 4:
        memcpy(rowptr, energyGrid[i].xs_ptrs, n_isotopes*sizeof(int));
        break;
      case 2:
        for(int s=0; s<n_isotopes; s++)
        {
          rowptr[s] = (IDX_TYPE)energyGrid[i].xs_ptrs[s];
        }
        break;
      default:
        printf("Error: sizeof(IDX_TYPE) is not supported\n");
        break;
      }
    }
    CUDA_CALL(cudaMemcpy(devIndexArray, lIndexArray, n_isotopes*n_isotopes*n_gridpoints*sizeof(IDX_TYPE), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(devEnergyGrid, lEnergyGrid, n_isotopes * n_gridpoints * sizeof(GridPoint_Index), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaFreeHost(lIndexArray));
    CUDA_CALL(cudaFreeHost(lEnergyGrid));
}
  // Run Experiments
  std::vector<double> perf;
  int dshared  = 0;

  switch(kernelId)
  {
  case 0:
    perf.push_back(runExperiment(devCalculateXS_d4, "LDG OPT Double Revised singular", numLookups, n_isotopes,  n_gridpoints, devEnergyGrid, devNumNucs, 
                                 numMats, maxNumNucs, devNuclideGrid, devMats, devConcs, devMacroXSVector, devEnergySearchGrid,
                                 devResults, NUM_RESULTS, cpuResults, 128, 1, dshared));  
    break;
  case 1:
    perf.push_back(runExperiment(devCalculateXS_opt_single, "LDG OPT singler", numLookups, n_isotopes,  n_gridpoints, devEnergyGrid, devNumNucs, 
                                 numMats, maxNumNucs, devNuclideGrid, devMats, devConcs, devMacroXSVector, devEnergySearchGrid,
                                 devResults, NUM_RESULTS, cpuResults, 128, 1, dshared));  
    break;
  case 2:
    perf.push_back(runExperiment(devCalculateXS_integrated_single, "Basic Kernel singular", numLookups, n_isotopes,  n_gridpoints, devEnergyGrid, devNumNucs, 
                                 numMats, maxNumNucs, devNuclideGrid, devMats, devConcs, devMacroXSVector, devEnergySearchGrid,
                                 devResults, NUM_RESULTS, cpuResults, 128, 1, dshared));  
    break;
  case 3:
    perf.push_back(runExperiment(devCalculateXS_d3, "LDG OPT Double (original) singular", numLookups, n_isotopes,  n_gridpoints, devEnergyGrid, devNumNucs, 
                                 numMats, maxNumNucs, devNuclideGrid, devMats, devConcs, devMacroXSVector, devEnergySearchGrid,
                                 devResults, NUM_RESULTS, cpuResults, 128, 1, dshared));
  default:
    printf("Error: unrecognized kernel id\n");
    break;
  }


  CUDA_CALL(cudaFree(devIndexArray));
  CUDA_CALL(cudaFree(devNumNucs));
  CUDA_CALL(cudaFree(devConcs));
  CUDA_CALL(cudaFree(devMats));

  if(devEnergyGrid != NULL)
    { CUDA_CALL(cudaFree(devEnergyGrid)); }

  CUDA_CALL(cudaFree(devNuclideGrid));
  CUDA_CALL(cudaFree(devMacroXSVector));

  std::vector<double>::iterator r = std::max_element(perf.begin(), perf.end());
  printf("Max Perf is %f\n", (*r));
  return (*r);

}
