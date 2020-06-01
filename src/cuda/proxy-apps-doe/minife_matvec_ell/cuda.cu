#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <assert.h>
using namespace std;
#define ScalarType double
#define LocalOrdinalType int
#define GlobalOrdinalType int

#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0); \
 }                                                                 \
}


union dbl2flt2 {
  double d;
  float2 f;
};
__device__ inline double __shfl(double A, int lane, int size=32) {
  dbl2flt2 c;
  c.d=A;
  c.f.x=__shfl(c.f.x,lane,size);
  c.f.y=__shfl(c.f.y,lane,size);
  return c.d;
}

__global__ void matvec_kernel(int n, ScalarType beta, LocalOrdinalType *Arowoffsets, GlobalOrdinalType *Acols, ScalarType *Acoefs, ScalarType *xcoefs, ScalarType *ycoefs) {

  for(int row=blockIdx.x*blockDim.y+threadIdx.y;row<n;row+=gridDim.x*blockDim.y) {
  //  if(row==0 && threadIdx.x==0)
  //    printf("row: %d, blockIdx.x: %d, blockIdx.y: %d, threadIdx.x: %d, threadIdx.y :%d\n", row, blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.x);
    ScalarType sum = 0;

    int jStart=__ldg(Arowoffsets+row);
    int jEnd=__ldg(Arowoffsets+row+1);
    for(LocalOrdinalType j=jStart+threadIdx.x; j<jEnd; j+=blockDim.x) {
        GlobalOrdinalType Acol=Acols[j];
        ScalarType Acoef=Acoefs[j];
        ScalarType Xcoef=__ldg(xcoefs+Acol);
        sum += Acoef*Xcoef;
    }
    sum+=__shfl(sum,threadIdx.x+16);
    sum+=__shfl(sum,threadIdx.x+8);
    sum+=__shfl(sum,threadIdx.x+4);
    sum+=__shfl(sum,threadIdx.x+2);
    sum+=__shfl(sum,threadIdx.x+1);
    if(threadIdx.x==0)
      ycoefs[row] = beta*ycoefs[row] + sum;
  }
}

void matvec(int n, int nnz, ScalarType beta, LocalOrdinalType *Arowoffsets, GlobalOrdinalType *Acols, ScalarType *Acoefs, ScalarType *xcoefs, ScalarType *ycoefs) {

  dim3 threads(32,4);
  dim3 blocks(ceil(n/(double)threads.y));
  matvec_kernel<<<blocks,threads>>>(n,beta,Arowoffsets,Acols,Acoefs,xcoefs,ycoefs);
 
}

int main() {
  ifstream fin("A.mtx");
  LocalOrdinalType rows;
  LocalOrdinalType nnz;
  printf("Reading input matrix\n");
  fin >> rows >> nnz;
  LocalOrdinalType *Arowoffsets=new LocalOrdinalType[rows+1];
  GlobalOrdinalType *Acols=new GlobalOrdinalType[nnz];
  ScalarType *Acoefs=new ScalarType[nnz];
  ScalarType *xcoefs=new ScalarType[rows];
  ScalarType *ycoefs=new ScalarType[rows];
  
  for(int i=0;i<rows;i++)
    xcoefs[i]=1;

  for(int i=0;i<rows+1;i++)
    Arowoffsets[i]=0;
  int j=0;
  while(!fin.eof()) {
    GlobalOrdinalType col;
    LocalOrdinalType row;
    ScalarType coef;
    fin >> row >> col >> coef;
    if(fin.eof())
      break;
    assert(j<nnz);
    Acols[j]=col;
    Acoefs[j]=coef;
    Arowoffsets[row]++;
    j++;
  }
  for(int i=0;i<rows;i++)
    Arowoffsets[i+1]+=Arowoffsets[i];

  LocalOrdinalType *d_Arowoffsets;
  GlobalOrdinalType *d_Acols;
  ScalarType *d_Acoefs;
  ScalarType *d_xcoefs;
  ScalarType *d_ycoefs;

  cudaMalloc(&d_Arowoffsets,sizeof(LocalOrdinalType)*(rows+1));
  cudaMalloc(&d_Acols,sizeof(GlobalOrdinalType)*(nnz));
  cudaMalloc(&d_Acoefs,sizeof(ScalarType)*nnz);
  cudaMalloc(&d_xcoefs,sizeof(ScalarType)*rows);
  cudaMalloc(&d_ycoefs,sizeof(ScalarType)*rows);
  cudaMemcpy(d_Arowoffsets,Arowoffsets,sizeof(LocalOrdinalType)*(rows+1),cudaMemcpyHostToDevice);
  cudaMemcpy(d_Acols,Acols,sizeof(GlobalOrdinalType)*(nnz),cudaMemcpyHostToDevice);
  cudaMemcpy(d_Acoefs,Acoefs,sizeof(ScalarType)*nnz,cudaMemcpyHostToDevice);
  cudaMemcpy(d_xcoefs,xcoefs,sizeof(ScalarType)*rows,cudaMemcpyHostToDevice);
  cudaMemcpy(d_ycoefs,ycoefs,sizeof(ScalarType)*rows,cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  cudaCheckError();

  {
    printf("matvec\n");
    for(int i=0;i<1;i++)
      matvec(rows,nnz,0,d_Arowoffsets,d_Acols,d_Acoefs,d_xcoefs,d_ycoefs);
  }
  
  cudaMemcpy(ycoefs,d_ycoefs,sizeof(ScalarType)*rows,cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  //for(int i=0;i<rows;i++)
  //  printf("row: %d, y: %lg\n", i, ycoefs[i]);

  cudaCheckError();
  cudaFree(d_Arowoffsets);
  cudaFree(d_Acols);
  cudaFree(d_Acoefs);
  cudaFree(d_xcoefs);
  cudaFree(d_ycoefs);
  cudaCheckError();
  printf("done\n");
}
