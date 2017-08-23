#include <cstdio>
#include "matrix.h"
#include "kernels.h"
using namespace std;

//#define validate

int main(int argc, char** argv) {
  printf("Usage: ./matvec inputbase alg\n");
  printf("    alg list:\n");
  printf("              0:  ell\n");
  printf("              1:  ell cv\n");
  printf("              2:  csr naive\n");
  printf("              3:  csr vector(2)\n");
  printf("              4:  csr vector(4)\n");
  printf("              5:  csr vector(8)\n");
  printf("              6:  csr vector(16)\n");
  printf("              7:  csr vector(32)\n");

  CSRMatrix csr_matrix;
  ELLMatrix ell_matrix;
  Vector y_gold, y, x;
  int alg=0;
  
  if(argc!=3) {
    printf("invalid arguements, please supply basename\n"); 
    exit(1); 
  }

  alg=atoi(argv[2]);
  
  read_matrix(csr_matrix,argv[1]);
  ell_matrix.allocate(csr_matrix.num_rows,27);
  x.allocate(csr_matrix.num_rows);
  y_gold.allocate(x.num_rows);
  y.allocate(x.num_rows);
  
  ell_matrix=csr_matrix;
  
  for(int i=0;i<x.num_rows;i++) {
    x.coefs[i]=drand48();
  }


  csr_matrix.copyToDevice();
  ell_matrix.copyToDevice();
  x.copyToDevice();

  switch(alg) {
    case 0:
      matvec_ell(ell_matrix,x,y);
      break;
    case 1:
      matvec_ell_cv(ell_matrix,x,y);
      break;
    case 2:
      matvec_csr(csr_matrix,x,y); 
      break;
    case 3:
      matvec_csr_vector<2>(csr_matrix,x,y);
      break;
    case 4:
      matvec_csr_vector<4>(csr_matrix,x,y);
      break;
    case 5:
      matvec_csr_vector<8>(csr_matrix,x,y);
      break;
    case 6:
      matvec_csr_vector<16>(csr_matrix,x,y);
      break;
    case 7:
      matvec_csr_vector<32>(csr_matrix,x,y);
      break;
  };

#ifdef validate
  matvec_csr(csr_matrix,x,y_gold); 
  y.copyToHost();
  y_gold.copyToHost();
  assert(y==y_gold);   	
#endif
  
  csr_matrix.deallocate();
  ell_matrix.deallocate();
  y_gold.deallocate();
  y.deallocate();
  x.deallocate();

  cudaDeviceSynchronize();
  cudaCheckError();
  cudaDeviceReset();
  printf("End of main\n");
}
