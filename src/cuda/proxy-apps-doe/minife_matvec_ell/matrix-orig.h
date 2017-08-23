#pragma once

#include <algorithm>
#include <assert.h>

#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                  \
 if(e!=cudaSuccess) {                                               \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0);                                                         \
 }                                                                  \
}

typedef double ScalarType;
typedef int GlobalOrdinalType;


class Vector {
  public:
    GlobalOrdinalType num_rows;
    ScalarType *coefs, *d_coefs;

    void allocate(GlobalOrdinalType num_rows) {
      this->num_rows=num_rows;
      coefs=(ScalarType*)malloc(num_rows*sizeof(ScalarType));
      cudaMalloc(&d_coefs,num_rows*sizeof(ScalarType));
    }
    void deallocate() {
      free(coefs);
      cudaFree(d_coefs);
    }
    void copyToDevice() {
      cudaMemcpy(d_coefs,coefs,num_rows*sizeof(ScalarType),cudaMemcpyHostToDevice);
    }
    void copyToHost() {
      cudaMemcpy(coefs,d_coefs,num_rows*sizeof(ScalarType),cudaMemcpyDeviceToHost);
    }

    Vector& operator=(const Vector &v) {
      assert(num_rows==v.num_rows);
      for(int i=0;i<num_rows;i++)
        coefs[i]=v.coefs[i];
      return *this;
    }

    bool operator==(const Vector &v) {
      assert(num_rows==v.num_rows);
      for(int i=0;i<num_rows;i++) {
        if(fabs(coefs[i]-v.coefs[i])>.00001) {
          printf("i: %d, coefs[i]: %lg, v.coefs[i]: %lg\n", i, coefs[i], v.coefs[i]);
          return false;
        }
      }
      return true;
    }
    bool operator!=(const Vector &v) {
      assert(num_rows==v.num_rows);
      for(int i=0;i<num_rows;i++) {
        if(fabs(coefs[i]-v.coefs[i])>.00001) {
          printf("i: %d, coefs[i]: %lg, v.coefs[i]: %lg\n", i, coefs[i], v.coefs[i]);
          return true;
        }
      }
      return false;
    }
};

class CSRMatrix {
  public:
    typedef ScalarType ScalarType;
    typedef GlobalOrdinalType GlobalOrdinalType;

    GlobalOrdinalType num_rows, num_nnz;
    GlobalOrdinalType* row_offsets, *d_row_offsets;
    GlobalOrdinalType* cols, *d_cols;
    ScalarType* coefs, *d_coefs;

    void allocate(GlobalOrdinalType num_rows, GlobalOrdinalType num_nnz) {
      this->num_rows=num_rows;
      this->num_nnz=num_nnz;
      row_offsets=(GlobalOrdinalType*)malloc((num_rows+1)*sizeof(GlobalOrdinalType));
      cols=(GlobalOrdinalType*)malloc(num_nnz*sizeof(GlobalOrdinalType));
      coefs=(ScalarType*)malloc(num_nnz*sizeof(ScalarType));
      cudaMalloc(&d_row_offsets,(num_rows+1)*sizeof(GlobalOrdinalType));
      cudaMalloc(&d_cols,num_nnz*sizeof(GlobalOrdinalType));
      cudaMalloc(&d_coefs,num_nnz*sizeof(ScalarType));
    };
    
    void deallocate() {
      free(row_offsets);
      free(cols);
      free(coefs);
      cudaFree(d_row_offsets);
      cudaFree(d_cols);
      cudaFree(d_coefs);
    }

    void copyToDevice() {
      cudaMemcpy(d_row_offsets,row_offsets,(num_rows+1)*sizeof(GlobalOrdinalType),cudaMemcpyHostToDevice);
      cudaMemcpy(d_cols,cols,num_nnz*sizeof(GlobalOrdinalType),cudaMemcpyHostToDevice);
      cudaMemcpy(d_coefs,coefs,num_nnz*sizeof(ScalarType),cudaMemcpyHostToDevice);
    }
    void copyToHost() {
      cudaMemcpy(row_offsets,d_row_offsets,(num_rows+1)*sizeof(GlobalOrdinalType),cudaMemcpyDeviceToHost);
      cudaMemcpy(cols,d_cols,num_nnz*sizeof(GlobalOrdinalType),cudaMemcpyDeviceToHost);
      cudaMemcpy(coefs,d_coefs,num_nnz*sizeof(ScalarType),cudaMemcpyDeviceToHost);
    }
  
};

class ELLMatrix {
  public:
    typedef ScalarType ScalarType;
    typedef GlobalOrdinalType GlobalOrdinalType;

    GlobalOrdinalType num_rows, num_nnz_per_row, num_nnz;
    GlobalOrdinalType pitch;
    GlobalOrdinalType* cols, *d_cols;
    ScalarType* coefs, *d_coefs;

    void allocate(GlobalOrdinalType num_rows, GlobalOrdinalType num_nnz_per_row) {
      int align=128/std::max(sizeof(GlobalOrdinalType),sizeof(ScalarType));
      this->pitch=(num_rows+align-1)/align*align;
      this->num_rows=num_rows;
      this->num_nnz_per_row=num_nnz_per_row;
      this->num_nnz=num_nnz_per_row*pitch;

      cols=(GlobalOrdinalType*)malloc(num_nnz*sizeof(GlobalOrdinalType));
      coefs=(ScalarType*)malloc(num_nnz*sizeof(ScalarType));
      cudaMalloc(&d_cols,num_nnz*sizeof(GlobalOrdinalType));
      cudaMalloc(&d_coefs,num_nnz*sizeof(ScalarType));
    };
    
    void deallocate() {
      free(cols);
      free(coefs);
      cudaFree(d_cols);
      cudaFree(d_coefs);
    }

    void copyToDevice() {
      cudaMemcpy(d_cols,cols,num_nnz*sizeof(GlobalOrdinalType),cudaMemcpyHostToDevice);
      cudaMemcpy(d_coefs,coefs,num_nnz*sizeof(ScalarType),cudaMemcpyHostToDevice);
    }
    void copyToHost() {
      cudaMemcpy(cols,d_cols,num_nnz*sizeof(GlobalOrdinalType),cudaMemcpyDeviceToHost);
      cudaMemcpy(coefs,d_coefs,num_nnz*sizeof(ScalarType),cudaMemcpyDeviceToHost);
    }

    ELLMatrix& operator=(const CSRMatrix &matrix) {
      for(int row_idx=0;row_idx<matrix.num_rows;row_idx++)
      {
        int col_start=matrix.row_offsets[row_idx];
        int col_end=matrix.row_offsets[row_idx+1];
  
        int jj=0;
        for(int j=col_start;j<col_end;j++,jj++)
        {
          //printf("jj* %d, pitch:%d, row_idx: %d, num_nnz: %d\n", jj,pitch,row_idx,num_nnz);
          assert(jj*pitch+row_idx<num_nnz);
          coefs[jj*pitch+row_idx]=matrix.coefs[j];
          cols[jj*pitch+row_idx]=matrix.cols[j];
        }
        for(;jj<num_nnz_per_row;jj++) {
          cols[jj*pitch+row_idx]=-1;
        }
      }
      return *this;
    }
  
};

#include <fstream>
using  namespace std;

#include <fstream>
using  namespace std;

void read_matrix(CSRMatrix &matrix, char* filename) {

  ifstream fin;
  fin.open(filename);
  if(!fin) {
    printf("Error opening file: '%s'\n",filename);
    exit(1);
  } else {
    printf("reading file: '%s'\n",filename);
  }
  
  GlobalOrdinalType num_rows, num_nnz;

  fin >> num_rows >> num_nnz;

  matrix.allocate(num_rows,num_nnz);

  matrix.row_offsets[0]=0;
  GlobalOrdinalType r=0;

  for(GlobalOrdinalType i=0;i<num_nnz;i++) {
    GlobalOrdinalType row, col;
    ScalarType coef;
    fin >> row >> col >> coef;
    if(!fin) {
      printf("error reading nonzeros\n");
      exit(1);
    }
    matrix.coefs[i]=coef;
    matrix.cols[i]=col;
    while(r!=row) {
      matrix.row_offsets[++r]=i;
    }
  }
  matrix.row_offsets[++r]=num_nnz;

  fin.close();
}

