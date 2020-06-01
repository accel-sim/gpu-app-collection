#pragma once

#include "cutil_subset.h"

struct CSRGraph {
  int nnodes;
  int nedges;

  int *row_offsets;
  int *columns;

  bool *sat;

  float *bias;
  bool *value;
  
  bool alloc()
  {
    assert(nnodes > 0);
    assert(nedges > 0);
    
    row_offsets = (int *) calloc(nnodes + 1, sizeof(*row_offsets));
    columns = (int *) calloc(nedges, sizeof(*columns));

    sat = (bool *) calloc(nnodes, sizeof(bool));
    bias = (float *) calloc(nnodes, sizeof(float));
    value = (bool *) calloc(nnodes, sizeof(bool));

    return (row_offsets != NULL) && (columns != NULL) && sat && bias && value;
  }
  
  void set_last_offset()
  {
    row_offsets[nnodes] = nedges;
  }

  int degree(const int node) const
  {
    return row_offsets[node + 1] - row_offsets[node];
  }

  void dump_edges() const
  {
    int i;
    for(i = 0; i < nedges; i++)
      printf("%d ", columns[i]);
    printf("\n");
  }

  void dump_offsets() const
  {
    int i;
    for(i = 0; i < nnodes; i++)
      printf("%d ", row_offsets[i]);
    printf("\n");
  }
};

struct GPUCSRGraph : CSRGraph {
  bool alloc()
  {
    assert(nnodes > 0);
    assert(nedges > 0);
    
    CUDA_SAFE_CALL(cudaMalloc(&row_offsets, (nnodes + 1) * sizeof(*row_offsets)));
    CUDA_SAFE_CALL(cudaMalloc(&columns, nedges * sizeof(*columns)));

    CUDA_SAFE_CALL(cudaMalloc(&sat, nnodes * sizeof(*sat)));

    CUDA_SAFE_CALL(cudaMalloc(&bias, nnodes * sizeof(*bias)));

    CUDA_SAFE_CALL(cudaMalloc(&value, nnodes * sizeof(*value)));

    return (row_offsets != NULL) && (columns != NULL) && sat && bias && value;
  }

  bool from_cpu(CSRGraph &cpu)
  {
    nnodes = cpu.nnodes;
    nedges = cpu.nedges;
    
    assert(alloc());
    
    CUDA_SAFE_CALL(cudaMemcpy(row_offsets, cpu.row_offsets, (nnodes + 1) * sizeof(*row_offsets), cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMemcpy(columns, cpu.columns, (nedges * sizeof(*columns)), cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMemcpy(sat, cpu.sat, (nnodes * sizeof(*sat)), cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMemcpy(bias, cpu.bias, (nnodes * sizeof(*bias)), cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMemcpy(value, cpu.value, (nnodes * sizeof(*value)), cudaMemcpyHostToDevice));

    return true;
  }

  bool to_cpu(CSRGraph &cpu, bool alloc = false)
  {
    if(alloc)
      {
	cpu.nnodes = nnodes;
	cpu.nedges = nedges;

	assert(cpu.alloc());

      }
    assert(nnodes == cpu.nnodes);
    assert(nedges == cpu.nedges);
        
    CUDA_SAFE_CALL(cudaMemcpy(cpu.row_offsets, row_offsets, (nnodes + 1) * sizeof(*row_offsets), cudaMemcpyDeviceToHost));

    CUDA_SAFE_CALL(cudaMemcpy(cpu.columns, columns, nedges * sizeof(*columns), cudaMemcpyDeviceToHost));

    CUDA_SAFE_CALL(cudaMemcpy(cpu.sat, sat, (nnodes * sizeof(*sat)), cudaMemcpyDeviceToHost));

    CUDA_SAFE_CALL(cudaMemcpy(cpu.bias, bias, (nnodes * sizeof(*bias)), cudaMemcpyDeviceToHost));

    CUDA_SAFE_CALL(cudaMemcpy(cpu.value, value, (nnodes * sizeof(*value)), cudaMemcpyDeviceToHost));


    return true;
  }

  __device__
    int degree(const int node) const
  {
    return row_offsets[node + 1] - row_offsets[node];
  } 
};

struct Edge {
  int nedges;
  int *src;
  int *dst;  
  bool *bar;
  float *eta;
  float *pi_0;
  float *pi_S;
  float *pi_U;

  bool alloc()
  {
    assert(nedges > 0);
    
    src = (int*) calloc(nedges, sizeof(*src));
    dst = (int*) calloc(nedges, sizeof(*dst));
    bar = (bool*) calloc(nedges, sizeof(*bar));
    eta = (float*) calloc(nedges, sizeof(*eta));
    pi_0 = (float*) calloc(nedges, sizeof(*pi_0));
    pi_S = (float*) calloc(nedges, sizeof(*pi_S));
    pi_U = (float*) calloc(nedges, sizeof(*pi_U));

    return (src && dst && bar && eta && pi_0 && pi_S && pi_U);
  }
};

struct GPUEdge : Edge {
  bool alloc()
  {
    assert(nedges > 0);
    
    CUDA_SAFE_CALL(cudaMalloc(&src, (nedges) * (sizeof(*src))));
    CUDA_SAFE_CALL(cudaMalloc(&dst, (nedges) * (sizeof(*dst))));
    CUDA_SAFE_CALL(cudaMalloc(&bar, (nedges) * (sizeof(*bar))));
    CUDA_SAFE_CALL(cudaMalloc(&eta, (nedges) * (sizeof(*eta))));
    CUDA_SAFE_CALL(cudaMalloc(&pi_0, (nedges) * (sizeof(*pi_0))));
    CUDA_SAFE_CALL(cudaMalloc(&pi_S, (nedges) * (sizeof(*pi_S))));
    CUDA_SAFE_CALL(cudaMalloc(&pi_U, (nedges) * (sizeof(*pi_U))));
 
    return (src && dst && bar && eta && pi_0 && pi_S && pi_U);
  }

  bool from_cpu(Edge &cpu)
  {
    nedges = cpu.nedges;
    
    assert(alloc());
    
    CUDA_SAFE_CALL(cudaMemcpy(src, cpu.src, (nedges) * sizeof(*src), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(dst, cpu.dst, (nedges) * sizeof(*dst), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(bar, cpu.bar, (nedges) * sizeof(*bar), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(eta, cpu.eta, (nedges) * sizeof(*eta), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(pi_0, cpu.pi_0, (nedges) * sizeof(*pi_0), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(pi_S, cpu.pi_S, (nedges) * sizeof(*pi_S), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(pi_U, cpu.pi_U, (nedges) * sizeof(*pi_U), cudaMemcpyHostToDevice));

    return true;
  }

  bool to_cpu(Edge &cpu, bool alloc = false)
  {
    if(alloc)
      {
	cpu.nedges = nedges;

	assert(cpu.alloc());
      }

    assert(nedges == cpu.nedges);

    CUDA_SAFE_CALL(cudaMemcpy(cpu.src, src, (nedges) * sizeof(*src), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(cpu.dst, dst, (nedges) * sizeof(*dst), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(cpu.bar, bar, (nedges) * sizeof(*bar), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(cpu.eta, eta, (nedges) * sizeof(*eta), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(cpu.pi_0, pi_0, (nedges) * sizeof(*pi_0), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(cpu.pi_S, pi_S, (nedges) * sizeof(*pi_S), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(cpu.pi_U, pi_U, (nedges) * sizeof(*pi_U), cudaMemcpyDeviceToHost));
        
    return true;
  }
};
