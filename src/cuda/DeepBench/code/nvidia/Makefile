
NVCC=nvcc
#ARCH?=sm_70

CUDA_PATH?=/usr/local/cuda
CUDA_LIB64=$(CUDA_PATH)/lib64
CUDNN_PATH?=/usr/local/cudnn
NCCL_PATH?=/usr/local/nccl
MPI_PATH?=/usr/local/openmpi
MPI_INCLUDE_PATH?=$(MPI_PATH)/include
BIN_DIR?=bin
MKDIR=mkdir -p
#BLAS
BLAS_LIBRARY?=cublas
BLAS_PATH?=$(CUDA_LIB64)
#CONV
CONV_LIBRARY?=cudnn
CONV_PATH?=$
KERNELS_DIR=../kernels/
PAD_KERNELS?=1
COMMA=,
#NVCC_ARCH_ARGS=$(foreach a,$(subst $(COMMA), ,$(ARCH)),--generate-code arch=$(patsubst sm_%,compute_%,$(a)),code=$(a))

.PHONY=all gemm conv rnn all_reduce nccl_single nccl_mpi clean

all: gemm conv rnn

gemm:
	$(MKDIR) $(BIN_DIR) 
	$(CUDA_PATH)/bin/$(NVCC) gemm_bench.cu -DUSE_TENSOR_CORES=0 -DPAD_KERNELS=$(PAD_KERNELS) -o $(BIN_DIR)/gemm_bench -I $(KERNELS_DIR) -I $(CUDA_PATH)/include -L $(BLAS_PATH) -l$(BLAS_LIBRARY) -L $(CUDA_LIB64) -lcurand $(CUDA_CPPFLAGS) -std=c++14
	$(CUDA_PATH)/bin/$(NVCC) gemm_bench.cu -DUSE_TENSOR_CORES=1 -DPAD_KERNELS=$(PAD_KERNELS) -o $(BIN_DIR)/gemm_bench-tencore -I $(KERNELS_DIR) -I $(CUDA_PATH)/include -L $(BLAS_PATH) -l$(BLAS_LIBRARY) -L $(CUDA_LIB64) -lcurand $(CUDA_CPPFLAGS) 

conv:
	$(MKDIR) $(BIN_DIR)
	$(CUDA_PATH)/bin/$(NVCC) conv_bench.cu -DUSE_TENSOR_CORES=0 -DPAD_KERNELS=$(PAD_KERNELS) -o $(BIN_DIR)/conv_bench -I $(KERNELS_DIR) -I $(CUDA_PATH)/include -I $(CUDNN_PATH)/include/ -L $(CUDNN_PATH)/lib64/ -L $(CUDA_LIB64) -lcurand -lcudnn $(CUDA_CPPFLAGS) -std=c++14
	$(CUDA_PATH)/bin/$(NVCC) conv_bench.cu -DUSE_TENSOR_CORES=1 -DPAD_KERNELS=$(PAD_KERNELS) -o $(BIN_DIR)/conv_bench-tencore -I $(KERNELS_DIR) -I $(CUDA_PATH)/include -I $(CUDNN_PATH)/include/ -L $(CUDNN_PATH)/lib64/ -L $(CUDA_LIB64) -lcurand -lcudnn $(CUDA_CPPFLAGS) 

rnn:
	$(MKDIR) $(BIN_DIR)
	$(CUDA_PATH)/bin/$(NVCC) rnn_bench.cu -DUSE_TENSOR_CORES=0 -o $(BIN_DIR)/rnn_bench -I $(KERNELS_DIR) -I $(CUDA_PATH)/include -I $(CUDNN_PATH)/include/ -L $(CUDNN_PATH)/lib64/ -L $(CUDA_LIB64) -lcurand -lcudnn $(CUDA_CPPFLAGS) -std=c++14
	$(CUDA_PATH)/bin/$(NVCC) rnn_bench.cu -DUSE_TENSOR_CORES=1 -o $(BIN_DIR)/rnn_bench-tencore -I $(KERNELS_DIR) -I $(CUDA_PATH)/include -I $(CUDNN_PATH)/include/ -L $(CUDNN_PATH)/lib64/ -L $(CUDA_LIB64) -lcurand -lcudnn $(CUDA_CPPFLAGS) 

all_reduce: nccl_single nccl_mpi

nccl_single:
	$(MKDIR) $(BIN_DIR)
	$(CUDA_PATH)/bin/$(NVCC) nccl_single_all_reduce.cu -o $(BIN_DIR)/nccl_single_all_reduce -I $(KERNELS_DIR) -I $(NCCL_PATH)/include/ -I $(CUDNN_PATH)/include/ -L $(NCCL_PATH)/lib/ -L $(CUDNN_PATH)/lib64 -lnccl -lcudart -lcurand $(CUDA_CPPFLAGS) 

nccl_mpi:
	$(MKDIR) $(BIN_DIR)
	$(CUDA_PATH)/bin/$(NVCC) nccl_mpi_all_reduce.cu -o $(BIN_DIR)/nccl_mpi_all_reduce -I $(KERNELS_DIR) -I $(NCCL_PATH)/include/ -I $(CUDNN_PATH)/include/ -I $(MPI_INCLUDE_PATH) -L $(NCCL_PATH)/lib/ -L $(CUDNN_PATH)/lib64 -L $(MPI_PATH)/lib -lnccl -lcurand -lcudart -lmpi $(CUDA_CPPFLAGS) 

sparse:
	$(MKDIR) $(BIN_DIR) 
	$(CUDA_PATH)/bin/$(NVCC) sparse_bench.cu -o $(BIN_DIR)/sparse_bench -I $(KERNELS_DIR) -I $(CUDA_PATH)/include -L $(BLAS_PATH) -l$(BLAS_LIBRARY) -L $(CUDA_LIB64) -lcurand -lcusparse $(CUDA_CPPFLAGS) 


clean:
	rm -rf $(BIN_DIR)

rebuild: clean all
