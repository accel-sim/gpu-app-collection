NVCC?=nvcc
CXX?=g++

all: bulk_test tensor_test

ARCH?=sm_90a sm_100a sm_101 sm_120 
GENCODE_FLAGS=$(foreach arch,$(ARCH),-gencode arch=compute_$(subst sm_,,$(arch)),code=$(arch))


bulk_test: tma_bulk_test.cu
	nvcc $(GENCODE_FLAGS) -ccbin=$(CXX) -O3 tma_bulk_test.cu -o tma_bulk_test

tensor_test: tma_tensor_test.cu
	nvcc $(GENCODE_FLAGS) -ccbin=$(CXX) -O3 tma_tensor_test.cu -o tma_tensor_test

clean:
	rm -f tma_bulk_test tma_tensor_test
