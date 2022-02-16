#!/usr/bin/make

# CUDA compiler
NVCC = nvcc
NVCC_FLAGS = -g -Wno-deprecated-gpu-targets

default:
	$(NVCC) $(NVCC_FLAGS) $(KERNEL_DIM) backprop.cu -lcudart -o backprop

prefetch:
	$(NVCC) $(NVCC_FLAGS) $(KERNEL_DIM) -DPREF backprop.cu -lcudart -o backprop

clean: $(SRC)
	rm -f backprop *.txt gpgpusim_power_report* _cuobjdump_*
