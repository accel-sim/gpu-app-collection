#!/usr/bin/make

# CUDA compiler
NVCC = nvcc
NVCC_FLAGS = -g -Wno-deprecated-gpu-targets

default:
	$(NVCC) $(NVCC_FLAGS) $(KERNEL_DIM) srad.cu -lcudart -o srad

prefetch:
	$(NVCC) $(NVCC_FLAGS) $(KERNEL_DIM) -DPREF srad.cu -lcudart -o srad

clean: $(SRC)
	rm -f srad *.txt gpgpusim_power_report* _cuobjdump_*
