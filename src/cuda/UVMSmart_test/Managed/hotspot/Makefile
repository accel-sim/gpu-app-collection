#!/usr/bin/make

# CUDA compiler
NVCC = nvcc
NVCC_FLAGS = -g -Wno-deprecated-gpu-targets

SRC = hotspot.cu

EXE = hotspot

default:
	$(NVCC) $(NVCC_FLAGS) $(KERNEL_DIM) hotspot.cu -lcudart -o hotspot

prefetch:
	$(NVCC) $(NVCC_FLAGS) $(KERNEL_DIM) -DPREF hotspot.cu -lcudart -o hotspot

clean:
	rm -f hotspot *.txt gpgpusim_power_report* _cuobjdump_*
