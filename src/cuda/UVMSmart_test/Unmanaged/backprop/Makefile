#!/usr/bin/make

# C compiler
CC = gcc
CC_FLAGS = -g  -O2

# CUDA compiler
NVCC = nvcc
NVCC_FLAGS = -g -Wno-deprecated-gpu-targets

backprop: backprop.o facetrain.o imagenet.o backprop_cuda.o 
	$(NVCC) $(NVCC_FLAGS) backprop.o facetrain.o imagenet.o backprop_cuda.o -lcudart -lm -o backprop

%.o: %.[ch]
	$(CC) $(CC_FLAGS) $< -c

facetrain.o: facetrain.c backprop.h
	$(CC) $(CC_FLAGS) facetrain.c -c
	
backprop.o: backprop.c backprop.h
	$(CC) $(CC_FLAGS) backprop.c -c

backprop_cuda.o: backprop_cuda.cu backprop.h
	$(NVCC) $(NVCC_FLAGS) -c backprop_cuda.cu

imagenet.o: imagenet.c backprop.h
	$(CC) $(CC_FLAGS) imagenet.c -c


clean:
	rm -f *.o *~ backprop *.linkinfo *.txt gpgpusim_power_report* _cuobjdump_*
