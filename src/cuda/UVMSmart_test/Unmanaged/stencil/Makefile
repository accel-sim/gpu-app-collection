#!/usr/bin/make

# C compiler
CC = gcc
CC_FLAGS = -g  -O2

# CUDA compiler
NVCC = nvcc
NVCC_FLAGS = -g -Wno-deprecated-gpu-targets

main: main.o file.o kernels.o
	$(NVCC) $(NVCC_FLAGS) main.o kernels.o file.o -lcudart -lm -o main

%.o: %.[ch]
	$(CC) $(CC_FLAGS) $< -c

main.o: main.cu
	$(NVCC) $(NVCC_FLAGS) -c main.cu

kernels.o: kernels.cu kernels.h
	$(NVCC) $(NVCC_FLAGS) -c kernels.cu

file.o: file.cc file.h
	$(CC) $(CC_FLAGS) file.cc -c


clean:
	rm -f *.o *~ main *.linkinfo *.txt gpgpusim_power_report* _cuobjdump_*
