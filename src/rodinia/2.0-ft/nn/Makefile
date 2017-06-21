include ../../common/make.config

LOCAL_CC = gcc -g -O3 -Wall
CC := $(CUDA_DIR)/bin/nvcc

all : nn

clean :
	rm -rf *.o nn nn_cuda.cu.cpp hurricane_gen

nn :
	$(CC) -cuda nn_cuda.cu
	$(CC) -o nn nn_cuda.cu.cpp

hurricane_gen : hurricane_gen.c
	$(LOCAL_CC) -o $@ $< -lm
