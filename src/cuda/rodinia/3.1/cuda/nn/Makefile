include ../../common/make.config

LOCAL_CC = gcc -g -O3 -Wall
CC := $(CUDA_DIR)/bin/nvcc

all : nn hurricane_gen

clean :
	rm -rf *.o nn hurricane_gen

nn : nn_cuda.cu
	$(CC) -cuda nn_cuda.cu -lcudart  -D__DEVICE_EMULATION__
	$(CC) -o nn nn_cuda.cu -lcudart  -D__DEVICE_EMULATION__

hurricane_gen : hurricane_gen.c
	$(LOCAL_CC) -o $@ $< -lm

#data :
#	mkdir data
#	./gen_dataset.sh
