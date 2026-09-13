# Link straight into the shared bin dir under the final benchmark name when
# setup_environment has been sourced, and build in place otherwise. This
# replaces the post-build 'mv' the top-level Makefile used to do, which
# stranded binaries in the source tree whenever a sibling app failed.
RODINIA_OUT ?= $(if $(BINDIR),$(BINDIR)/$(BINSUBDIR),.)

include ../../common/make.config

LOCAL_CC = gcc -g -O3 -Wall
CC := $(CUDA_DIR)/bin/nvcc $(GENCODE_SM10) $(GENCODE_SM13) $(GENCODE_SM80) $(GENCODE_SM86) $(GENCODE_SM90) $(GENCODE_SM100) $(GENCODE_SM103) $(GENCODE_ARCH) $(GENCODE_SM20) $(GENCODE_SM30) $(GENCODE_SM35) $(GENCODE_SM50) $(GENCODE_SM60) $(GENCODE_SM62) $(GENCODE_SM70) $(GENCODE_SM75)

all : $(RODINIA_OUT)/nn-rodinia-3.1 hurricane_gen

clean :
	rm -rf *.o nn hurricane_gen

$(RODINIA_OUT)/nn-rodinia-3.1 : nn_cuda.cu
	$(CC) -cuda nn_cuda.cu -lcudart  -D__DEVICE_EMULATION__
	$(CC) -o $(RODINIA_OUT)/nn-rodinia-3.1 nn_cuda.cu -lcudart  -D__DEVICE_EMULATION__

hurricane_gen : hurricane_gen.c
	$(LOCAL_CC) -o $@ $< -lm

#data :
#	mkdir data
#	./gen_dataset.sh
