# Link straight into the shared bin dir under the final benchmark name when
# setup_environment has been sourced, and build in place otherwise. This
# replaces the post-build 'mv' the top-level Makefile used to do, which
# stranded binaries in the source tree whenever a sibling app failed.
RODINIA_OUT ?= $(if $(BINDIR),$(BINDIR)/$(BINSUBDIR),.)

#makefile

include ../../common/make.config

CC := $(CUDA_DIR)/bin/nvcc $(GENCODE_SM10) $(GENCODE_SM13) $(GENCODE_SM80) $(GENCODE_SM86) $(GENCODE_SM90) $(GENCODE_SM100) $(GENCODE_SM103) $(GENCODE_ARCH) $(GENCODE_SM20) $(GENCODE_SM30) $(GENCODE_SM35) $(GENCODE_SM50) $(GENCODE_SM60) $(GENCODE_SM62) $(GENCODE_SM70) $(GENCODE_SM75)

INCLUDE := $(CUDA_DIR)/include

all: naive float

naive: ex_particle_CUDA_naive_seq.cu
	$(CC) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -g -lm -O3 -use_fast_math  ex_particle_CUDA_naive_seq.cu -o $(RODINIA_OUT)/particlefilter_naive-rodinia-3.1
	
float: ex_particle_CUDA_float_seq.cu
	$(CC) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -g -lm -O3 -use_fast_math ex_particle_CUDA_float_seq.cu -o $(RODINIA_OUT)/particlefilter_float-rodinia-3.1

clean:
	rm -f particlefilter_naive particlefilter_float
