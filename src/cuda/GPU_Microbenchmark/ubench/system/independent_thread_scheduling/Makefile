BASE_DIR := $(shell pwd)
BIN_DIR  := $(BASE_DIR)/../../../bin/

SRC := independent_thread_scheduling.cu
EXE := independent_thread_scheduling

# Independent thread scheduling was introduced with Volta (SM 70). Keep PTX
# for simulation and SASS for validation on NVIDIA hardware.
GENCODE_SM70 ?= -gencode=arch=compute_70,code=\"sm_70,compute_70\"
GENCODE_SM90 ?= -gencode=arch=compute_90,code=\"sm_90,compute_90\"
NVCC_FLAGS := $(GENCODE_SM70) $(GENCODE_SM90) -std=c++17

include ../../../common/common.mk
