SRC = mbarrier.cu

EXE = mbarrier

ARCH?=sm_80 sm_90a sm_100a sm_101 sm_120 
# Unset the CUDA_CPPFLAGS which is set based on CUDA version
CUDA_CPPFLAGS=

NVCC_FLAGS := $(foreach arch,$(ARCH),-gencode arch=compute_$(subst sm_,,$(arch)),code=$(arch)) -std=c++14

include ../../../common/common.mk
