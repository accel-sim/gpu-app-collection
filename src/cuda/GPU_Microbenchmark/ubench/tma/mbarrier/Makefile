SRC = mbarrier.cu

EXE = mbarrier

ARCH?=sm_80 sm_90a sm_100a sm_101 sm_120 
# Unset the CUDA_CPPFLAGS which is set based on CUDA version
CUDA_CPPFLAGS=$(foreach arch,$(ARCH),-gencode arch=compute_$(subst sm_,,$(arch)),code=$(arch))

NVCC_FLAGS := $(GENCODE_FLAGS)

include ../../../common/common.mk
