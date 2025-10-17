SRC = tma_bulk.cu

EXE = tma_bulk

# TMA is supported on SM_90a and above
ARCH?=sm_90a sm_100a sm_101 sm_120 
# Unset the CUDA_CPPFLAGS which is set based on CUDA version
# but TMA is only supported on SM_90a and above
CUDA_CPPFLAGS=$(foreach arch,$(ARCH),-gencode arch=compute_$(subst sm_,,$(arch)),code=$(arch))

include ../../../common/common.mk
