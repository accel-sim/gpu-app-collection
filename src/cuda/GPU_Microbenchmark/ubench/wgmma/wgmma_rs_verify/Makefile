SRC = wgmma_rs_verify.cu
EXE = wgmma_rs_verify

INCLUDE += -I../lat_gmma/include -I$(GPUAPPS_ROOT)/src/cuda/cutlass-bench/include -I./

ARCH ?= sm_90a
CUDA_CPPFLAGS=
HOPPER_CUDA_CPPFLAGS=$(foreach arch,$(ARCH),-gencode=arch=compute_$(subst sm_,,$(arch)),code=$(arch) -gencode=arch=compute_$(subst sm_,,$(arch)),code=compute_$(subst sm_,,$(arch)))

NVCC_FLAGS := $(HOPPER_CUDA_CPPFLAGS) -std=c++17

include ../../../common/common.mk
