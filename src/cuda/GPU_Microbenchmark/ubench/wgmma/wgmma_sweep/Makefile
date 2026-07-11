SRC = wgmma_sweep.cu
EXE = wgmma_sweep

INCLUDE += -I../lat_gmma/include -I$(GPUAPPS_ROOT)/src/cuda/cutlass-bench/include -I./

ARCH ?= sm_90a
CUDA_CPPFLAGS=
HOPPER_CUDA_CPPFLAGS=$(foreach arch,$(ARCH),-gencode=arch=compute_$(subst sm_,,$(arch)),code=$(arch) -gencode=arch=compute_$(subst sm_,,$(arch)),code=compute_$(subst sm_,,$(arch)))

NVCC_FLAGS := $(HOPPER_CUDA_CPPFLAGS) -std=c++17

# Needs CuTe/CUTLASS headers (bundled lat_gmma/include or the cutlass-bench
# submodule). Skip gracefully when absent (e.g. CI containers).
CUTE_HDR := $(or $(wildcard ../lat_gmma/include/cute/tensor.hpp), \
             $(wildcard $(GPUAPPS_ROOT)/src/cuda/cutlass-bench/include/cute/tensor.hpp))
ifeq ($(CUTE_HDR),)
release clean:
	@echo "$(EXE): SKIPPED (CuTe/CUTLASS headers not found)"
else
include ../../../common/common.mk
endif
