# Drop any architecture the installed toolkit will not accept, instead of
# failing the build. These ubenches pin an explicit ARCH list (TMA and GMMA
# need the accelerated targets), and those lists drift: CUDA 13 removed
# sm_101, for example. The accelerated "a" variants are matched on their
# base number, so sm_90a survives as long as compute_90 is supported.
#
# Include this after setting ARCH and before deriving NVCC_FLAGS from it.
NVCC_SUPPORTED_ARCH := $(shell nvcc --list-gpu-arch 2>/dev/null | sed -e 's/compute_//')
ifneq ($(strip $(NVCC_SUPPORTED_ARCH)),)
ARCH := $(foreach a,$(ARCH),$(if $(filter $(patsubst %a,%,$(patsubst sm_%,%,$(a))),$(NVCC_SUPPORTED_ARCH)),$(a)))
endif
