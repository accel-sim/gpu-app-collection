BASE_DIR := $(shell pwd)
BIN_DIR  := $(BASE_DIR)/../../../bin/

# ld.global.L2::128B microbenchmark: issues `ld.global.L2::128B.u32` via
# explicit inline PTX (plus a plain `ld.global.u32` control path) and verifies
# from the host that the load still returns correct data — proving the
# .L2::128B prefetch hint is functionally inert while the underlying global
# load performs a real memory transaction. Single source -> single executable.
ALL_BENCHES := ld_global_l2_128b

# .L2::128B is a PTX .level::prefetch_size hint (PTX ISA 7.4+, SM_75+).
# Target Ampere and newer to match the rest of the modern ubench suite.
# Override on the command line, e.g. `make ARCH=sm_90a`, for a single GPU.
ARCH ?= sm_80 sm_90a sm_100a sm_101 sm_120
CC   := nvcc

# Unset CUDA_CPPFLAGS from setup_environment; use explicit codegen here.
CUDA_CPPFLAGS=

# Generate both SASS (code=sm_XX) and PTX (code=compute_XX) for each arch.
NVCC_FLAGS := $(foreach arch,$(ARCH),-gencode=arch=compute_$(subst sm_,,$(arch)),code=$(arch) -gencode=arch=compute_$(subst sm_,,$(arch)),code=compute_$(subst sm_,,$(arch))) -std=c++17

.DEFAULT_GOAL := all

all: $(ALL_BENCHES)

%.a: %.cu
	$(CC) $(NVCC_FLAGS) $(CUDA_CPPFLAGS) --lib $< -o $@

$(ALL_BENCHES): %: %.a
	$(CC) $(NVCC_FLAGS) $< -o $@ -lcudart --cudart shared
	mv $@ $(BIN_DIR)

clean:
	rm -f *.o *.a *.ptx *.ptxas *.sass $(ALL_BENCHES)

.PHONY: all clean
