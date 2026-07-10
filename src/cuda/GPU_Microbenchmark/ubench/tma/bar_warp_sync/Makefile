BASE_DIR := $(shell pwd)
BIN_DIR  := $(BASE_DIR)/../../../bin/

# bar.warp.sync microbenchmark: issues the warp-level barrier (PTX
# `bar.warp.sync`, i.e. __syncwarp()) via explicit inline PTX and verifies an
# intra-warp exchange on the host. Single source -> single executable.
ALL_BENCHES := bar_warp_sync

# bar.warp.sync is a universal instruction, so target a broad range while
# matching the architectures used by the rest of the tma/ suite. Override on the
# command line, e.g. `make ARCH=sm_90a`, to build for a single GPU.
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
