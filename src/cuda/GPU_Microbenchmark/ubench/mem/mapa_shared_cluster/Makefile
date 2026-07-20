BASE_DIR := $(shell pwd)
BIN_DIR  := $(BASE_DIR)/../../../bin/

# mapa.shared::cluster.u32 careful functional microbenchmark.
# Remaps a shared-window address onto a cluster CTA rank; Gates A/B/C
# verify rank sensitivity and remote store / local load semantics.
ALL_BENCHES := mapa_shared_cluster

# mapa.shared::cluster is Hopper+ PTX (PTX ISA 7.8+, SM_90 and above).
# Override on the command line, e.g. `make ARCH=sm_90a`, for a single GPU.
ARCH ?= sm_90a sm_100a sm_101 sm_120
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
