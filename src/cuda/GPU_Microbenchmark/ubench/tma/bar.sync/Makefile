BASE_DIR := $(shell pwd)
BIN_DIR  := $(BASE_DIR)/../../../bin/

# bar.sync microbenchmark: exercises the CTA-wide barrier (PTX `bar.sync 0;`,
# i.e. __syncthreads()) via explicit inline PTX and verifies the result on the
# host. The single source file bar_sync.cu builds a single executable.
ALL_BENCHES := bar_sync

# bar.sync is a universal barrier instruction (present on every SM generation),
# so we target a broad range while matching the architectures used by the rest
# of the tma/ suite. Override on the command line, e.g. `make ARCH=sm_90a`, to
# build for a single GPU.
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
