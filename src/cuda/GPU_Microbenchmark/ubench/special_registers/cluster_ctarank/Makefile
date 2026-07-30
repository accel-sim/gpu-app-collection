BASE_DIR := $(shell pwd)
BIN_DIR  := $(BASE_DIR)/../../../bin/

ALL_BENCHES := cluster_ctarank cluster_ctarank_3d

# Thread-block clusters and %cluster_ctarank require SM_90 or newer.
ARCH ?= sm_120
CC   := nvcc

# Unset CUDA_CPPFLAGS from setup_environment; use explicit codegen here.
CUDA_CPPFLAGS=

NVCC_FLAGS := $(foreach arch,$(ARCH),\
    -gencode=arch=compute_$(subst sm_,,$(arch)),code=$(arch) \
    -gencode=arch=compute_$(subst sm_,,$(arch)),code=compute_$(subst sm_,,$(arch))) \
    -std=c++17

.DEFAULT_GOAL := all

all: $(ALL_BENCHES)

%.a: %.cu
	$(CC) $(NVCC_FLAGS) $(CUDA_CPPFLAGS) --lib $< -o $@

$(ALL_BENCHES): %: %.a
	$(CC) $(NVCC_FLAGS) $< -o $@ -lcudart --cudart shared
	mkdir -p $(BIN_DIR)
	mv $@ $(BIN_DIR)

clean:
	rm -f *.o *.a *.ptx *.ptxas *.sass $(ALL_BENCHES)

.PHONY: all clean
