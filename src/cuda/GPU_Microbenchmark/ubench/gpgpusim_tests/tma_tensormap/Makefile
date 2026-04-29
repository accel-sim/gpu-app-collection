BASE_DIR := $(shell pwd)
BIN_DIR  := $(BASE_DIR)/../../../bin

CC  := nvcc
EXE := gpgpusim_tma_tensor_benchmark
SRC := tma_tensor_benchmark.cu

ifneq ($(filter 13 14 15 16,$(CUDA_VERSION_MAJOR)),)
ARCH ?= sm_90a sm_100a sm_110 sm_120
else
ARCH ?= sm_90a sm_100a sm_101 sm_120
endif

CUDA_CPPFLAGS =
NVCC_FLAGS := $(foreach arch,$(ARCH),-gencode=arch=compute_$(subst sm_,,$(arch)),code=$(arch) -gencode=arch=compute_$(subst sm_,,$(arch)),code=compute_$(subst sm_,,$(arch))) -std=c++17

.DEFAULT_GOAL := all

all: $(BIN_DIR)/$(EXE)

$(BIN_DIR):
	mkdir -p $@

$(EXE).a: $(SRC)
	$(CC) $(NVCC_FLAGS) $(CUDA_CPPFLAGS) --lib $< -o $@

$(BIN_DIR)/$(EXE): $(EXE).a | $(BIN_DIR)
	$(CC) $(NVCC_FLAGS) $< -o $@ -lcudart --cudart shared

run-fixtures: $(BIN_DIR)/$(EXE)
	set -e; \
	tmp_dir="$$(mktemp -d)"; \
	trap 'rm -rf "$$tmp_dir"' EXIT; \
	for expected in expected/tma_tensor_test_*.txt; do \
		base="$$(basename "$$expected" .txt)"; \
		spec="$${base#tma_tensor_test_}"; \
		width="$${spec##*_}"; \
		spec="$${spec%_*}"; \
		height="$${spec##*_}"; \
		opcode="$${spec%_*}"; \
		( cd "$$tmp_dir" && "$(BIN_DIR)/$(EXE)" -w "$$width" -h "$$height" -o "$$opcode" ); \
		actual="$$tmp_dir/$$(basename "$$expected")"; \
		cmp -s "$$expected" "$$actual" || diff -u "$$expected" "$$actual"; \
	done

clean:
	rm -f *.o *.a *.ptx *.ptxas *.sass $(EXE)
	rm -f $(BIN_DIR)/$(EXE)

.PHONY: all clean run-fixtures
