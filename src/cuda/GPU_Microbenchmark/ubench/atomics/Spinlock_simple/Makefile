SRC = spinlock_simple.cu

EXE = spinlock_simple

# Spinlock_simple is only supported on SM_70 and above
# Check https://arxiv.org/abs/2407.02944
GENCODE_SM70 ?= -gencode=arch=compute_70,code=\"sm_70,compute_70\"
GENCODE_SM75 ?= -gencode=arch=compute_75,code=\"sm_75,compute_75\"

NVCC_FLAGS := $(GENCODE_SM70) $(GENCODE_SM75)

include ../../../common/common.mk
