GENCODE_SM50 :=
GENCODE_SM61 :=
GENCODE_SM30 :=
GENCODE_SM35 :=
GENCODE_SM60 :=
GENCODE_SM62 :=

SRC = tensor_bw_half.cu

EXE = tensor_bw_half

# <mma.h> only declares nvcuda::wmma for __CUDA_ARCH__ >= 700, so a gencode
# list that still reaches back to Maxwell/Pascal fails here with "wmma is not
# a class or namespace name". The GENCODE_SMxx assignments above predate
# CUDA_CPPFLAGS and never applied to it.
MIN_ARCH = 70

include ../../../common/common.mk
