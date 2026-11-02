SRC = ldgsts.cu

EXE = cp_async

NVCC_FLGAS = -Xptxas -dlcm=cg -Xptxas -dscm=wt

# ldgsts.cu uses cp.async, which ptxas rejects below sm_80
MIN_ARCH = 80

include ../../../common/common.mk
