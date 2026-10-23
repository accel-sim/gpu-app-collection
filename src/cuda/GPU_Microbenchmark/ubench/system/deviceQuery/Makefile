
SRC = deviceQuery.cpp

EXE = deviceQuery
# cuda-samples is cloned by a later target in the top-level Makefile, so it is
# not there yet when this builds. The only thing needed from it here is
# _ConvertSMVer2Cores, which the vendored sdk-compat headers also provide.
INCLUDE :=  -I../../../../cuda-samples/Common/ -I$(GPUAPPS_SDK_INC)
NVCC_FLGAS =

include ../../../common/common.mk
