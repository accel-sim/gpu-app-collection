EXECUTABLE := recursiveGaussian_hpc
CUFILES := recursiveGaussian_cuda.cu
CCFILES := recursiveGaussian.cpp
OMIT_CUTIL_LIB := 1
OMIT_SHRUTIL_LIB := 1
SM_VERSIONS := 75 80 90

# Build in benchmark mode without OpenGL
NVCCFLAGS += -DNO_OPENGL
CXXFLAGS += -DNO_OPENGL

include ../../../common/common.mk
