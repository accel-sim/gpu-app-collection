CPPSRC = ../graph_parser/util.cpp parse.cpp
CUSRC =
BASEEXE = fw
VARIANT ?= DEFAULT
# CUDA source files (compiled with cudacc)
ifeq ($(VARIANT),DEFAULT)
    EXECUTABLE = $(BINDIR)/$(BINSUBDIR)/$(BASEEXE)
    CUSRC += Floyd-Warshall.cu
else ifeq ($(VARIANT),BLOCK)
    EXECUTABLE = $(BINDIR)/$(BINSUBDIR)/$(BASEEXE)_block
    CUSRC += Floyd-Warshall-Block.cu
endif
OPTS = -O3
CUOPTS = -arch sm_20
INC = -I$(NVIDIA_CUDA_SDK_LOCATION)/common/inc -I$(CUDAHOME)/include
LIBS = -L$(NVIDIA_CUDA_SDK_LOCATION)/lib -lcutil_x86_64

OBJS = $(notdir $(CPPSRC:.cpp=.o))

all: $(OBJS)
	nvcc $(NVOPTS) $(OPTS) $(CUOPTS) $(CUSRC) *.o $(INC) -o $(EXECUTABLE) $(LIBS)

$(OBJS):
	g++ -c $(CPPSRC) $(OPTS)

clean:
	rm -f *.o *.cu.cpp $(EXECUTABLE)

cleancucpp:
	rm -f *.cu.cpp
