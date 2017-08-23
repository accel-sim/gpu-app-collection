CPPSRC = ../graph_parser/util.cpp ../graph_parser/parse.cpp
CUSRC =
BASEEXE = color
VARIANT ?= MAX
# CUDA source files (compiled with cudacc)
ifeq ($(VARIANT),MAX)
    EXECUTABLE = $(BINDIR)/$(BINSUBDIR)/$(BASEEXE)_max
    CUSRC += coloring_max.cu
else ifeq ($(VARIANT),MAXMIN)
    EXECUTABLE = $(BINDIR)/$(BINSUBDIR)/$(BASEEXE)_maxmin
    CUSRC += coloring_maxmin.cu
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
