CPPSRC = ../graph_parser/util.cpp ../graph_parser/parse.cpp
CUSRC =
BASEEXE = sssp
VARIANT ?= CSR
# CUDA source files (compiled with cudacc)
ifeq ($(VARIANT),CSR)
    EXECUTABLE = $(BINDIR)/$(BINSUBDIR)/$(BASEEXE)
    CUSRC += sssp_csr.cu
else ifeq ($(VARIANT),ELL)
    EXECUTABLE = $(BINDIR)/$(BINSUBDIR)/$(BASEEXE)_ell
    CUSRC += sssp_ell.cu
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
